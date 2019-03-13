## 元编程（Metaprogramming）

* 元编程将计算在编译期完成，避免了运行期计算的开销

```cpp
#include <type_traits>

namespace jc {

template <int N, int... Ns>
struct max;

template <int N>
struct max<N> : std::integral_constant<int, N> {};

template <int N1, int N2, int... Ns>
struct max<N1, N2, Ns...>
    : std::integral_constant<int, (N1 < N2) ? max<N2, Ns...>::value
                                            : max<N1, Ns...>::value> {};

template <int... Ns>
inline constexpr auto max_v = max<Ns...>::value;

}  // namespace jc

static_assert(jc::max_v<3, 2, 1, 5, 4> == 5);

int main() {}
```

* 模板元编程通常使用偏特化和递归实现，由于编译期需要实例化代码，如果递归层次过深，会带来代码体积膨胀的问题

```cpp
#include <type_traits>

namespace jc {

template <int N, int L = 1, int R = N>
struct sqrt {
  static constexpr auto M = L + (R - L) / 2;
  static constexpr auto T = N / M;
  static constexpr auto value =  // 避免递归实例化所有分支
      std::conditional_t<(T < M), sqrt<N, L, M>, sqrt<N, M + 1, R>>::value;
};

template <int N, int M>
struct sqrt<N, M, M> {
  static constexpr auto value = M - 1;
};

template <int N>
inline constexpr auto sqrt_v = sqrt<N, 1, N>::value;

}  // namespace jc

static_assert(jc::sqrt_v<10000> == 100);

int main() {}
```

* C++14 支持 constexpr 函数，简化了实现并且没有递归实例化的代码膨胀问题

```cpp
namespace jc {

template <int N>
constexpr int sqrt() {
  if constexpr (N <= 1) {
    return N;
  }
  int l = 1;
  int r = N;
  while (l < r) {
    int m = l + (r - l) / 2;
    int t = N / m;
    if (m == t) {
      return m;
    } else if (m > t) {
      r = m;
    } else {
      l = m + 1;
    }
  }
  return l - 1;
}

}  // namespace jc

static_assert(jc::sqrt<10000>() == 100);

int main() {}
```

* Typelist

```cpp
#include <type_traits>

namespace jc {

template <typename...>
struct typelist {};

template <typename List>
struct front;

template <typename Head, typename... Tail>
struct front<typelist<Head, Tail...>> {
  using type = Head;
};

template <typename List>
using front_t = typename front<List>::type;

// pop_front_t
template <typename List>
struct pop_front;

template <typename Head, typename... Tail>
struct pop_front<typelist<Head, Tail...>> {
  using type = typelist<Tail...>;
};

template <typename List>
using pop_front_t = typename pop_front<List>::type;

// push_front_t
template <typename List, typename NewElement>
struct push_front;

template <typename... Elements, typename NewElement>
struct push_front<typelist<Elements...>, NewElement> {
  using type = typelist<NewElement, Elements...>;
};

template <typename List, typename NewElement>
using push_front_t = typename push_front<List, NewElement>::type;

// nth_element_t
template <typename List, std::size_t N>
struct nth_element : nth_element<pop_front_t<List>, N - 1> {};

template <typename List>
struct nth_element<List, 0> : front<List> {};

template <typename List, std::size_t N>
using nth_element_t = typename nth_element<List, N>::type;

// is_empty
template <typename T>
struct is_empty {
  static constexpr bool value = false;
};

template <>
struct is_empty<typelist<>> {
  static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_empty_v = is_empty<T>::value;

// push_back_t
template <typename List, typename NewElement, bool = is_empty_v<List>>
struct push_back_impl;

template <typename List, typename NewElement>
struct push_back_impl<List, NewElement, false> {
 private:
  using head = front_t<List>;
  using tail = pop_front_t<List>;
  using new_tail = typename push_back_impl<tail, NewElement>::type;

 public:
  using type = push_front_t<new_tail, head>;
};

template <typename List, typename NewElement>
struct push_back_impl<List, NewElement, true> {
  using type = push_front_t<List, NewElement>;
};

template <typename List, typename NewElement>
struct push_back : push_back_impl<List, NewElement> {};

/*
 * template <typename List, typename NewElement>
 * struct push_back;
 *
 * template <typename... Elements, typename NewElement>
 * struct push_back<typelist<Elements...>, NewElement> {
 * using type = typelist<Elements..., NewElement>;
 * };
 */

template <typename List, typename NewElement>
using push_back_t = typename push_back<List, NewElement>::type;

// reverse_t
template <typename List, bool Empty = is_empty_v<List>>
struct reverse;

template <typename List>
using reverse_t = typename reverse<List>::type;

template <typename List>
struct reverse<List, false>
    : push_back<reverse_t<pop_front_t<List>>, front_t<List>> {};

template <typename List>
struct reverse<List, true> {
  using type = List;
};

// pop_back_t
template <typename List>
struct pop_back {
  using type = reverse_t<pop_front_t<reverse_t<List>>>;
};

template <typename List>
using pop_back_t = typename pop_back<List>::type;

// largest_type_t
template <typename List, bool = is_empty_v<List>>
struct largest_type;

template <typename List>
struct largest_type<List, false> {
 private:
  using contender = front_t<List>;
  using best = typename largest_type<pop_front_t<List>>::type;

 public:
  using type =
      std::conditional_t<(sizeof(contender) >= sizeof(best)), contender, best>;
};

template <typename List>
struct largest_type<List, true> {
  using type = char;
};

template <typename List>
using largest_type_t = typename largest_type<List>::type;

// transform_t
template <typename List, template <typename T> class MetaFun,
          bool = is_empty_v<List>>
struct transform;

/*
 * template <typename List, template <typename T> class MetaFun>
 * struct transform<List, MetaFun, false>
 *     : push_front<typename transform<pop_front_t<List>, MetaFun>::type,
 *                  typename MetaFun<front_t<List>>::type> {};
 */

template <typename... Elements, template <typename T> class MetaFun>
struct transform<typelist<Elements...>, MetaFun, false> {
  using type = typelist<typename MetaFun<Elements>::type...>;
};

template <typename List, template <typename T> class MetaFun>
struct transform<List, MetaFun, true> {
  using type = List;
};

template <typename List, template <typename T> class MetaFun>
using transform_t = typename transform<List, MetaFun>::type;

// accumulate_t
template <typename List, template <typename T, typename U> class F,
          typename Init, bool = is_empty_v<List>>
struct accumulate;

template <typename List, template <typename T, typename U> class MetaFun,
          typename Init>
struct accumulate<List, MetaFun, Init, false>
    : accumulate<pop_front_t<List>, MetaFun,
                 typename MetaFun<Init, front_t<List>>::type> {};

template <typename List, template <typename T, typename U> class MetaFun,
          typename Init>
struct accumulate<List, MetaFun, Init, true> {
  using type = Init;
};

template <typename List, template <typename T, typename U> class MetaFun,
          typename Init>
using accumulate_t = typename accumulate<List, MetaFun, Init>::type;

// insert_sorted_t
template <typename T>
struct type_identity {
  using type = T;
};

template <typename List, typename Element,
          template <typename T, typename U> class Compare,
          bool = is_empty_v<List>>
struct insert_sorted;

template <typename List, typename Element,
          template <typename T, typename U> class Compare>
struct insert_sorted<List, Element, Compare, false> {
 private:
  // compute the tail of the resulting list:
  using new_tail = typename std::conditional_t<
      Compare<Element, front_t<List>>::value, type_identity<List>,
      insert_sorted<pop_front_t<List>, Element, Compare>>::type;

  // compute the head of the resulting list:
  using new_head = std::conditional_t<Compare<Element, front_t<List>>::value,
                                      Element, front_t<List>>;

 public:
  using type = push_front_t<new_tail, new_head>;
};

template <typename List, typename Element,
          template <typename T, typename U> class Compare>
struct insert_sorted<List, Element, Compare, true> : push_front<List, Element> {
};

template <typename List, typename Element,
          template <typename T, typename U> class Compare>
using insert_sorted_t = typename insert_sorted<List, Element, Compare>::type;

// insertion_sort_t
template <typename List, template <typename T, typename U> class Compare,
          bool = is_empty_v<List>>
struct insertion_sort;

template <typename List, template <typename T, typename U> class Compare>
using insertion_sort_t = typename insertion_sort<List, Compare>::type;

template <typename List, template <typename T, typename U> class Compare>
struct insertion_sort<List, Compare, false>
    : insert_sorted<insertion_sort_t<pop_front_t<List>, Compare>, front_t<List>,
                    Compare> {};

template <typename List, template <typename T, typename U> class Compare>
struct insertion_sort<List, Compare, true> {
  using type = List;
};

// multiply_t
template <typename T, typename U>
struct multiply;

template <typename T, T Value1, T Value2>
struct multiply<std::integral_constant<T, Value1>,
                std::integral_constant<T, Value2>> {
  using type = std::integral_constant<T, Value1 * Value2>;
};

template <typename T, typename U>
using multiply_t = typename multiply<T, U>::type;

// for std::index_sequence
template <std::size_t... Values>
struct is_empty<std::index_sequence<Values...>> {
  static constexpr bool value = sizeof...(Values) == 0;
};

template <std::size_t Head, std::size_t... Tail>
struct front<std::index_sequence<Head, Tail...>> {
  using type = std::integral_constant<std::size_t, Head>;
  static constexpr std::size_t value = Head;
};

template <std::size_t Head, std::size_t... Tail>
struct pop_front<std::index_sequence<Head, Tail...>> {
  using type = std::index_sequence<Tail...>;
};

template <std::size_t... Values, std::size_t New>
struct push_front<std::index_sequence<Values...>,
                  std::integral_constant<std::size_t, New>> {
  using type = std::index_sequence<New, Values...>;
};

template <std::size_t... Values, std::size_t New>
struct push_back<std::index_sequence<Values...>,
                 std::integral_constant<std::size_t, New>> {
  using type = std::index_sequence<Values..., New>;
};

// select_t
template <typename Types, typename Indices>
struct select;

template <typename Types, std::size_t... Indices>
struct select<Types, std::index_sequence<Indices...>> {
  using type = typelist<nth_element_t<Types, Indices>...>;
};

template <typename Types, typename Indices>
using select_t = typename select<Types, Indices>::type;

// Cons
struct Nil {};

template <typename Head, typename Tail = Nil>
struct Cons {
  using head = Head;
  using tail = Tail;
};

template <typename List>
struct front {
  using type = typename List::head;
};

template <typename List, typename Element>
struct push_front {
  using type = Cons<Element, List>;
};

template <typename List>
struct pop_front {
  using type = typename List::tail;
};

template <>
struct is_empty<Nil> {
  static constexpr bool value = true;
};

}  // namespace jc

namespace jc::test {

template <typename T, typename U>
struct smaller {
  static constexpr bool value = sizeof(T) < sizeof(U);
};

template <typename T, typename U>
struct less;

template <typename T, T M, T N>
struct less<std::integral_constant<T, M>, std::integral_constant<T, N>> {
  static constexpr bool value = M < N;
};

template <typename T, T... Values>
using integral_constant_typelist =
    typelist<std::integral_constant<T, Values>...>;

static_assert(std::is_same_v<integral_constant_typelist<int, 2, 3, 5>,
                             typelist<std::integral_constant<int, 2>,
                                      std::integral_constant<int, 3>,
                                      std::integral_constant<int, 5>>>);
static_assert(is_empty_v<typelist<>>);
using T1 = push_front_t<typelist<>, char>;
static_assert(std::is_same_v<T1, typelist<char>>);
static_assert(std::is_same_v<front_t<T1>, char>);
using T2 = push_front_t<T1, double>;
static_assert(std::is_same_v<T2, typelist<double, char>>);
static_assert(std::is_same_v<front_t<T2>, double>);
static_assert(std::is_same_v<pop_front_t<T2>, typelist<char>>);
using T3 = push_back_t<T2, int*>;
static_assert(std::is_same_v<T3, typelist<double, char, int*>>);
static_assert(std::is_same_v<nth_element_t<T3, 0>, double>);
static_assert(std::is_same_v<nth_element_t<T3, 1>, char>);
static_assert(std::is_same_v<nth_element_t<T3, 2>, int*>);
static_assert(std::is_same_v<reverse_t<T3>, typelist<int*, char, double>>);
static_assert(std::is_same_v<pop_back_t<T3>, typelist<double, char>>);
static_assert(std::is_same_v<largest_type_t<T3>, double>);
static_assert(std::is_same_v<transform_t<T3, std::add_const>,
                             typelist<const double, const char, int* const>>);
static_assert(std::is_same_v<accumulate_t<T3, push_front, typelist<>>,
                             typelist<int*, char, double>>);
static_assert(std::is_same_v<insertion_sort_t<T3, smaller>,
                             typelist<char, int*, double>>);
static_assert(accumulate_t<integral_constant_typelist<int, 2, 3, 5>, multiply,
                           std::integral_constant<int, 1>>::value == 30);

static_assert(
    std::is_same_v<insertion_sort_t<std::index_sequence<2, 3, 0, 1>, less>,
                   std::index_sequence<0, 1, 2, 3>>);
static_assert(is_empty_v<std::index_sequence<>>);
static_assert(std::is_same_v<std::make_index_sequence<4>,
                             std::index_sequence<0, 1, 2, 3>>);
static_assert(front<std::make_index_sequence<4>>::value == 0);
static_assert(std::is_same_v<front_t<std::make_index_sequence<4>>,
                             std::integral_constant<std::size_t, 0>>);
static_assert(std::is_same_v<pop_front_t<std::make_index_sequence<4>>,
                             std::index_sequence<1, 2, 3>>);
static_assert(
    std::is_same_v<push_front_t<std::make_index_sequence<4>,
                                std::integral_constant<std::size_t, 4>>,
                   std::index_sequence<4, 0, 1, 2, 3>>);
static_assert(
    std::is_same_v<push_back_t<std::make_index_sequence<4>,
                               std::integral_constant<std::size_t, 4>>,
                   std::index_sequence<0, 1, 2, 3, 4>>);
static_assert(std::is_same_v<select_t<typelist<bool, char, int, double>,
                                      std::index_sequence<2, 3, 0, 1>>,
                             typelist<int, double, bool, char>>);

using ConsList = Cons<int, Cons<char, Cons<short, Cons<double>>>>;
static_assert(is_empty_v<Nil>);
static_assert(std::is_same_v<
              push_front_t<ConsList, bool>,
              Cons<bool, Cons<int, Cons<char, Cons<short, Cons<double>>>>>>);
static_assert(std::is_same_v<pop_front_t<ConsList>,
                             Cons<char, Cons<short, Cons<double>>>>);
static_assert(std::is_same_v<front_t<ConsList>, int>);
static_assert(std::is_same_v<insertion_sort_t<ConsList, smaller>,
                             Cons<char, Cons<short, Cons<int, Cons<double>>>>>);

}  // namespace jc::test

int main() {}
```

## 循环展开（Loop Unrolling）

* 在一些机器上，for 循环的汇编将产生分支指令

```cpp
#include <array>
#include <cassert>

namespace jc {

template <typename T, std::size_t N>
auto dot_product(const std::array<T, N>& lhs, const std::array<T, N>& rhs) {
  T res{};
  for (std::size_t i = 0; i < N; ++i) {
    res += lhs[i] * rhs[i];
  }
  return res;
}

}  // namespace jc

int main() {
  std::array<int, 3> a{1, 2, 3};
  std::array<int, 3> b{4, 5, 6};
  assert(jc::dot_product(a, b) == 32);
}
```

* 循环展开是一种牺牲体积加快程序执行速度的方法，现代编译器会优化循环为目标平台最高效形式。使用元编程可以展开循环，虽然已经没有必要，但还是给出实现

```cpp
#include <array>
#include <cassert>

namespace jc {

template <typename T, std::size_t N>
struct dot_product_impl {
  static T value(const T* lhs, const T* rhs) {
    return *lhs * *rhs + dot_product_impl<T, N - 1>::value(lhs + 1, rhs + 1);
  }
};

template <typename T>
struct dot_product_impl<T, 0> {
  static T value(const T*, const T*) { return T{}; }
};

template <typename T, std::size_t N>
auto dot_product(const std::array<T, N>& lhs, const std::array<T, N>& rhs) {
  return dot_product_impl<T, N>::value(&*std::begin(lhs), &*std::begin(rhs));
}

}  // namespace jc

int main() {
  std::array<int, 3> a{1, 2, 3};
  std::array<int, 3> b{4, 5, 6};
  assert(jc::dot_product(a, b) == 32);
}
```

## [Unit Type](https://en.wikipedia.org/wiki/Unit_type)

* [std::ratio](https://en.cppreference.com/w/cpp/numeric/ratio/ratio)

```cpp
#include <cassert>
#include <cmath>
#include <type_traits>

namespace jc {

template <int N, int D = 1>
struct ratio {
  static constexpr int num = N;
  static constexpr int den = D;
  using type = ratio<num, den>;
};

template <typename R1, typename R2>
struct ratio_add_impl {
 private:
  static constexpr int den = R1::den * R2::den;
  static constexpr int num = R1::num * R2::den + R2::num * R1::den;

 public:
  using type = ratio<num, den>;
};

template <typename R1, typename R2>
using ratio_add = typename ratio_add_impl<R1, R2>::type;

template <typename T, typename U = ratio<1>>
class duration {
 public:
  using rep = T;
  using period = typename U::type;

 public:
  constexpr duration(rep r = 0) : r_(r) {}
  constexpr rep count() const { return r_; }

 private:
  rep r_;
};

template <typename T1, typename U1, typename T2, typename U2>
constexpr auto operator+(const duration<T1, U1>& lhs,
                         const duration<T2, U2>& rhs) {
  using CommonType = ratio<1, ratio_add<U1, U2>::den>;
  auto res =
      (lhs.count() * U1::num / U1::den + rhs.count() * U2::num / U2::den) *
      CommonType::den;
  return duration<decltype(res), CommonType>{res};
}

}  // namespace jc

int main() {
  constexpr auto a = jc::duration<double, jc::ratio<1, 1000>>(10);  // 10 ms
  constexpr auto b = jc::duration<double, jc::ratio<1, 3>>(7.5);    // 2.5 s
  constexpr auto c = a + b;  // 10 * 3 + 7.5 * 1000 = 7530 * 1/3000 s
  assert(std::abs(c.count() - 7530) < 1e-6);
  static_assert(std::is_same_v<std::decay_t<decltype(c)>,
                               jc::duration<double, jc::ratio<1, 3000>>>);
  static_assert(decltype(c)::period::num == 1);
  static_assert(decltype(c)::period::den == 3000);
}
```
