//===------- dlwrap.h - Convenience wrapper around dlopen/dlsym  -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The openmp plugins depend on extern libraries. These can be used via:
//  - bitcode file statically linked
//  - (relocatable) object file statically linked
//  - static library
//  - dynamic library, linked at build time
//  - dynamic library, loaded at application run time by dlopen
//
// This file factors out most boilerplate for using a dlopened library.
// - Function symbols are generated that are statically linked against
// - The dlopen can be done implicitly when initializing the library
// - dlsym lookups are done once and cached
// - The abstraction is very thin to permit varied uses of the library
//
// Given int foo(char, double, void*);, writing DLWRAP(foo, 3) will expand to:
// int foo(char x0, double x1, void* x2) {
//   constexpr size_t index = id();
//   void * dlsymResult = pointer(index);
//   return ((int (*)(char, double, void*))dlsymResult)(x0, x1, x2);
// }
//
// Multiple calls to DLWRAP(symbol_name, arity) with bespoke
// initialization code that can use the thin abstraction:
// namespace dlwrap {
//   static size_t size();
//   static const char *symbol(size_t);
//   static void **pointer(size_t);
// }
// will compile to an object file that only exposes the symbols that the
// dynamic library would do, with the right function types.
//
//===----------------------------------------------------------------------===//

#ifndef DLWRAP_H_INCLUDED
#define DLWRAP_H_INCLUDED

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>

// Where symbol is a function, these expand to some book keeping and an
// implementation of that function
#define DLWRAP(SYMBOL, ARITY) DLWRAP_IMPL(SYMBOL, ARITY)
#define DLWRAP_INTERNAL(SYMBOL, ARITY) DLWRAP_INTERNAL_IMPL(SYMBOL, ARITY)

// For example, given a prototype:
// int foo(char, double);
//
// DLWRAP(foo, 2) expands to:
//
// namespace dlwrap {
// struct foo_Trait : public dlwrap::trait<decltype(&foo)> {
//   using T = dlwrap::trait<decltype(&foo)>;
//   static T::FunctionType get() {
//     constexpr size_t Index = getIndex();
//     void *P = *dlwrap::pointer(Index);
//     return reinterpret_cast<T::FunctionType>(P);
//   }
// };
// }
// int foo(char x0, double x1) { return dlwrap::foo_Trait::get()(x0, x1); }
//
// DLWRAP_INTERNAL is similar, except the function it expands to is:
// static int dlwrap_foo(char x0, double x1) { ... }
// so that the function pointer call can be wrapped in library-specific code
//
// DLWRAP_INITIALIZE() declares static functions:
#define DLWRAP_INITIALIZE()                                                    \
  namespace dlwrap {                                                           \
  static size_t size();                                                        \
  static const char *symbol(size_t); /* get symbol name in [0, size()) */      \
  static void **                                                               \
      pointer(size_t); /* get pointer to function pointer in [0, size()) */    \
  }

// DLWRAP_FINALIZE() implements the functions from DLWRAP_INITIALIZE
#define DLWRAP_FINALIZE() DLWRAP_FINALIZE_IMPL()

// Implementation details follow.

namespace dlwrap {

// Extract return / argument types from address of function symbol
template <typename F> struct trait;
template <typename R, typename... Ts> struct trait<R (*)(Ts...)> {
  constexpr static const size_t nargs = sizeof...(Ts);
  typedef R ReturnType;
  template <size_t i> struct arg {
    typedef typename std::tuple_element<i, std::tuple<Ts...>>::type type;
  };

  typedef R (*FunctionType)(Ts...);
};

namespace type {
// Book keeping is by type specialization

template <size_t S> struct count {
  static constexpr size_t N = count<S - 1>::N;
};

template <> struct count<0> { static constexpr size_t N = 0; };

// Get a constexpr size_t ID, starts at zero
#define DLWRAP_ID() (dlwrap::type::count<__LINE__>::N)

// Increment value returned by DLWRAP_ID
#define DLWRAP_INC()                                                           \
  template <> struct dlwrap::type::count<__LINE__> {                           \
    static constexpr size_t N = 1 + dlwrap::type::count<__LINE__ - 1>::N;      \
  }

template <size_t N> struct symbol;
#define DLWRAP_SYMBOL(SYMBOL, ID)                                              \
  template <> struct dlwrap::type::symbol<ID> {                                \
    static constexpr const char *call() { return #SYMBOL; }                    \
  }
} // namespace type

template <size_t N, size_t... Is>
constexpr std::array<const char *, N> static getSymbolArray(
    std::index_sequence<Is...>) {
  return {{dlwrap::type::symbol<Is>::call()...}};
}

template <size_t Requested, size_t Required> constexpr void verboseAssert() {
  static_assert(Requested == Required, "Arity Error");
}

} // namespace dlwrap

#define DLWRAP_INSTANTIATE(SYM_USE, SYM_DEF, ARITY)                            \
  DLWRAP_INSTANTIATE_##ARITY(SYM_USE, SYM_DEF,                                 \
                             dlwrap::trait<decltype(&SYM_USE)>)

#define DLWRAP_FINALIZE_IMPL()                                                 \
  static size_t dlwrap::size() { return DLWRAP_ID(); }                         \
  static const char *dlwrap::symbol(size_t i) {                                \
    static constexpr const std::array<const char *, DLWRAP_ID()>               \
        dlwrap_symbols = getSymbolArray<DLWRAP_ID()>(                          \
            std::make_index_sequence<DLWRAP_ID()>());                          \
    return dlwrap_symbols[i];                                                  \
  }                                                                            \
  static void **dlwrap::pointer(size_t i) {                                    \
    static std::array<void *, DLWRAP_ID()> dlwrap_pointers;                    \
    return &dlwrap_pointers.data()[i];                                         \
  }

#define DLWRAP_COMMON(SYMBOL, ARITY)                                           \
  DLWRAP_INC();                                                                \
  DLWRAP_SYMBOL(SYMBOL, DLWRAP_ID() - 1);                                      \
  namespace dlwrap {                                                           \
  struct SYMBOL##_Trait : public dlwrap::trait<decltype(&SYMBOL)> {            \
    using T = dlwrap::trait<decltype(&SYMBOL)>;                                \
    static T::FunctionType get() {                                             \
      verboseAssert<ARITY, trait<decltype(&SYMBOL)>::nargs>();                 \
      constexpr size_t Index = DLWRAP_ID() - 1;                                \
      void *P = *dlwrap::pointer(Index);                                       \
      return reinterpret_cast<T::FunctionType>(P);                             \
    }                                                                          \
  };                                                                           \
  }

#define DLWRAP_IMPL(SYMBOL, ARITY)                                             \
  DLWRAP_COMMON(SYMBOL, ARITY);                                                \
  DLWRAP_INSTANTIATE(SYMBOL, SYMBOL, ARITY)

#define DLWRAP_INTERNAL_IMPL(SYMBOL, ARITY)                                    \
  DLWRAP_COMMON(SYMBOL, ARITY);                                                \
  static DLWRAP_INSTANTIATE(SYMBOL, dlwrap_##SYMBOL, ARITY)

#define DLWRAP_INSTANTIATE_0(SYM_USE, SYM_DEF, T)                              \
  T::ReturnType SYM_DEF() { return dlwrap::SYM_USE##_Trait::get()(); }
#define DLWRAP_INSTANTIATE_1(SYM_USE, SYM_DEF, T)                              \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0) {                \
    return dlwrap::SYM_USE##_Trait::get()(x0);                                 \
  }
#define DLWRAP_INSTANTIATE_2(SYM_USE, SYM_DEF, T)                              \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1) {                \
    return dlwrap::SYM_USE##_Trait::get()(x0, x1);                             \
  }
#define DLWRAP_INSTANTIATE_3(SYM_USE, SYM_DEF, T)                              \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2) {                \
    return dlwrap::SYM_USE##_Trait::get()(x0, x1, x2);                         \
  }
#define DLWRAP_INSTANTIATE_4(SYM_USE, SYM_DEF, T)                              \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3) {                \
    return dlwrap::SYM_USE##_Trait::get()(x0, x1, x2, x3);                     \
  }
#define DLWRAP_INSTANTIATE_5(SYM_USE, SYM_DEF, T)                              \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4) {                \
    return dlwrap::SYM_USE##_Trait::get()(x0, x1, x2, x3, x4);                 \
  }
#define DLWRAP_INSTANTIATE_6(SYM_USE, SYM_DEF, T)                              \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4,                  \
                        typename T::template arg<5>::type x5) {                \
    return dlwrap::SYM_USE##_Trait::get()(x0, x1, x2, x3, x4, x5);             \
  }

#define DLWRAP_INSTANTIATE_7(SYM_USE, SYM_DEF, T)                              \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4,                  \
                        typename T::template arg<5>::type x5,                  \
                        typename T::template arg<6>::type x6) {                \
    return dlwrap::SYM_USE##_Trait::get()(x0, x1, x2, x3, x4, x5, x6);         \
  }

#define DLWRAP_INSTANTIATE_8(SYM_USE, SYM_DEF, T)                              \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4,                  \
                        typename T::template arg<5>::type x5,                  \
                        typename T::template arg<6>::type x6,                  \
                        typename T::template arg<7>::type x7) {                \
    return dlwrap::SYM_USE##_Trait::get()(x0, x1, x2, x3, x4, x5, x6, x7);     \
  }
#define DLWRAP_INSTANTIATE_9(SYM_USE, SYM_DEF, T)                              \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4,                  \
                        typename T::template arg<5>::type x5,                  \
                        typename T::template arg<6>::type x6,                  \
                        typename T::template arg<7>::type x7,                  \
                        typename T::template arg<8>::type x8) {                \
    return dlwrap::SYM_USE##_Trait::get()(x0, x1, x2, x3, x4, x5, x6, x7, x8); \
  }
#define DLWRAP_INSTANTIATE_10(SYM_USE, SYM_DEF, T)                             \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4,                  \
                        typename T::template arg<5>::type x5,                  \
                        typename T::template arg<6>::type x6,                  \
                        typename T::template arg<7>::type x7,                  \
                        typename T::template arg<8>::type x8,                  \
                        typename T::template arg<9>::type x9) {                \
    return dlwrap::SYM_USE##_Trait::get()(x0, x1, x2, x3, x4, x5, x6, x7, x8,  \
                                          x9);                                 \
  }
#define DLWRAP_INSTANTIATE_11(SYM_USE, SYM_DEF, T)                             \
  T::ReturnType SYM_DEF(typename T::template arg<0>::type x0,                  \
                        typename T::template arg<1>::type x1,                  \
                        typename T::template arg<2>::type x2,                  \
                        typename T::template arg<3>::type x3,                  \
                        typename T::template arg<4>::type x4,                  \
                        typename T::template arg<5>::type x5,                  \
                        typename T::template arg<6>::type x6,                  \
                        typename T::template arg<7>::type x7,                  \
                        typename T::template arg<8>::type x8,                  \
                        typename T::template arg<9>::type x9,                  \
                        typename T::template arg<10>::type x10) {              \
    return dlwrap::SYM_USE##_Trait::get()(x0, x1, x2, x3, x4, x5, x6, x7, x8,  \
                                          x9, x10);                            \
  }

#endif
