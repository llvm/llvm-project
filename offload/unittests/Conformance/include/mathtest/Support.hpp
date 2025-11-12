//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of various metaprogramming helpers and
/// support utilities for the math test framework.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_SUPPORT_HPP
#define MATHTEST_SUPPORT_HPP

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mathtest {

//===----------------------------------------------------------------------===//
// Function & Type Traits
//===----------------------------------------------------------------------===//

namespace detail {

template <typename T> struct FunctionTraitsImpl;

template <typename RetType, typename... ArgTypes>
struct FunctionTraitsImpl<RetType(ArgTypes...)> {
  using ReturnType = RetType;
  using ArgTypesTuple = std::tuple<ArgTypes...>;
};

template <typename RetType, typename... ArgTypes>
struct FunctionTraitsImpl<RetType(ArgTypes...) noexcept>
    : FunctionTraitsImpl<RetType(ArgTypes...)> {};

template <typename FuncType>
struct FunctionTraitsImpl<FuncType *> : FunctionTraitsImpl<FuncType> {};
} // namespace detail

template <auto Func>
using FunctionTraits = detail::FunctionTraitsImpl<
    std::remove_pointer_t<std::decay_t<decltype(Func)>>>;

template <typename FuncType>
using FunctionTypeTraits = detail::FunctionTraitsImpl<FuncType>;

template <typename T> struct TypeIdentityOf {
  using type = T;
};

template <typename TupleTypes, template <typename...> class Template>
struct ApplyTupleTypes;

template <template <typename...> class Template, typename... Ts>
struct ApplyTupleTypes<std::tuple<Ts...>, Template> {
  using type = Template<Ts...>;
};

template <typename TupleTypes, template <typename...> class Template>
using ApplyTupleTypes_t = typename ApplyTupleTypes<TupleTypes, Template>::type;

namespace detail {

template <typename T> struct KernelSignatureOfImpl;

template <typename RetType, typename... ArgTypes>
struct KernelSignatureOfImpl<RetType(ArgTypes...)> {
  using type = void(const std::decay_t<ArgTypes> *..., RetType *, std::size_t);
};

template <typename RetType, typename... ArgTypes>
struct KernelSignatureOfImpl<RetType(ArgTypes...) noexcept>
    : KernelSignatureOfImpl<RetType(ArgTypes...)> {};
} // namespace detail

template <auto Func>
using KernelSignatureOf = detail::KernelSignatureOfImpl<
    std::remove_pointer_t<std::decay_t<decltype(Func)>>>;

template <auto Func>
using KernelSignatureOf_t = typename KernelSignatureOf<Func>::type;

//===----------------------------------------------------------------------===//
// Kernel Argument Packing
//===----------------------------------------------------------------------===//

template <typename... ArgTypes> struct KernelArgsPack;

template <typename ArgType> struct KernelArgsPack<ArgType> {
  std::decay_t<ArgType> Arg;

  constexpr KernelArgsPack(ArgType &&Arg) : Arg(std::forward<ArgType>(Arg)) {}
};

template <typename ArgType0, typename ArgType1, typename... ArgTypes>
struct KernelArgsPack<ArgType0, ArgType1, ArgTypes...> {
  std::decay_t<ArgType0> Arg0;
  KernelArgsPack<ArgType1, ArgTypes...> Args;

  constexpr KernelArgsPack(ArgType0 &&Arg0, ArgType1 &&Arg1, ArgTypes &&...Args)
      : Arg0(std::forward<ArgType0>(Arg0)),
        Args(std::forward<ArgType1>(Arg1), std::forward<ArgTypes>(Args)...) {}
};

template <typename... ArgTypes>
KernelArgsPack<ArgTypes...> makeKernelArgsPack(ArgTypes &&...Args) {
  return KernelArgsPack<ArgTypes...>(std::forward<ArgTypes>(Args)...);
}

//===----------------------------------------------------------------------===//
// Configuration Helpers
//===----------------------------------------------------------------------===//

template <auto Func> struct FunctionConfig;

namespace detail {

template <typename... BufferTypes>
static constexpr std::size_t getDefaultBufferSize() {
  static_assert(sizeof...(BufferTypes) > 0,
                "At least one buffer type must be provided");

  constexpr std::size_t TotalMemoryInBytes = 512ULL << 20; // 512 MiB
  constexpr std::size_t ElementTupleSize = (sizeof(BufferTypes) + ...);

  static_assert(ElementTupleSize > 0,
                "Cannot calculate buffer size for empty types");

  return TotalMemoryInBytes / ElementTupleSize;
}
} // namespace detail

template <typename BufferType, typename BufferTupleTypes>
struct DefaultBufferSizeFor;

template <typename BufferType, typename... BufferTypes>
struct DefaultBufferSizeFor<BufferType, std::tuple<BufferTypes...>> {
  static constexpr std::size_t value // NOLINT(readability-identifier-naming)
      = detail::getDefaultBufferSize<BufferType, BufferTypes...>();
};

template <typename OutType, typename InTypesTuple>
inline constexpr std::size_t
    DefaultBufferSizeFor_v // NOLINT(readability-identifier-naming)
    = DefaultBufferSizeFor<OutType, InTypesTuple>::value;
} // namespace mathtest

#endif // MATHTEST_SUPPORT_HPP
