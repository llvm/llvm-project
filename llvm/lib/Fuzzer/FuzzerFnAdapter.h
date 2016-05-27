//===- FuzzerAdapter.h - Arbitrary function Fuzzer adapter -------*- C++ -*===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// W A R N I N G :  E X P E R I M E N T A L.
//
// Defines an adapter to fuzz functions with (almost) arbitrary signatures.
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_ADAPTER_H
#define LLVM_FUZZER_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

namespace fuzzer {

/// Unpacks bytes from \p Data according to \p F argument types
/// and calls the function.
/// Use to automatically adapt LLVMFuzzerTestOneInput interface to
/// a specific function.
/// Supported argument types: primitive types, std::vector<uint8_t>.
template <typename Fn> bool Adapt(Fn F, const uint8_t *Data, size_t Size);

// The implementation performs several steps:
// - function argument types are obtained (Args...)
// - data is unpacked into std::tuple<Args...> one by one
// - function is called with std::tuple<Args...> containing arguments.
namespace impl {

// Single argument unpacking.

template <typename T>
size_t UnpackPrimitive(const uint8_t *Data, size_t Size, T *Value) {
  if (Size < sizeof(T))
    return Size;
  *Value = *reinterpret_cast<const T *>(Data);
  return Size - sizeof(T);
}

/// Unpacks into a given Value and returns the Size - num_consumed_bytes.
/// Return value equal to Size signals inability to unpack the data (typically
/// because there are not enough bytes).
template <typename T>
size_t UnpackSingle(const uint8_t *Data, size_t Size, T *Value);

#define UNPACK_SINGLE_PRIMITIVE(Type)                                          \
  template <>                                                                  \
  size_t UnpackSingle<Type>(const uint8_t *Data, size_t Size, Type *Value) {   \
    return UnpackPrimitive(Data, Size, Value);                                 \
  }

UNPACK_SINGLE_PRIMITIVE(char)
UNPACK_SINGLE_PRIMITIVE(signed char)
UNPACK_SINGLE_PRIMITIVE(unsigned char)

UNPACK_SINGLE_PRIMITIVE(short int)
UNPACK_SINGLE_PRIMITIVE(unsigned short int)

UNPACK_SINGLE_PRIMITIVE(int)
UNPACK_SINGLE_PRIMITIVE(unsigned int)

UNPACK_SINGLE_PRIMITIVE(long int)
UNPACK_SINGLE_PRIMITIVE(unsigned long int)

UNPACK_SINGLE_PRIMITIVE(bool)
UNPACK_SINGLE_PRIMITIVE(wchar_t)

UNPACK_SINGLE_PRIMITIVE(float)
UNPACK_SINGLE_PRIMITIVE(double)
UNPACK_SINGLE_PRIMITIVE(long double)

#undef UNPACK_SINGLE_PRIMITIVE

template <>
size_t UnpackSingle<std::vector<uint8_t>>(const uint8_t *Data, size_t Size,
                                          std::vector<uint8_t> *Value) {
  if (Size < 1)
    return Size;
  size_t Len = std::min(static_cast<size_t>(*Data), Size - 1);
  std::vector<uint8_t> V(Data + 1, Data + 1 + Len);
  Value->swap(V);
  return Size - Len - 1;
}

template <>
size_t UnpackSingle<std::string>(const uint8_t *Data, size_t Size,
    std::string *Value) {
  if (Size < 1)
    return Size;
  size_t Len = std::min(static_cast<size_t>(*Data), Size - 1);
  std::string S(Data + 1, Data + 1 + Len);
  Value->swap(S);
  return Size - Len - 1;
}

// Unpacking into arbitrary tuple.

// Recursion guard.
template <int N, typename TupleT>
typename std::enable_if<N == std::tuple_size<TupleT>::value, bool>::type
UnpackImpl(const uint8_t *Data, size_t Size, TupleT *Tuple) {
  return true;
}

// Unpack tuple elements starting from Nth.
template <int N, typename TupleT>
typename std::enable_if<N < std::tuple_size<TupleT>::value, bool>::type
UnpackImpl(const uint8_t *Data, size_t Size, TupleT *Tuple) {
  size_t NewSize = UnpackSingle(Data, Size, &std::get<N>(*Tuple));
  if (NewSize == Size) {
    return false;
  }

  return UnpackImpl<N + 1, TupleT>(Data + (Size - NewSize), NewSize, Tuple);
}

// Unpacks into arbitrary tuple and returns true if successful.
template <typename... Args>
bool Unpack(const uint8_t *Data, size_t Size, std::tuple<Args...> *Tuple) {
  return UnpackImpl<0, std::tuple<Args...>>(Data, Size, Tuple);
}

// Helper integer sequence templates.

template <int...> struct Seq {};

template <int N, int... S> struct GenSeq : GenSeq<N - 1, N - 1, S...> {};

// GenSeq<N>::type is Seq<0, 1, ..., N-1>
template <int... S> struct GenSeq<0, S...> { typedef Seq<S...> type; };

// Function signature introspection.

template <typename T> struct FnTraits {};

template <typename ReturnType, typename... Args>
struct FnTraits<ReturnType (*)(Args...)> {
  enum { Arity = sizeof...(Args) };
  typedef std::tuple<Args...> ArgsTupleT;
};

// Calling a function with arguments in a tuple.

template <typename Fn, int... S>
void ApplyImpl(Fn F, const typename FnTraits<Fn>::ArgsTupleT &Params,
               Seq<S...>) {
  F(std::get<S>(Params)...);
}

template <typename Fn>
void Apply(Fn F, const typename FnTraits<Fn>::ArgsTupleT &Params) {
  // S is Seq<0, ..., Arity-1>
  auto S = typename GenSeq<FnTraits<Fn>::Arity>::type();
  ApplyImpl(F, Params, S);
}

// Unpacking data into arguments tuple of correct type and calling the function.
template <typename Fn>
bool UnpackAndApply(Fn F, const uint8_t *Data, size_t Size) {
  typename FnTraits<Fn>::ArgsTupleT Tuple;
  if (!Unpack(Data, Size, &Tuple))
    return false;

  Apply(F, Tuple);
  return true;
}

} // namespace impl

template <typename Fn> bool Adapt(Fn F, const uint8_t *Data, size_t Size) {
  return impl::UnpackAndApply(F, Data, Size);
}

} // namespace fuzzer

#endif
