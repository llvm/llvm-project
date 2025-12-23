//===---------=------- Value.cpp - Value implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/Value.h"

namespace llvm::orc {

Value::Value(ExecutorAddr OpaqueType) : OpaqueType(OpaqueType) {}

Value::Value(const Value &RHS)
    : OpaqueType(RHS.OpaqueType), Data(RHS.Data), ValueKind(RHS.ValueKind) {}

Value::Value(Value &&RHS) noexcept {
  OpaqueType = std::exchange(RHS.OpaqueType, ExecutorAddr());
  Data = RHS.Data;
  ValueKind = std::exchange(RHS.ValueKind, K_Unspecified);
}

Value &Value::operator=(const Value &RHS) {
  OpaqueType = RHS.OpaqueType;
  Data = RHS.Data;
  ValueKind = RHS.ValueKind;
  return *this;
}

Value &Value::operator=(Value &&RHS) noexcept {
  OpaqueType = std::exchange(RHS.OpaqueType, ExecutorAddr());
  ValueKind = std::exchange(RHS.ValueKind, K_Unspecified);
  Data = RHS.Data;
  return *this;
}

template <typename T> Value Value::from(ExecutorAddr Ty, T result) {
  Value Val(Ty, result);
  Val.setValue<T>(result);
  return Val;
}

template <typename T, typename I32, typename I64> struct MapToFixedWidth {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "Unsupported size for integer type (must be 4 or 8 bytes)");
  using type = std::conditional_t<sizeof(T) == 4, I32, I64>;
};

template <typename T> struct LongEquivalentType {
  static_assert(
      sizeof(T) == 0,
      "LongEquivalentType is only defined for specific integral types");
};

template <>
struct LongEquivalentType<long> : MapToFixedWidth<long, int32_t, int64_t> {};

template <>
struct LongEquivalentType<unsigned long>
    : MapToFixedWidth<unsigned long, uint32_t, uint64_t> {};

template <>
struct LongEquivalentType<long long>
    : MapToFixedWidth<long long, int32_t, int64_t> {};

template <>
struct LongEquivalentType<unsigned long long>
    : MapToFixedWidth<unsigned long long, uint32_t, uint64_t> {};

template <typename T>
using NormalizedIntType = typename LongEquivalentType<T>::type;

template <typename T> void Value::setValue(T Val) {
  using DecayedT = std::decay_t<T>;

  if constexpr (std::is_same_v<DecayedT, long> ||
                std::is_same_v<DecayedT, unsigned long> ||
                std::is_same_v<DecayedT, long long> ||
                std::is_same_v<DecayedT, unsigned long long>) {
    using CanonicalType = NormalizedIntType<DecayedT>;
    setValue<CanonicalType>(static_cast<CanonicalType>(Val));
    return;
  }
#define X(type, name)                                                          \
  else if constexpr (std::is_same_v<DecayedT, type>) {                         \
    set##name(Val);                                                            \
    ValueKind = Value::K_##name;                                               \
  }

  BUILTIN_TYPES

#undef X
  else {
    static_assert(std::is_pointer_v<T>, "Unsupported type for setValue");

    // if constexpr (std::is_function_v<T>) {
    //   setPtrOrObj(ExecutorSymbolDef::fromPtr(&Val));
    // } else if constexpr (std::is_pointer_v<T> || std::is_array_v<T>) {
    //   setPtrOrObj(ExecutorSymbolDef::fromPtr(Val));
    // } else if constexpr (std::is_class_v<T> || std::is_union_v<T>) {
    //   setPtrOrObj(ExecutorSymbolDef::fromPtr(&Val));
    if constexpr (std::is_pointer_v<T>) {
      setPtrOrObj(ExecutorSymbolDef::fromPtr(Val));
    } else {
      static_assert(!std::is_same_v<T, T>, "Unsupported non-builtin type");
    }
    ValueKind = Value::K_PtrOrObj;
  }
}

} // end namespace llvm::orc