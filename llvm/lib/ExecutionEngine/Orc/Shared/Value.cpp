//===---------=------- Value.cpp - Value implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/Value.h"

namespace llvm::orc {

template <typename T> Value Value::from(ExecutorAddr Ty, T result) {
  Value Val(Ty);
  Val.setValue<T>(result);
  return Val;
}

template <typename T>
struct LongEquivalentType {
  static_assert(sizeof(T) == 0, "LongEquivalentType is only defined for long and unsigned long");
};

template <> struct LongEquivalentType<long> {
  static_assert(sizeof(long) == 4 || sizeof(long) == 8,
                "'long' must be either 4 or 8 bytes");

  using type = std::conditional_t<sizeof(long) == 4, int32_t, int64_t>;
};

template <> struct LongEquivalentType<unsigned long> {
  static_assert(sizeof(unsigned long) == 4 || sizeof(unsigned long) == 8,
                "'unsigned long' must be either 4 or 8 bytes");

  using type =
      std::conditional_t<sizeof(unsigned long) == 4, uint32_t, uint64_t>;
};

template <typename T>
using NormalizedIntType = typename LongEquivalentType<T>::type;

template <typename T> void Value::setValue(T Val) {
  using DecayedT = std::decay_t<T>;

  if constexpr (std::is_same_v<DecayedT, long> || std::is_same_v<DecayedT, unsigned long>) {
    using CanonicalType = typename LongEquivalentType<DecayedT>::type;
    setValue(static_cast<CanonicalType>(Val));
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