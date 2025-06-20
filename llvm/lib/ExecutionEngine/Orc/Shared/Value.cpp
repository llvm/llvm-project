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

template <typename T> void setValue(T Val) {
  using DecayedT = std::decay_t<T>;
  if constexpr (std::is_void_v<DecayedT>) {
    ValueKind = K_Void;
  }
#define X(type, name)                                                          \
  else if constexpr (std::is_same_v<DecayedT, type>) {                         \
    set##name(Val);                                                            \
    ValueKind = K_##name;                                                      \
  }
  BUILTIN_TYPES
#undef X
  else {
    // static_assert(std::is_trivially_copyable_v<T> || std::is_pointer_v<T>,
    //               "Unsupported type for setValue");
    static_assert(std::is_pointer_v<T>,
                  "Unsupported type for setValue");

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
    ValueKind = K_PtrOrObj;
  }
}

} // end namespace llvm::orc