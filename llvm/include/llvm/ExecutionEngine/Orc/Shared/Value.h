//===------------------------- Value.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Value class for capturing raw runtime values along with their type
// information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_VALUE_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_VALUE_H

#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"

namespace llvm {
namespace orc {

#define BUILTIN_TYPES                                                          \
  X(bool, Bool)                                                                \
  X(char, Char_S)                                                              \
  X(int8_t, SChar)                                                             \
  X(uint8_t, Char_U)                                                           \
  X(uint8_t, UChar)                                                            \
  X(int16_t, Short)                                                            \
  X(uint16_t, UShort)                                                          \
  X(int32_t, Int)                                                              \
  X(uint32_t, UInt)                                                            \
  X(int64_t, LongLong)                                                         \
  X(uint64_t, ULongLong)                                                       \
  // X(long, Long)                                                                \
  // X(unsigned long, ULong)                                                      \
  // X(float, Float)                                                              \
  // X(double, Double)                                                            \
  // X(long double, LongDouble)

class Value {
public:
  struct Storage {
    union {
#define X(type, name) type m_##name;
      BUILTIN_TYPES
#undef X
    };
    ExecutorSymbolDef m_Ptr;
  };

  enum Kind {
#define X(type, name) K_##name,
    BUILTIN_TYPES
#undef X

        K_Void,
    K_PtrOrObj,
    K_Unspecified
  };

  Value() = default;
  explicit Value(ExecutorAddr OpaqueType);
  Value(const Value &RHS);
  Value(Value &&RHS) noexcept;
  Value &operator=(const Value &RHS);
  Value &operator=(Value &&RHS) noexcept;
  ~Value() = default;

  template <typename T> static Value from(ExecutorAddr Ty, T result);
  template <typename T> void setValue(T Val);

  ExecutorAddr getOpaqueType() const { return OpaqueType; }

#define X(type, name)                                                          \
  void set##name(type Val) { Data.m_##name = Val; }                            \
  type get##name() const { return Data.m_##name; }
  BUILTIN_TYPES
#undef X
  void setPtrOrObj(void *Ptr) { Data.m_Ptr = ExecutorSymbolDef::fromPtr(Ptr); }
  void setPtrOrObj(ExecutorSymbolDef Ptr) { Data.m_Ptr = Ptr; }
  ExecutorSymbolDef getPtrOrObj() const {
    assert(ValueKind == K_PtrOrObj);
    return Data.m_Ptr;
  }

  Kind getKind() const { return ValueKind; }
  void setKind(Kind K) { ValueKind = K; }

  bool isValid() const { return ValueKind != K_Unspecified; }
  bool isVoid() const { return ValueKind == K_Void; }
  bool hasValue() const { return isValid() && !isVoid(); }
  explicit operator bool() const { return isValid(); }

private:
  ExecutorAddr OpaqueType;
  Storage Data;
  Kind ValueKind = K_Unspecified;
};

namespace shared {

struct SPSStorage {
  union {
#define X(type, name) type m_##name;
    BUILTIN_TYPES
#undef X
    SPSExecutorSymbolDef m_Ptr;
  };
};

using SPSValue = SPSTuple<SPSExecutorAddr, int32_t, SPSStorage>;

template <> class SPSSerializationTraits<SPSValue, Value> {
public:
  static size_t size(const Value &V) {
    size_t total = 0;

    total += SPSArgList<SPSExecutorAddr>::size(V.getOpaqueType());

    total += SPSArgList<int32_t>::size(static_cast<int32_t>(V.getKind()));

    switch (V.getKind()) {
#define X(type, name)                                                          \
  case Value::K_##name:                                                        \
    total += SPSArgList<type>::size(V.get##name());                            \
    break;
      BUILTIN_TYPES
#undef X

    case Value::K_PtrOrObj:
      total += SPSArgList<SPSExecutorSymbolDef>::size(V.getPtrOrObj());
      break;

    case Value::K_Void:
    case Value::K_Unspecified:
      break;
    }

    return total;
  }

  static bool serialize(SPSOutputBuffer &OB, const Value &V) {
    if (!SPSArgList<SPSExecutorAddr>::serialize(OB, V.getOpaqueType()))
      return false;

    if (!SPSArgList<int32_t>::serialize(OB, static_cast<int32_t>(V.getKind())))
      return false;

    switch (V.getKind()) {
#define X(type, name)                                                          \
  case Value::K_##name:                                                        \
    if (!SPSArgList<type>::serialize(OB, V.get##name()))                       \
      return false;                                                            \
    break;
      BUILTIN_TYPES
#undef X

    case Value::K_PtrOrObj:
      if (!SPSArgList<SPSExecutorSymbolDef>::serialize(OB, V.getPtrOrObj()))
        return false;
      break;

    case Value::K_Void:
    case Value::K_Unspecified:
      // No payload to serialize
      break;
    }

    return true;
  }

  static bool deserialize(SPSInputBuffer &IB, Value &V) {
    ExecutorAddr OpaqueTy;
    if (!SPSArgList<SPSExecutorAddr>::deserialize(IB, OpaqueTy))
      return false;

    int32_t KindInt;
    if (!SPSArgList<int32_t>::deserialize(IB, KindInt))
      return false;

    Value::Kind K = static_cast<Value::Kind>(KindInt);
    V = Value(OpaqueTy);
    V.setKind(K);

    switch (K) {
#define X(type, name)                                                          \
  case Value::K_##name: {                                                      \
    type T{};                                                                  \
    if (!SPSArgList<type>::deserialize(IB, T))                                 \
      return false;                                                            \
    V.set##name(T);                                                            \
    break;                                                                     \
  }
      BUILTIN_TYPES
#undef X

    case Value::K_PtrOrObj: {
      ExecutorSymbolDef Sym;
      if (!SPSArgList<SPSExecutorSymbolDef>::deserialize(IB, Sym))
        return false;
      V.setPtrOrObj(Sym);
      break;
    }

    case Value::K_Void:
    case Value::K_Unspecified:
      // No payload to deserialize
      break;
    }

    return true;
  }
};

} // end namespace shared
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_VALUE_H
