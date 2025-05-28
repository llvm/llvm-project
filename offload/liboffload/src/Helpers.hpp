//===- helpers.hpp- GetInfo return helpers for the new LLVM/Offload API ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The getInfo*/ReturnHelper facilities provide shortcut way of writing return
// data + size for the various getInfo APIs. Based on the equivalent
// implementations in Unified Runtime.
//
//===----------------------------------------------------------------------===//

#include "OffloadAPI.h"
#include "Shared/OffloadError.h"
#include "llvm/Support/Error.h"

#include <cstring>

template <typename T, typename Assign>
llvm::Error getInfoImpl(size_t ParamValueSize, void *ParamValue,
                        size_t *ParamValueSizeRet, T Value, size_t ValueSize,
                        Assign &&AssignFunc) {
  if (!ParamValue && !ParamValueSizeRet) {
    return error::createOffloadError(error::ErrorCode::INVALID_NULL_POINTER,
                                     "value and size outputs are nullptr");
  }

  if (ParamValue != nullptr) {
    if (ParamValueSize < ValueSize) {
      return error::createOffloadError(error::ErrorCode::INVALID_SIZE,
                                       "provided size is invalid");
    }
    AssignFunc(ParamValue, Value, ValueSize);
  }

  if (ParamValueSizeRet != nullptr) {
    *ParamValueSizeRet = ValueSize;
  }

  return llvm::Error::success();
}

template <typename T>
llvm::Error getInfo(size_t ParamValueSize, void *ParamValue,
                    size_t *ParamValueSizeRet, T Value) {
  auto Assignment = [](void *ParamValue, T Value, size_t) {
    *static_cast<T *>(ParamValue) = Value;
  };

  return getInfoImpl(ParamValueSize, ParamValue, ParamValueSizeRet, Value,
                     sizeof(T), Assignment);
}

template <typename T>
llvm::Error getInfoArray(size_t array_length, size_t ParamValueSize,
                         void *ParamValue, size_t *ParamValueSizeRet,
                         const T *Value) {
  return getInfoImpl(ParamValueSize, ParamValue, ParamValueSizeRet, Value,
                     array_length * sizeof(T), memcpy);
}

template <>
inline llvm::Error
getInfo<const char *>(size_t ParamValueSize, void *ParamValue,
                      size_t *ParamValueSizeRet, const char *Value) {
  return getInfoArray(strlen(Value) + 1, ParamValueSize, ParamValue,
                      ParamValueSizeRet, Value);
}

class ReturnHelper {
public:
  ReturnHelper(size_t ParamValueSize, void *ParamValue,
               size_t *ParamValueSizeRet)
      : ParamValueSize(ParamValueSize), ParamValue(ParamValue),
        ParamValueSizeRet(ParamValueSizeRet) {}

  // A version where in/out info size is represented by a single pointer
  // to a value which is updated on return
  ReturnHelper(size_t *ParamValueSize, void *ParamValue)
      : ParamValueSize(*ParamValueSize), ParamValue(ParamValue),
        ParamValueSizeRet(ParamValueSize) {}

  // Scalar return Value
  template <class T> llvm::Error operator()(const T &t) {
    return getInfo(ParamValueSize, ParamValue, ParamValueSizeRet, t);
  }

  // Array return Value
  template <class T> llvm::Error operator()(const T *t, size_t s) {
    return getInfoArray(s, ParamValueSize, ParamValue, ParamValueSizeRet, t);
  }

protected:
  size_t ParamValueSize;
  void *ParamValue;
  size_t *ParamValueSizeRet;
};
