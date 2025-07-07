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
#include "OffloadError.h"
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

llvm::Error getInfoString(size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet, llvm::StringRef Value) {
  return getInfoArray(Value.size() + 1, ParamValueSize, ParamValue,
                      ParamValueSizeRet, Value.data());
}

class InfoWriter {
public:
  InfoWriter(size_t Size, void *Target, size_t *SizeRet)
      : Size(Size), Target(Target), SizeRet(SizeRet) {};
  InfoWriter() = delete;
  InfoWriter(InfoWriter &) = delete;
  ~InfoWriter() = default;

  template <typename T> llvm::Error write(llvm::Expected<T> &&Val) {
    if (Val)
      return getInfo(Size, Target, SizeRet, *Val);
    return Val.takeError();
  }

  template <typename T>
  llvm::Error writeArray(llvm::Expected<T> &&Val, size_t Elems) {
    if (Val)
      return getInfoArray(Elems, Size, Target, SizeRet, *Val);
    return Val.takeError();
  }

  llvm::Error writeString(llvm::Expected<llvm::StringRef> &&Val) {
    if (Val)
      return getInfoString(Size, Target, SizeRet, *Val);
    return Val.takeError();
  }

private:
  size_t Size;
  void *Target;
  size_t *SizeRet;
};
