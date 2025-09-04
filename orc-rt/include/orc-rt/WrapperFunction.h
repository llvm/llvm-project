//===-------- WrapperFunction.h - Wrapper function utils --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines WrapperFunctionBuffer and related APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_WRAPPERFUNCTION_H
#define ORC_RT_WRAPPERFUNCTION_H

#include "orc-rt-c/WrapperFunction.h"

#include <utility>

namespace orc_rt {

/// A C++ convenience wrapper for orc_rt_WrapperFunctionBuffer. Auto-disposes
/// the contained result on destruction.
class WrapperFunctionBuffer {
public:
  /// Create a default WrapperFunctionBuffer.
  WrapperFunctionBuffer() { orc_rt_WrapperFunctionBufferInit(&B); }

  /// Create a WrapperFunctionBuffer from a WrapperFunctionBuffer. This
  /// instance takes ownership of the result object and will automatically
  /// call dispose on the result upon destruction.
  WrapperFunctionBuffer(orc_rt_WrapperFunctionBuffer B) : B(B) {}

  WrapperFunctionBuffer(const WrapperFunctionBuffer &) = delete;
  WrapperFunctionBuffer &operator=(const WrapperFunctionBuffer &) = delete;

  WrapperFunctionBuffer(WrapperFunctionBuffer &&Other) {
    orc_rt_WrapperFunctionBufferInit(&B);
    std::swap(B, Other.B);
  }

  WrapperFunctionBuffer &operator=(WrapperFunctionBuffer &&Other) {
    orc_rt_WrapperFunctionBufferDispose(&B);
    orc_rt_WrapperFunctionBufferInit(&B);
    std::swap(B, Other.B);
    return *this;
  }

  ~WrapperFunctionBuffer() { orc_rt_WrapperFunctionBufferDispose(&B); }

  /// Relinquish ownership of and return the
  /// orc_rt_WrapperFunctionBuffer.
  orc_rt_WrapperFunctionBuffer release() {
    orc_rt_WrapperFunctionBuffer Tmp;
    orc_rt_WrapperFunctionBufferInit(&Tmp);
    std::swap(B, Tmp);
    return Tmp;
  }

  /// Get a pointer to the data contained in this instance.
  char *data() { return orc_rt_WrapperFunctionBufferData(&B); }

  /// Returns the size of the data contained in this instance.
  size_t size() const { return orc_rt_WrapperFunctionBufferSize(&B); }

  /// Returns true if this value is equivalent to a default-constructed
  /// WrapperFunctionBuffer.
  bool empty() const { return orc_rt_WrapperFunctionBufferEmpty(&B); }

  /// Create a WrapperFunctionBuffer with the given size and return a pointer
  /// to the underlying memory.
  static WrapperFunctionBuffer allocate(size_t Size) {
    return orc_rt_WrapperFunctionBufferAllocate(Size);
  }

  /// Copy from the given char range.
  static WrapperFunctionBuffer copyFrom(const char *Source, size_t Size) {
    return orc_rt_CreateWrapperFunctionBufferFromRange(Source, Size);
  }

  /// Copy from the given null-terminated string (includes the null-terminator).
  static WrapperFunctionBuffer copyFrom(const char *Source) {
    return orc_rt_CreateWrapperFunctionBufferFromString(Source);
  }

  /// Create an out-of-band error by copying the given string.
  static WrapperFunctionBuffer createOutOfBandError(const char *Msg) {
    return orc_rt_CreateWrapperFunctionBufferFromOutOfBandError(Msg);
  }

  /// If this value is an out-of-band error then this returns the error message,
  /// otherwise returns nullptr.
  const char *getOutOfBandError() const {
    return orc_rt_WrapperFunctionBufferGetOutOfBandError(&B);
  }

private:
  orc_rt_WrapperFunctionBuffer B;
};

} // namespace orc_rt

#endif // ORC_RT_WRAPPERFUNCTION_H
