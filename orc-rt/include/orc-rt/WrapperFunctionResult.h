//===---- WrapperFunctionResult.h -- blob-of-bytes container ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines WrapperFunctionResult and related APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_WRAPPERFUNCTIONRESULT_H
#define ORC_RT_WRAPPERFUNCTIONRESULT_H

#include "orc-rt-c/WrapperFunctionResult.h"

#include <utility>

namespace orc_rt {

/// A C++ convenience wrapper for orc_rt_WrapperFunctionResult. Auto-disposes
/// the contained result on destruction.
class WrapperFunctionResult {
public:
  /// Create a default WrapperFunctionResult.
  WrapperFunctionResult() { orc_rt_WrapperFunctionResultInit(&R); }

  /// Create a WrapperFunctionResult from a WrapperFunctionResult. This
  /// instance takes ownership of the result object and will automatically
  /// call dispose on the result upon destruction.
  WrapperFunctionResult(orc_rt_WrapperFunctionResult R) : R(R) {}

  WrapperFunctionResult(const WrapperFunctionResult &) = delete;
  WrapperFunctionResult &operator=(const WrapperFunctionResult &) = delete;

  WrapperFunctionResult(WrapperFunctionResult &&Other) {
    orc_rt_WrapperFunctionResultInit(&R);
    std::swap(R, Other.R);
  }

  WrapperFunctionResult &operator=(WrapperFunctionResult &&Other) {
    orc_rt_DisposeWrapperFunctionResult(&R);
    orc_rt_WrapperFunctionResultInit(&R);
    std::swap(R, Other.R);
    return *this;
  }

  ~WrapperFunctionResult() { orc_rt_DisposeWrapperFunctionResult(&R); }

  /// Relinquish ownership of and return the
  /// orc_rt_WrapperFunctionResult.
  orc_rt_WrapperFunctionResult release() {
    orc_rt_WrapperFunctionResult Tmp;
    orc_rt_WrapperFunctionResultInit(&Tmp);
    std::swap(R, Tmp);
    return Tmp;
  }

  /// Get a pointer to the data contained in this instance.
  char *data() { return orc_rt_WrapperFunctionResultData(&R); }

  /// Returns the size of the data contained in this instance.
  size_t size() const { return orc_rt_WrapperFunctionResultSize(&R); }

  /// Returns true if this value is equivalent to a default-constructed
  /// WrapperFunctionResult.
  bool empty() const { return orc_rt_WrapperFunctionResultEmpty(&R); }

  /// Create a WrapperFunctionResult with the given size and return a pointer
  /// to the underlying memory.
  static WrapperFunctionResult allocate(size_t Size) {
    WrapperFunctionResult R;
    R.R = orc_rt_WrapperFunctionResultAllocate(Size);
    return R;
  }

  /// Copy from the given char range.
  static WrapperFunctionResult copyFrom(const char *Source, size_t Size) {
    return orc_rt_CreateWrapperFunctionResultFromRange(Source, Size);
  }

  /// Copy from the given null-terminated string (includes the null-terminator).
  static WrapperFunctionResult copyFrom(const char *Source) {
    return orc_rt_CreateWrapperFunctionResultFromString(Source);
  }

  /// Create an out-of-band error by copying the given string.
  static WrapperFunctionResult createOutOfBandError(const char *Msg) {
    return orc_rt_CreateWrapperFunctionResultFromOutOfBandError(Msg);
  }

  /// If this value is an out-of-band error then this returns the error message,
  /// otherwise returns nullptr.
  const char *getOutOfBandError() const {
    return orc_rt_WrapperFunctionResultGetOutOfBandError(&R);
  }

private:
  orc_rt_WrapperFunctionResult R;
};

} // namespace orc_rt

#endif // ORC_RT_WRAPPERFUNCTIONRESULT_H
