//===- LogicalResult.h - Utilities for handling success/failure -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_LOGICAL_RESULT_H
#define MLIR_SUPPORT_LOGICAL_RESULT_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Optional.h"

namespace mlir {

/// Values that can be used to signal success/failure. This should be used in
/// conjunction with the utility functions below.
struct LogicalResult {
  enum ResultEnum { Success, Failure } value;
  LogicalResult(ResultEnum v) : value(v) {}
};

/// Utility function to generate a LogicalResult. If isSuccess is true a
/// `success` result is generated, otherwise a 'failure' result is generated.
inline LogicalResult success(bool isSuccess = true) {
  return LogicalResult{isSuccess ? LogicalResult::Success
                                 : LogicalResult::Failure};
}

/// Utility function to generate a LogicalResult. If isFailure is true a
/// `failure` result is generated, otherwise a 'success' result is generated.
inline LogicalResult failure(bool isFailure = true) {
  return LogicalResult{isFailure ? LogicalResult::Failure
                                 : LogicalResult::Success};
}

/// Utility function that returns true if the provided LogicalResult corresponds
/// to a success value.
inline bool succeeded(LogicalResult result) {
  return result.value == LogicalResult::Success;
}

/// Utility function that returns true if the provided LogicalResult corresponds
/// to a failure value.
inline bool failed(LogicalResult result) {
  return result.value == LogicalResult::Failure;
}

/// This class provides support for representing a failure result, or a valid
/// value of type `T`. This allows for integrating with LogicalResult, while
/// also providing a value on the success path.
template <typename T> class LLVM_NODISCARD FailureOr : public Optional<T> {
public:
  /// Allow constructing from a LogicalResult. The result *must* be a failure.
  /// Success results should use a proper instance of type `T`.
  FailureOr(LogicalResult result) {
    assert(failed(result) &&
           "success should be constructed with an instance of 'T'");
  }
  FailureOr() : FailureOr(failure()) {}
  FailureOr(T &&y) : Optional<T>(std::forward<T>(y)) {}

  operator LogicalResult() const { return success(this->hasValue()); }

private:
  /// Hide the bool conversion as it easily creates confusion.
  using Optional<T>::operator bool;
  using Optional<T>::hasValue;
};

} // namespace mlir

#endif // MLIR_SUPPORT_LOGICAL_RESULT_H
