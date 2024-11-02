//===- LogicalResult.h - Utilities for handling success/failure -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_LOGICALRESULT_H
#define MLIR_SUPPORT_LOGICALRESULT_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Optional.h"

namespace mlir {

/// This class represents an efficient way to signal success or failure. It
/// should be preferred over the use of `bool` when appropriate, as it avoids
/// all of the ambiguity that arises in interpreting a boolean result. This
/// class is marked as NODISCARD to ensure that the result is processed. Users
/// may explicitly discard a result by using `(void)`, e.g.
/// `(void)functionThatReturnsALogicalResult();`. Given the intended nature of
/// this class, it generally shouldn't be used as the result of functions that
/// very frequently have the result ignored. This class is intended to be used
/// in conjunction with the utility functions below.
struct LLVM_NODISCARD LogicalResult {
public:
  /// If isSuccess is true a `success` result is generated, otherwise a
  /// 'failure' result is generated.
  static LogicalResult success(bool isSuccess = true) {
    return LogicalResult(isSuccess);
  }

  /// If isFailure is true a `failure` result is generated, otherwise a
  /// 'success' result is generated.
  static LogicalResult failure(bool isFailure = true) {
    return success(!isFailure);
  }

  /// Returns true if the provided LogicalResult corresponds to a success value.
  bool succeeded() const { return isSuccess; }

  /// Returns true if the provided LogicalResult corresponds to a failure value.
  bool failed() const { return !succeeded(); }

private:
  LogicalResult(bool isSuccess) : isSuccess(isSuccess) {}

  /// Boolean indicating if this is a success result, if false this is a
  /// failure result.
  bool isSuccess;
};

/// Utility function to generate a LogicalResult. If isSuccess is true a
/// `success` result is generated, otherwise a 'failure' result is generated.
inline LogicalResult success(bool isSuccess = true) {
  return LogicalResult::success(isSuccess);
}

/// Utility function to generate a LogicalResult. If isFailure is true a
/// `failure` result is generated, otherwise a 'success' result is generated.
inline LogicalResult failure(bool isFailure = true) {
  return LogicalResult::failure(isFailure);
}

/// Utility function that returns true if the provided LogicalResult corresponds
/// to a success value.
inline bool succeeded(LogicalResult result) { return result.succeeded(); }

/// Utility function that returns true if the provided LogicalResult corresponds
/// to a failure value.
inline bool failed(LogicalResult result) { return result.failed(); }

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
  FailureOr(const T &y) : Optional<T>(y) {}
  template <typename U,
            std::enable_if_t<std::is_constructible<T, U>::value> * = nullptr>
  FailureOr(const FailureOr<U> &other)
      : Optional<T>(failed(other) ? Optional<T>() : Optional<T>(*other)) {}

  operator LogicalResult() const { return success(this->hasValue()); }

private:
  /// Hide the bool conversion as it easily creates confusion.
  using Optional<T>::operator bool;
  using Optional<T>::hasValue;
};

/// This class represents success/failure for parsing-like operations that find
/// it important to chain together failable operations with `||`.  This is an
/// extended version of `LogicalResult` that allows for explicit conversion to
/// bool.
///
/// This class should not be used for general error handling cases - we prefer
/// to keep the logic explicit with the `succeeded`/`failed` predicates.
/// However, traditional monadic-style parsing logic can sometimes get
/// swallowed up in boilerplate without this, so we provide this for narrow
/// cases where it is important.
///
class LLVM_NODISCARD ParseResult : public LogicalResult {
public:
  ParseResult(LogicalResult result = success()) : LogicalResult(result) {}

  /// Failure is true in a boolean context.
  explicit operator bool() const { return failed(); }
};

} // namespace mlir

#endif // MLIR_SUPPORT_LOGICALRESULT_H
