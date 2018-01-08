//===--- status.h - Status and Expected classes -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef ACXXEL_STATUS_H
#define ACXXEL_STATUS_H

#include <cassert>
#include <string>

// The clang compiler supports annotating class declarations with the
// warn_unused_result attribute, and this has the meaning that whenever that
// type is returned from a function, the function is marked as
// warn_unused_result.
//
// The gcc compiler does not support warn_unused_result for classes, only for
// functions, so we only use this feature with clang.
#ifdef __clang__
#define ACXXEL_WARN_UNUSED_RESULT_TYPE __attribute__((warn_unused_result))
#else
#define ACXXEL_WARN_UNUSED_RESULT_TYPE
#endif

namespace acxxel {

/// Status type.
///
/// May represent failure with a string error message, or may indicate success.
class ACXXEL_WARN_UNUSED_RESULT_TYPE Status {
public:
  /// Creates a Status representing success.
  Status() : HasMessage(false) {}

  /// Creates a Status representing failure with the given error message.
  explicit Status(const std::string &Message)
      : HasMessage(true), Message(Message) {}

  Status(const Status &) = default;

  Status &operator=(const Status &) = default;

  Status(Status &&) noexcept = default;

  // Cannot use default because the move assignment operator for std::string is
  // not marked noexcept.
  Status &operator=(Status &&That) noexcept {
    HasMessage = That.HasMessage;
    Message = std::move(That.Message);
    return *this;
  }

  ~Status() = default;

  /// Returns true if this Status represents failure. Otherwise, returns false.
  bool isError() const { return HasMessage; }

  /// Returns true if this Status represents success. Otherwise, returns false.
  operator bool() const { return !HasMessage; }

  /// Gets a reference to the error message for this Status.
  ///
  /// Should only be called if isError() returns true.
  const std::string &getMessage() const { return Message; }

private:
  bool HasMessage;
  std::string Message;
};

class ExpectedBase {
protected:
  enum class State {
    SUCCESS,
    FAILURE,
    MOVED,
  };
};

/// Either a value of type T or a Status representing failure.
template <typename T> class Expected : public ExpectedBase {
public:
  /// Creates an Expected representing failure with the given Error status.
  // Intentionally implicit.
  Expected(Status AnError)
      : TheState(State::FAILURE), TheError(std::move(AnError)) {
    assert(AnError.isError() && "constructing an error Expected value from a "
                                "success status is not allowed");
  }

  /// Creates an Expected representing success with the given value.
  // Intentionally implicit.
  Expected(T Value) : TheState(State::SUCCESS), TheValue(std::move(Value)) {}

  Expected(const Expected &That) : TheState(That.TheState) {
    switch (TheState) {
    case State::SUCCESS:
      new (&TheValue) T(That.TheValue);
      break;
    case State::FAILURE:
      new (&TheError) Status(That.TheError);
      break;
    case State::MOVED:
      // Nothing to do in this case.
      break;
    }
  }

  Expected &operator=(Expected That) {
    TheState = That.TheState;
    switch (TheState) {
    case State::SUCCESS:
      new (&TheValue) T(std::move(That.TheValue));
      break;
    case State::FAILURE:
      new (&TheError) Status(std::move(That.TheError));
      break;
    case State::MOVED:
      // Nothing to do in this case.
      break;
    }
    return *this;
  }

  Expected(Expected &&That) noexcept : TheState(That.TheState) {
    switch (TheState) {
    case State::SUCCESS:
      new (&TheValue) T(std::move(That.TheValue));
      break;
    case State::FAILURE:
      new (&TheError) Status(std::move(That.TheError));
      break;
    case State::MOVED:
      // Nothing to do in this case.
      break;
    }
    That.TheState = State::MOVED;
  }

  template <typename U>
  Expected(const Expected<U> &That) : TheState(That.TheState) {
    switch (TheState) {
    case State::SUCCESS:
      new (&TheValue) T(That.TheValue);
      break;
    case State::FAILURE:
      new (&TheError) Status(That.TheError);
      break;
    case State::MOVED:
      // Nothing to do in this case.
      break;
    }
  }

  template <typename U> Expected(Expected<U> &&That) : TheState(That.TheState) {
    switch (TheState) {
    case State::SUCCESS:
      new (&TheValue) T(std::move(That.TheValue));
      break;
    case State::FAILURE:
      new (&TheError) Status(std::move(That.TheError));
      break;
    case State::MOVED:
      // Nothing to do in this case.
      break;
    }
  }

  ~Expected() {
    switch (TheState) {
    case State::SUCCESS:
      TheValue.~T();
      break;
    case State::FAILURE:
      TheError.~Status();
      break;
    case State::MOVED:
      // Nothing to do for this case.
      break;
    }
  }

  /// Returns true if this instance represents failure.
  bool isError() const { return TheState != State::SUCCESS; }

  /// Gets a reference to the Status object.
  ///
  /// Should only be called if isError() returns true.
  const Status &getError() const {
    assert(isError());
    return TheError;
  }

  /// Gets a const reference to the value object.
  ///
  /// Should only be called if isError() returns false.
  const T &getValue() const {
    assert(!isError());
    return TheValue;
  }

  /// Gets a reference to the value object.
  ///
  /// Should only be called if isError() returns false.
  T &getValue() {
    assert(!isError());
    return TheValue;
  }

  /// Takes the value from this object by moving it to the return value.
  ///
  /// Should only be called if isError() returns false.
  T takeValue() {
    assert(!isError());
    TheState = State::MOVED;
    return std::move(TheValue);
  }

private:
  template <typename U> friend class Expected;

  State TheState;

  union {
    T TheValue;
    Status TheError;
  };
};

} // namespace acxxel

#endif // ACXXEL_STATUS_H
