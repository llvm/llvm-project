//===-------- Error.h - Enforced error checking for ORC RT ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_ERROR_H
#define ORC_RT_ERROR_H

#include "orc-rt/Compiler.h"
#include "orc-rt/RTTI.h"

#include <cassert>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>

namespace orc_rt {

/// Base class for all errors.
class ErrorInfoBase : public RTTIExtends<ErrorInfoBase, RTTIRoot> {
public:
  virtual std::string toString() const = 0;
};

/// Represents an environmental error.
class ORC_RT_NODISCARD Error {

  template <typename T> friend class Expected;

  friend Error make_error(std::unique_ptr<ErrorInfoBase> Payload);

  template <typename... HandlerTs>
  friend Error handleErrors(Error E, HandlerTs &&...Hs);

public:
  /// Destroy this error. Aborts if error was not checked, or was checked but
  /// not handled.
  ~Error() { assertIsChecked(); }

  Error(const Error &) = delete;
  Error &operator=(const Error &) = delete;

  /// Move-construct an error. The newly constructed error is considered
  /// unchecked, even if the source error had been checked. The original error
  /// becomes a checked success value.
  Error(Error &&Other) {
    setChecked(true);
    *this = std::move(Other);
  }

  /// Move-assign an error value. The current error must represent success, you
  /// you cannot overwrite an unhandled error. The current error is then
  /// considered unchecked. The source error becomes a checked success value,
  /// regardless of its original state.
  Error &operator=(Error &&Other) {
    // Don't allow overwriting of unchecked values.
    assertIsChecked();
    setPtr(Other.getPtr());

    // This Error is unchecked, even if the source error was checked.
    setChecked(false);

    // Null out Other's payload and set its checked bit.
    Other.setPtr(nullptr);
    Other.setChecked(true);

    return *this;
  }

  /// Create a success value.
  static Error success() { return Error(); }

  /// Error values convert to true for failure values, false otherwise.
  explicit operator bool() {
    setChecked(getPtr() == nullptr);
    return getPtr() != nullptr;
  }

  /// Return true if this Error contains a failure value of the given type.
  template <typename ErrT> bool isA() const {
    return getPtr() && getPtr()->isA<ErrT>();
  }

private:
  Error() = default;

  Error(std::unique_ptr<ErrorInfoBase> ErrInfo) {
    auto RawErrPtr = reinterpret_cast<uintptr_t>(ErrInfo.release());
    assert((RawErrPtr & 0x1) == 0 && "ErrorInfo is insufficiently aligned");
    ErrPtr = RawErrPtr | 0x1;
  }

  void assertIsChecked() {
    if (ORC_RT_UNLIKELY(!isChecked() || getPtr())) {
      fprintf(stderr, "Error must be checked prior to destruction.\n");
      abort(); // Some sort of JIT program abort?
    }
  }

  template <typename ErrT = ErrorInfoBase> ErrT *getPtr() const {
    return reinterpret_cast<ErrT *>(ErrPtr & ~uintptr_t(1));
  }

  void setPtr(ErrorInfoBase *Ptr) {
    ErrPtr = (reinterpret_cast<uintptr_t>(Ptr) & ~uintptr_t(1)) | (ErrPtr & 1);
  }

  bool isChecked() const { return ErrPtr & 0x1; }

  void setChecked(bool Checked) { ErrPtr = (ErrPtr & ~uintptr_t(1)) | Checked; }

  template <typename ErrT = ErrorInfoBase> std::unique_ptr<ErrT> takePayload() {
    static_assert(std::is_base_of_v<ErrorInfoBase, ErrT>,
                  "ErrT is not an ErrorInfoBase subclass");
    std::unique_ptr<ErrT> Tmp(getPtr<ErrT>());
    setPtr(nullptr);
    setChecked(true);
    return Tmp;
  }

  uintptr_t ErrPtr = 0;
};

/// Create an Error from an ErrorInfoBase.
inline Error make_error(std::unique_ptr<ErrorInfoBase> Payload) {
  return Error(std::move(Payload));
}

/// Construct an error of ErrT with the given arguments.
template <typename ErrT, typename... ArgTs> Error make_error(ArgTs &&...Args) {
  static_assert(std::is_base_of<ErrorInfoBase, ErrT>::value,
                "ErrT is not an ErrorInfoBase subclass");
  return make_error(std::make_unique<ErrT>(std::forward<ArgTs>(Args)...));
}

/// Traits class for selecting and applying error handlers.
template <typename HandlerT>
struct ErrorHandlerTraits
    : public ErrorHandlerTraits<
          decltype(&std::remove_reference_t<HandlerT>::operator())> {};

// Specialization functions of the form 'Error (const ErrT&)'.
template <typename ErrT> struct ErrorHandlerTraits<Error (&)(ErrT &)> {
  static bool appliesTo(const ErrorInfoBase &E) {
    return E.template isA<ErrT>();
  }

  template <typename HandlerT>
  static Error apply(HandlerT &&H, std::unique_ptr<ErrorInfoBase> E) {
    assert(appliesTo(*E) && "Applying incorrect handler");
    return H(static_cast<ErrT &>(*E));
  }
};

// Specialization functions of the form 'void (const ErrT&)'.
template <typename ErrT> struct ErrorHandlerTraits<void (&)(ErrT &)> {
  static bool appliesTo(const ErrorInfoBase &E) {
    return E.template isA<ErrT>();
  }

  template <typename HandlerT>
  static Error apply(HandlerT &&H, std::unique_ptr<ErrorInfoBase> E) {
    assert(appliesTo(*E) && "Applying incorrect handler");
    H(static_cast<ErrT &>(*E));
    return Error::success();
  }
};

/// Specialization for functions of the form 'Error (std::unique_ptr<ErrT>)'.
template <typename ErrT>
struct ErrorHandlerTraits<Error (&)(std::unique_ptr<ErrT>)> {
  static bool appliesTo(const ErrorInfoBase &E) {
    return E.template isA<ErrT>();
  }

  template <typename HandlerT>
  static Error apply(HandlerT &&H, std::unique_ptr<ErrorInfoBase> E) {
    assert(appliesTo(*E) && "Applying incorrect handler");
    std::unique_ptr<ErrT> SubE(static_cast<ErrT *>(E.release()));
    return H(std::move(SubE));
  }
};

/// Specialization for functions of the form 'void (std::unique_ptr<ErrT>)'.
template <typename ErrT>
struct ErrorHandlerTraits<void (&)(std::unique_ptr<ErrT>)> {
  static bool appliesTo(const ErrorInfoBase &E) {
    return E.template isA<ErrT>();
  }

  template <typename HandlerT>
  static Error apply(HandlerT &&H, std::unique_ptr<ErrorInfoBase> E) {
    assert(appliesTo(*E) && "Applying incorrect handler");
    std::unique_ptr<ErrT> SubE(static_cast<ErrT *>(E.release()));
    H(std::move(SubE));
    return Error::success();
  }
};

// Specialization for member functions of the form 'RetT (const ErrT&)'.
template <typename C, typename RetT, typename ErrT>
class ErrorHandlerTraits<RetT (C::*)(ErrT &)>
    : public ErrorHandlerTraits<RetT (&)(ErrT &)> {};

// Specialization for member functions of the form 'RetT (const ErrT&) const'.
template <typename C, typename RetT, typename ErrT>
class ErrorHandlerTraits<RetT (C::*)(ErrT &) const>
    : public ErrorHandlerTraits<RetT (&)(ErrT &)> {};

// Specialization for member functions of the form 'RetT (const ErrT&)'.
template <typename C, typename RetT, typename ErrT>
class ErrorHandlerTraits<RetT (C::*)(const ErrT &)>
    : public ErrorHandlerTraits<RetT (&)(ErrT &)> {};

// Specialization for member functions of the form 'RetT (const ErrT&) const'.
template <typename C, typename RetT, typename ErrT>
class ErrorHandlerTraits<RetT (C::*)(const ErrT &) const>
    : public ErrorHandlerTraits<RetT (&)(ErrT &)> {};

/// Specialization for member functions of the form
/// 'RetT (std::unique_ptr<ErrT>)'.
template <typename C, typename RetT, typename ErrT>
class ErrorHandlerTraits<RetT (C::*)(std::unique_ptr<ErrT>)>
    : public ErrorHandlerTraits<RetT (&)(std::unique_ptr<ErrT>)> {};

/// Specialization for member functions of the form
/// 'RetT (std::unique_ptr<ErrT>) const'.
template <typename C, typename RetT, typename ErrT>
class ErrorHandlerTraits<RetT (C::*)(std::unique_ptr<ErrT>) const>
    : public ErrorHandlerTraits<RetT (&)(std::unique_ptr<ErrT>)> {};

inline Error handleErrorsImpl(std::unique_ptr<ErrorInfoBase> Payload) {
  return make_error(std::move(Payload));
}

template <typename HandlerT, typename... HandlerTs>
Error handleErrorsImpl(std::unique_ptr<ErrorInfoBase> Payload,
                       HandlerT &&Handler, HandlerTs &&...Handlers) {
  if (ErrorHandlerTraits<HandlerT>::appliesTo(*Payload))
    return ErrorHandlerTraits<HandlerT>::apply(std::forward<HandlerT>(Handler),
                                               std::move(Payload));
  return handleErrorsImpl(std::move(Payload),
                          std::forward<HandlerTs>(Handlers)...);
}

/// Pass the ErrorInfo(s) contained in E to their respective handlers. Any
/// unhandled errors (or Errors returned by handlers) are re-concatenated and
/// returned.
/// Because this function returns an error, its result must also be checked
/// or returned. If you intend to handle all errors use handleAllErrors
/// (which returns void, and will abort() on unhandled errors) instead.
template <typename... HandlerTs>
Error handleErrors(Error E, HandlerTs &&...Hs) {
  if (!E)
    return Error::success();
  return handleErrorsImpl(E.takePayload(), std::forward<HandlerTs>(Hs)...);
}

/// Behaves the same as handleErrors, except that by contract all errors
/// *must* be handled by the given handlers (i.e. there must be no remaining
/// errors after running the handlers, or llvm_unreachable is called).
template <typename... HandlerTs>
void handleAllErrors(Error E, HandlerTs &&...Handlers) {
  cantFail(handleErrors(std::move(E), std::forward<HandlerTs>(Handlers)...));
}

/// Helper for Errors used as out-parameters.
/// Sets the 'checked' flag on construction, resets it on destruction.
class ErrorAsOutParameter {
public:
  ErrorAsOutParameter(Error *Err) : Err(Err) {
    // Raise the checked bit if Err is success.
    if (Err)
      (void)!!*Err;
  }

  ~ErrorAsOutParameter() {
    // Clear the checked bit.
    if (Err && !*Err)
      *Err = Error::success();
  }

private:
  Error *Err;
};

template <typename T> class ORC_RT_NODISCARD Expected {

  template <class OtherT> friend class Expected;

  static constexpr bool IsRef = std::is_reference_v<T>;
  using wrap = std::reference_wrapper<std::remove_reference_t<T>>;
  using error_type = std::unique_ptr<ErrorInfoBase>;
  using storage_type = std::conditional_t<IsRef, wrap, T>;
  using value_type = T;

  using reference = std::remove_reference_t<T> &;
  using const_reference = const std::remove_reference_t<T> &;
  using pointer = std::remove_reference_t<T> *;
  using const_pointer = const std::remove_reference_t<T> *;

public:
  /// Create an Expected from a failure value.
  Expected(Error Err) : HasError(true), Unchecked(true) {
    assert(Err && "Cannot create Expected<T> from Error success value");
    new (getErrorStorage()) error_type(Err.takePayload());
  }

  /// Create an Expected from a T value.
  template <typename OtherT>
  Expected(OtherT &&Val,
           std::enable_if_t<std::is_convertible_v<OtherT, T>> * = nullptr)
      : HasError(false), Unchecked(true) {
    new (getStorage()) storage_type(std::forward<OtherT>(Val));
  }

  /// Move-construct an Expected<T> from an Expected<OtherT>.
  Expected(Expected &&Other) { moveConstruct(std::move(Other)); }

  /// Move construct an Expected<T> value from an Expected<OtherT>, where OtherT
  /// must be convertible to T.
  template <class OtherT>
  Expected(Expected<OtherT> &&Other,
           std::enable_if_t<std::is_convertible_v<OtherT, T>> * = nullptr) {
    moveConstruct(std::move(Other));
  }

  /// Move construct an Expected<T> value from an Expected<OtherT>, where OtherT
  /// isn't convertible to T.
  template <class OtherT>
  explicit Expected(
      Expected<OtherT> &&Other,
      std::enable_if_t<!std::is_convertible_v<OtherT, T>> * = nullptr) {
    moveConstruct(std::move(Other));
  }

  /// Move-assign from another Expected<T>.
  Expected &operator=(Expected &&Other) {
    moveAssign(std::move(Other));
    return *this;
  }

  /// Destroy an Expected<T>.
  ~Expected() {
    assertIsChecked();
    if (!HasError)
      getStorage()->~storage_type();
    else
      getErrorStorage()->~error_type();
  }

  /// Returns true if this Expected value is in a success state (holding a T),
  /// and false if this Expected value is in a failure state.
  explicit operator bool() {
    Unchecked = HasError;
    return !HasError;
  }

  /// Returns true if this Expected value holds an Error of type error_type.
  template <typename ErrT> bool isFailureOfType() const {
    return HasError && (*getErrorStorage())->template isFailureOfType<ErrT>();
  }

  /// Take ownership of the stored error.
  ///
  /// If this Expected value is in a success state (holding a T) then this
  /// method is a no-op and returns Error::success.
  ///
  /// If thsi Expected value is in a failure state (holding an Error) then this
  /// method returns the contained error and leaves this Expected in an
  /// 'empty' state from which it may be safely destructed but not otherwise
  /// accessed.
  Error takeError() {
    Unchecked = false;
    return HasError ? Error(std::move(*getErrorStorage())) : Error::success();
  }

  /// Returns a pointer to the stored T value.
  pointer operator->() {
    assertIsChecked();
    return toPointer(getStorage());
  }

  /// Returns a pointer to the stored T value.
  const_pointer operator->() const {
    assertIsChecked();
    return toPointer(getStorage());
  }

  /// Returns a reference to the stored T value.
  reference operator*() {
    assertIsChecked();
    return *getStorage();
  }

  /// Returns a reference to the stored T value.
  const_reference operator*() const {
    assertIsChecked();
    return *getStorage();
  }

private:
  template <class T1>
  static bool compareThisIfSameType(const T1 &a, const T1 &b) {
    return &a == &b;
  }

  template <class T1, class T2>
  static bool compareThisIfSameType(const T1 &a, const T2 &b) {
    return false;
  }

  template <class OtherT> void moveConstruct(Expected<OtherT> &&Other) {
    HasError = Other.HasError;
    Unchecked = true;
    Other.Unchecked = false;

    if (!HasError)
      new (getStorage()) storage_type(std::move(*Other.getStorage()));
    else
      new (getErrorStorage()) error_type(std::move(*Other.getErrorStorage()));
  }

  template <class OtherT> void moveAssign(Expected<OtherT> &&Other) {
    assertIsChecked();

    if (compareThisIfSameType(*this, Other))
      return;

    this->~Expected();
    new (this) Expected(std::move(Other));
  }

  pointer toPointer(pointer Val) { return Val; }

  const_pointer toPointer(const_pointer Val) const { return Val; }

  pointer toPointer(wrap *Val) { return &Val->get(); }

  const_pointer toPointer(const wrap *Val) const { return &Val->get(); }

  storage_type *getStorage() {
    assert(!HasError && "Cannot get value when an error exists!");
    return reinterpret_cast<storage_type *>(&TStorage);
  }

  const storage_type *getStorage() const {
    assert(!HasError && "Cannot get value when an error exists!");
    return reinterpret_cast<const storage_type *>(&TStorage);
  }

  error_type *getErrorStorage() {
    assert(HasError && "Cannot get error when a value exists!");
    return reinterpret_cast<error_type *>(&ErrorStorage);
  }

  const error_type *getErrorStorage() const {
    assert(HasError && "Cannot get error when a value exists!");
    return reinterpret_cast<const error_type *>(&ErrorStorage);
  }

  void assertIsChecked() {
    if (ORC_RT_UNLIKELY(Unchecked)) {
      fprintf(stderr,
              "Expected<T> must be checked before access or destruction.\n");
      abort();
    }
  }

  union {
    alignas(storage_type) char TStorage[sizeof(storage_type)];
    alignas(error_type) char ErrorStorage[sizeof(error_type)];
  };

  bool HasError : 1;
  bool Unchecked : 1;
};

/// Consume an error without doing anything.
inline void consumeError(Error Err) {
  handleAllErrors(std::move(Err), [](const ErrorInfoBase &) {});
}

/// Consumes success values. It is a programmatic error to call this function
/// on a failure value.
inline void cantFail(Error Err) {
#ifndef NDEBUG
  // TODO: Log unhandled error.
  if (Err)
    abort();
#else
  Err.operator bool(); // Reset checked flag.
#endif
}

/// Auto-unwrap an Expected<T> value in the success state. It is a programmatic
/// error to call this function on a failure value.
template <typename T> T cantFail(Expected<T> E) {
  assert(E && "cantFail called on failure value");
  consumeError(E.takeError());
  return std::move(*E);
}

/// Auto-unwrap an Expected<T> value in the success state. It is a programmatic
/// error to call this function on a failure value.
template <typename T> T &cantFail(Expected<T &> E) {
  assert(E && "cantFail called on failure value");
  consumeError(E.takeError());
  return *E;
}

/// Convert the given error to a string. The error value is consumed in the
/// process.
inline std::string toString(Error Err) {
  assert(Err && "Cannot convert success value to string");
  std::string ErrMsg;
  handleAllErrors(std::move(Err),
                  [&](const ErrorInfoBase &EIB) { ErrMsg = EIB.toString(); });
  return ErrMsg;
}

/// Simple string error type.
class StringError : public RTTIExtends<StringError, ErrorInfoBase> {
public:
  StringError(std::string ErrMsg) : ErrMsg(std::move(ErrMsg)) {}
  std::string toString() const override { return ErrMsg; }

private:
  std::string ErrMsg;
};

} // namespace orc_rt

#endif // ORC_RT_ERROR_H
