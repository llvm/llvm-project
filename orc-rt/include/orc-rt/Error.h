//===-------- Error.h - Enforced error checking for ORC RT ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_ERROR_H
#define ORC_RT_ERROR_H

#include "orc-rt-c/config.h"
#include "orc-rt/CallableTraitsHelper.h"
#include "orc-rt/Compiler.h"
#include "orc-rt/RTTI.h"

#include <cassert>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>

#if ORC_RT_ENABLE_EXCEPTIONS
#include <exception>
#endif // ORC_RT_ENABLE_EXCEPTIONS

namespace orc_rt {

class Error;

/// Base class for all errors.
class ErrorInfoBase : public RTTIExtends<ErrorInfoBase, RTTIRoot> {
public:
  virtual std::string toString() const noexcept = 0;

private:
#if ORC_RT_ENABLE_EXCEPTIONS
  friend class Error;
  friend Error restore_error(ErrorInfoBase &&);

  virtual void throwAsException() = 0;

  virtual Error restoreError() noexcept = 0;
#endif // ORC_RT_ENABLE_EXCEPTIONS
};

/// Like RTTI-extends, but injects error-related helper methods.
template <typename ThisT, typename ParentT>
class ErrorExtends : public ParentT {
public:
  static_assert(std::is_base_of_v<ErrorInfoBase, ParentT>,
                "ErrorExtends must extend ErrorInfoBase derivatives");

  // Inherit constructors and isA methods from ParentT.
  using ParentT::isA;
  using ParentT::ParentT;

  static char ID;

  static const void *classID() noexcept { return &ThisT::ID; }

  const void *dynamicClassID() const noexcept override { return &ThisT::ID; }

  bool isA(const void *const ClassID) const noexcept override {
    return ClassID == classID() || ParentT::isA(ClassID);
  }

  static bool classof(const RTTIRoot *R) { return R->isA<ThisT>(); }

#if ORC_RT_ENABLE_EXCEPTIONS
  void throwAsException() override {
    throw ThisT(std::move(static_cast<ThisT &>(*this)));
  }

  Error restoreError() noexcept override;
#endif // ORC_RT_ENABLE_EXCEPTIONS
};

template <typename ThisT, typename ParentT>
char ErrorExtends<ThisT, ParentT>::ID = 0;

/// Represents an environmental error.
class ORC_RT_NODISCARD Error {

  template <typename T> friend class Expected;

  friend Error make_error(std::unique_ptr<ErrorInfoBase> Payload) noexcept;

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
  Error(Error &&Other) noexcept {
    setChecked(true);
    *this = std::move(Other);
  }

  /// Move-assign an error value. The current error must represent success, you
  /// you cannot overwrite an unhandled error. The current error is then
  /// considered unchecked. The source error becomes a checked success value,
  /// regardless of its original state.
  Error &operator=(Error &&Other) noexcept {
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
  static Error success() noexcept { return Error(); }

  /// Error values convert to true for failure values, false otherwise.
  explicit operator bool() noexcept {
    setChecked(getPtr() == nullptr);
    return getPtr() != nullptr;
  }

  /// Return true if this Error contains a failure value of the given type.
  template <typename ErrT> bool isA() const noexcept {
    return getPtr() && getPtr()->isA<ErrT>();
  }

#if ORC_RT_ENABLE_EXCEPTIONS
  void throwOnFailure() {
    if (auto P = takePayload())
      P->throwAsException();
  }
#endif // ORC_RT_ENABLE_EXCEPTIONS

private:
  Error() noexcept = default;

  Error(std::unique_ptr<ErrorInfoBase> ErrInfo) noexcept {
    auto RawErrPtr = reinterpret_cast<uintptr_t>(ErrInfo.release());
    assert((RawErrPtr & 0x1) == 0 && "ErrorInfo is insufficiently aligned");
    ErrPtr = RawErrPtr | 0x1;
  }

  void assertIsChecked() noexcept {
    if (ORC_RT_UNLIKELY(!isChecked() || getPtr())) {
      fprintf(stderr, "Error must be checked prior to destruction.\n");
      abort(); // Some sort of JIT program abort?
    }
  }

  template <typename ErrT = ErrorInfoBase> ErrT *getPtr() const noexcept {
    return reinterpret_cast<ErrT *>(ErrPtr & ~uintptr_t(1));
  }

  void setPtr(ErrorInfoBase *Ptr) noexcept {
    ErrPtr = (reinterpret_cast<uintptr_t>(Ptr) & ~uintptr_t(1)) | (ErrPtr & 1);
  }

  bool isChecked() const noexcept { return ErrPtr & 0x1; }

  void setChecked(bool Checked) noexcept {
    ErrPtr = (ErrPtr & ~uintptr_t(1)) | Checked;
  }

  template <typename ErrT = ErrorInfoBase>
  std::unique_ptr<ErrT> takePayload() noexcept {
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
inline Error make_error(std::unique_ptr<ErrorInfoBase> Payload) noexcept {
  return Error(std::move(Payload));
}

#if ORC_RT_ENABLE_EXCEPTIONS

template <typename ThisT, typename ParentT>
Error ErrorExtends<ThisT, ParentT>::restoreError() noexcept {
  return make_error(
      std::make_unique<ThisT>(std::move(*static_cast<ThisT *>(this))));
}

inline Error restore_error(ErrorInfoBase &&EIB) { return EIB.restoreError(); }

#endif // ORC_RT_ENABLE_EXCEPTIONS

/// Construct an error of ErrT with the given arguments.
template <typename ErrT, typename... ArgTs> Error make_error(ArgTs &&...Args) {
  static_assert(std::is_base_of<ErrorInfoBase, ErrT>::value,
                "ErrT is not an ErrorInfoBase subclass");
  return make_error(std::make_unique<ErrT>(std::forward<ArgTs>(Args)...));
}

namespace detail {

template <typename RetT, typename ArgT> struct ErrorHandlerTraitsImpl;

// Specialization for Error(ErrT&).
template <typename ErrT> struct ErrorHandlerTraitsImpl<Error, ErrT &> {
  static bool appliesTo(const ErrorInfoBase &E) {
    return E.template isA<ErrT>();
  }
  template <typename HandlerT>
  static Error apply(HandlerT &&H, std::unique_ptr<ErrorInfoBase> E) {
    assert(appliesTo(*E) && "Applying incorrect handler");
    return H(static_cast<ErrT &>(*E));
  }
};

// Specialization for void(ErrT&).
template <typename ErrT> struct ErrorHandlerTraitsImpl<void, ErrT &> {
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

// Specialization for Error(std::unique_ptr<ErrT>).
template <typename ErrT>
struct ErrorHandlerTraitsImpl<Error, std::unique_ptr<ErrT>> {
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

// Specialization for void(std::unique_ptr<ErrT>).
template <typename ErrT>
struct ErrorHandlerTraitsImpl<void, std::unique_ptr<ErrT>> {
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

} // namespace detail.

template <typename C>
struct ErrorHandlerTraits
    : public CallableTraitsHelper<detail::ErrorHandlerTraitsImpl, C> {};

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

  ErrorAsOutParameter(Error &Err) : Err(&Err) { (void)!!Err; }

  ~ErrorAsOutParameter() {
    // Clear the checked bit.
    if (Err && !*Err)
      *Err = Error::success();
  }

private:
  Error *Err;
};

/// Tag to force construction of an Expected value in the success state. See
/// Expected constructor for details.
struct ForceExpectedSuccessValue {};

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

  template <typename OtherT>
  Expected(OtherT &&Val, ForceExpectedSuccessValue _,
           std::enable_if_t<std::is_convertible_v<OtherT, T>> * = nullptr)
      : HasError(false), Unchecked(true) {
    new (getStorage()) storage_type(std::forward<OtherT>(Val));
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
inline std::string toString(Error Err) noexcept {
  assert(Err && "Cannot convert success value to string");
  std::string ErrMsg;
  handleAllErrors(std::move(Err),
                  [&](const ErrorInfoBase &EIB) { ErrMsg = EIB.toString(); });
  return ErrMsg;
}

/// Simple string error type.
class StringError : public ErrorExtends<StringError, ErrorInfoBase> {
public:
  StringError(std::string ErrMsg) : ErrMsg(std::move(ErrMsg)) {}
  std::string toString() const noexcept override { return ErrMsg; }

private:
  std::string ErrMsg;
};

/// APIs for C++ exception interop.
#if ORC_RT_ENABLE_EXCEPTIONS

class ExceptionError : public ErrorExtends<ExceptionError, ErrorInfoBase> {
public:
  ExceptionError(std::exception_ptr E) : E(std::move(E)) {}
  std::string toString() const noexcept override;
  void throwAsException() override { std::rethrow_exception(E); }

private:
  mutable std::exception_ptr E;
};

namespace detail {

// In general we need to wrap a return type of T with an Expected.
template <typename RetT> struct ErrorWrapImpl {
  typedef Expected<RetT> return_type;

  template <typename OpFn> static return_type run(OpFn &&Op) { return Op(); }
};

// If the return is already an Expected value then we don't need to add
// an additional level of wrapping.
template <typename RetT> struct ErrorWrapImpl<Expected<RetT>> {
  typedef Expected<RetT> return_type;

  template <typename OpFn> static return_type run(OpFn &&Op) { return Op(); }
};

// Errors stay errors.
template <> struct ErrorWrapImpl<Error> {
  typedef Error return_type;

  template <typename OpFn> static return_type run(OpFn &&Op) { return Op(); }
};

// void returns become Error returns.
template <> struct ErrorWrapImpl<void> {
  typedef Error return_type;

  template <typename OpFn> static return_type run(OpFn &&Op) {
    Op();
    return Error::success();
  }
};

template <typename Callable>
struct ErrorWrap
    : public CallableTraitsHelper<detail::ErrorWrapImpl, Callable> {};

} // namespace detail

/// Run the given callback capturing any exceptions thrown into an
/// Error / Expected failure value.
///
/// The return type depends on the return type of the callback:
///   - void callbacks return Error
///   - Error callbacks return Error
///   - Expected<T> callbacks return Expected<T>
///   - other T callbacks return Expected<T>
///
/// If the operation succeeds then...
///   - If its result is non-void it is returned as an Expected<T> success
///     value
///   - If its result is void then Error::success() is retured
///
/// If the operation fails then...
///   - If the exception type is std::unique_ptr<ErrorInfoBase> (i.e. a throw
///     orc_rt failure value) then an Error is constructed to hold the
///     failure value.
///   - If the exception has any other type then it's captured as an
///     ExceptionError.
///
/// The scheme allaws...
///   1. orc_rt::Error values that have been converted to exceptions via
///      Error::throwOnFailure to be converted back into Errors without loss
///      of dynamic type info.
///   2. Other Exceptions caught by this function to be converted back into
///      exceptions via Error::throwOnFailure without loss of dynamic
///      type info.

template <typename OpFn>
typename detail::ErrorWrap<OpFn>::return_type
runCapturingExceptions(OpFn &&Op) noexcept {
  try {
    return detail::ErrorWrap<OpFn>::run(std::forward<OpFn>(Op));
  } catch (ErrorInfoBase &EIB) {
    return restore_error(std::move(EIB));
  } catch (...) {
    return make_error<ExceptionError>(std::current_exception());
  }
}

#endif // ORC_RT_ENABLE_EXCEPTIONS

} // namespace orc_rt

#endif // ORC_RT_ERROR_H
