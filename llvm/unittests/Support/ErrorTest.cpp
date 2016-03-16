//===----- unittests/ErrorTest.cpp - Error.h tests ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Error.h"
#include "llvm/Support/Errc.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {

// Test:
//
// Constructing success.
//   - Silent acceptance of tested success.
//   - Programmatic error on untested success.
//
// Custom error class.
//   - Creation of a custom error class with a default base class.
//   - Handling of a custom error class with a default base class.
//   - Handler type deduction.
//   - Failure to handle a custom error class.
//   - Isa testing of a custom error class.
//
//   - Creation of a custom error class with a custom base class.
//   - Handling of a custom error class with a default base class.
//   - Correct shadowing of handlers.
//
// Utility functions:
//   - join_errors to defer errors.
//   - consume_error to consume a "safe" error without any output.
//   - handleAllUnhandledErrors to assert that all errors are handled.
//   - logAllUnhandledErrors to log errors to a stream.
//
// Expected tests:
//   - Expected<T> with T.
//   - Expected<T> with Error.
//   - Failure to handle an Expected<T> in failure mode.
//   - Error extraction (Expected -> Error).
//
// std::error_code:
//   - std::error_code to Error in success mode.
//   - std::error_code to Error (ECError) in failure mode.
//   - Error to std::error_code in success mode.
//   - Error (ECError) to std::error_code in failure mode.

// Custom error class with a default base class and some random 'info' attached.
class CustomError : public ErrorInfo<CustomError> {
public:
  // Create an error with some info attached.
  CustomError(int Info) : Info(Info) {}

  // Get the info attached to this error.
  int getInfo() const { return Info; }

  // Log this error to a stream.
  void log(raw_ostream &OS) const override {
    OS << "CustomError { " << getInfo() << "}";
  }

protected:
  // This error is subclassed below, but we can't use inheriting constructors
  // yet, so we can't propagate the constructors through ErrorInfo. Instead
  // we have to have a default constructor and have the subclass initialize all
  // fields.
  CustomError() : Info(0) {}

  int Info;
};

// Custom error class with a custom base class and some additional random
// 'info'.
class CustomSubError : public ErrorInfo<CustomSubError, CustomError> {
public:
  // Create a sub-error with some info attached.
  CustomSubError(int Info, int ExtraInfo) : ExtraInfo(ExtraInfo) {
    this->Info = Info;
  }

  // Get the extra info attached to this error.
  int getExtraInfo() const { return ExtraInfo; }

  // Log this error to a stream.
  void log(raw_ostream &OS) const override {
    OS << "CustomSubError { " << getInfo() << ", " << getExtraInfo() << "}";
  }

protected:
  int ExtraInfo;
};

// Verify that success values that are checked (i.e. cast to 'bool') are
// destructed without error, and that unchecked success values cause an
// abort.
TEST(Error, CheckSuccess) {
  // Test checked success.
  {
    Error E;
    EXPECT_FALSE(E) << "Unexpected error while testing Error 'Success'";
  }

// Test unchecked success.
// Test runs in debug mode only.
#ifndef NDEBUG
  {
    auto DropUncheckedSuccess = []() { Error E; };
    EXPECT_DEATH(DropUncheckedSuccess(),
                 "Program aborted due to an unhandled Error:")
        << "Unchecked Error Succes value did not cause abort()";
  }
#endif
}

static Error handleCustomError(const CustomError &CE) { return Error(); }

static void handleCustomErrorVoid(const CustomError &CE) {}

static Error handleCustomErrorUP(std::unique_ptr<CustomError> CE) {
  return Error();
}

static void handleCustomErrorUPVoid(std::unique_ptr<CustomError> CE) {}

// Verify creation and handling of custom error classes.
TEST(Error, CheckCustomErrors) {
// Check that we abort on unhandled failure cases. (Force conversion to bool
// to make sure that we don't accidentally treat checked errors as handled).
// Test runs in debug mode only.
#ifndef NDEBUG
  {
    auto DropUnhandledError = []() {
      Error E = make_error<CustomError>(42);
      (void)!E;
    };
    EXPECT_DEATH(DropUnhandledError(),
                 "Program aborted due to an unhandled Error:")
        << "Unhandled Error failure value did not cause abort()";
  }
#endif

  // Check 'isA' handling.
  {
    Error E = make_error<CustomError>(1);
    Error F = make_error<CustomSubError>(1, 2);

    EXPECT_TRUE(E.isA<CustomError>());
    EXPECT_FALSE(E.isA<CustomSubError>());
    EXPECT_TRUE(F.isA<CustomError>());
    EXPECT_TRUE(F.isA<CustomSubError>());

    consumeError(std::move(E));
    consumeError(std::move(F));
  }

  // Check that we can handle a custom error.
  {
    int CaughtErrorInfo = 0;
    handleAllErrors(make_error<CustomError>(42), [&](const CustomError &CE) {
      CaughtErrorInfo = CE.getInfo();
    });

    EXPECT_TRUE(CaughtErrorInfo == 42)
        << "Wrong result from CustomError handler";
  }

  // Check that handler type deduction also works for handlers
  // of the following types:
  // void (const Err&)
  // Error (const Err&) mutable
  // void (const Err&) mutable
  // Error (Err&)
  // void (Err&)
  // Error (Err&) mutable
  // void (Err&) mutable
  // Error (unique_ptr<Err>)
  // void (unique_ptr<Err>)
  // Error (unique_ptr<Err>) mutable
  // void (unique_ptr<Err>) mutable

  handleAllErrors(make_error<CustomError>(42), [](const CustomError &CE) {});

  handleAllErrors(
      make_error<CustomError>(42),
      [](const CustomError &CE) mutable { return Error::success(); });

  handleAllErrors(make_error<CustomError>(42),
                  [](const CustomError &CE) mutable {});

  handleAllErrors(make_error<CustomError>(42),
                  [](CustomError &CE) { return Error::success(); });

  handleAllErrors(make_error<CustomError>(42), [](CustomError &CE) {});

  handleAllErrors(make_error<CustomError>(42),
                  [](CustomError &CE) mutable { return Error::success(); });

  handleAllErrors(make_error<CustomError>(42), [](CustomError &CE) mutable {});

  handleAllErrors(
      make_error<CustomError>(42),
      [](std::unique_ptr<CustomError> CE) { return Error::success(); });

  handleAllErrors(make_error<CustomError>(42),
                  [](std::unique_ptr<CustomError> CE) {});

  handleAllErrors(
      make_error<CustomError>(42),
      [](std::unique_ptr<CustomError> CE) mutable { return Error::success(); });

  handleAllErrors(make_error<CustomError>(42),
                  [](std::unique_ptr<CustomError> CE) mutable {});

  // Check that named handlers of type 'Error (const Err&)' work.
  handleAllErrors(make_error<CustomError>(42), handleCustomError);

  // Check that named handlers of type 'void (const Err&)' work.
  handleAllErrors(make_error<CustomError>(42), handleCustomErrorVoid);

  // Check that named handlers of type 'Error (std::unique_ptr<Err>)' work.
  handleAllErrors(make_error<CustomError>(42), handleCustomErrorUP);

  // Check that named handlers of type 'Error (std::unique_ptr<Err>)' work.
  handleAllErrors(make_error<CustomError>(42), handleCustomErrorUPVoid);

  // Check that we can handle a custom error with a custom base class.
  {
    int CaughtErrorInfo = 0;
    int CaughtErrorExtraInfo = 0;
    handleAllErrors(make_error<CustomSubError>(42, 7),
                    [&](const CustomSubError &SE) {
                      CaughtErrorInfo = SE.getInfo();
                      CaughtErrorExtraInfo = SE.getExtraInfo();
                    });

    EXPECT_TRUE(CaughtErrorInfo == 42 && CaughtErrorExtraInfo == 7)
        << "Wrong result from CustomSubError handler";
  }

  // Check that we trigger only the first handler that applies.
  {
    int DummyInfo = 0;
    int CaughtErrorInfo = 0;
    int CaughtErrorExtraInfo = 0;

    handleAllErrors(make_error<CustomSubError>(42, 7),
                    [&](const CustomSubError &SE) {
                      CaughtErrorInfo = SE.getInfo();
                      CaughtErrorExtraInfo = SE.getExtraInfo();
                    },
                    [&](const CustomError &CE) { DummyInfo = CE.getInfo(); });

    EXPECT_TRUE(CaughtErrorInfo == 42 && CaughtErrorExtraInfo == 7 &&
                DummyInfo == 0)
        << "Activated the wrong Error handler(s)";
  }

  // Check that general handlers shadow specific ones.
  {
    int CaughtErrorInfo = 0;
    int DummyInfo = 0;
    int DummyExtraInfo = 0;

    handleAllErrors(
        make_error<CustomSubError>(42, 7),
        [&](const CustomError &CE) { CaughtErrorInfo = CE.getInfo(); },
        [&](const CustomSubError &SE) {
          DummyInfo = SE.getInfo();
          DummyExtraInfo = SE.getExtraInfo();
        });

    EXPECT_TRUE(CaughtErrorInfo = 42 && DummyInfo == 0 && DummyExtraInfo == 0)
        << "General Error handler did not shadow specific handler";
  }
}

// Test utility functions.
TEST(Error, CheckErrorUtilities) {

  // Test joinErrors
  {
    int CustomErrorInfo1 = 0;
    int CustomErrorInfo2 = 0;
    int CustomErrorExtraInfo = 0;
    Error E = joinErrors(make_error<CustomError>(7),
                         make_error<CustomSubError>(42, 7));

    handleAllErrors(std::move(E),
                    [&](const CustomSubError &SE) {
                      CustomErrorInfo2 = SE.getInfo();
                      CustomErrorExtraInfo = SE.getExtraInfo();
                    },
                    [&](const CustomError &CE) {
                      // Assert that the CustomError instance above is handled
                      // before the
                      // CustomSubError - joinErrors should preserve error
                      // ordering.
                      EXPECT_EQ(CustomErrorInfo2, 0)
                          << "CustomErrorInfo2 should be 0 here. "
                             "joinErrors failed to preserve ordering.\n";
                      CustomErrorInfo1 = CE.getInfo();
                    });

    EXPECT_TRUE(CustomErrorInfo1 == 7 && CustomErrorInfo2 == 42 &&
                CustomErrorExtraInfo == 7)
        << "Failed handling compound Error.";
  }

  // Test consumeError for both success and error cases.
  {
    Error E;
    consumeError(std::move(E));
  }
  {
    Error E = make_error<CustomError>(7);
    consumeError(std::move(E));
  }

// Test that handleAllUnhandledErrors crashes if an error is not caught.
// Test runs in debug mode only.
#ifndef NDEBUG
  {
    auto FailToHandle = []() {
      handleAllErrors(make_error<CustomError>(7),
                      [&](const CustomSubError &SE) {
                        errs() << "This should never be called";
                        exit(1);
                      });
    };

    EXPECT_DEATH(FailToHandle(), "Program aborted due to an unhandled Error:")
        << "Unhandled Error in handleAllErrors call did not cause an "
           "abort()";
  }
#endif

// Test that handleAllUnhandledErrors crashes if an error is returned from a
// handler.
// Test runs in debug mode only.
#ifndef NDEBUG
  {
    auto ReturnErrorFromHandler = []() {
      handleAllErrors(make_error<CustomError>(7),
                      [&](std::unique_ptr<CustomSubError> SE) {
                        return Error(std::move(SE));
                      });
    };

    EXPECT_DEATH(ReturnErrorFromHandler(),
                 "Program aborted due to an unhandled Error:")
        << " Error returned from handler in handleAllErrors call did not "
           "cause abort()";
  }
#endif

  // Test that we can return values from handleErrors.
  {
    int ErrorInfo = 0;

    Error E = handleErrors(
        make_error<CustomError>(7),
        [&](std::unique_ptr<CustomError> CE) { return Error(std::move(CE)); });

    handleAllErrors(std::move(E),
                    [&](const CustomError &CE) { ErrorInfo = CE.getInfo(); });

    EXPECT_EQ(ErrorInfo, 7)
        << "Failed to handle Error returned from handleErrors.";
  }
}

// Test Expected behavior.
TEST(Error, CheckExpected) {

  // Check that non-errors convert to 'true'.
  {
    Expected<int> A = 7;
    EXPECT_TRUE(!!A)
        << "Expected with non-error value doesn't convert to 'true'";
  }

  // Check that non-error values are accessible via operator*.
  {
    Expected<int> A = 7;
    EXPECT_EQ(*A, 7) << "Incorrect Expected non-error value";
  }

  // Check that errors convert to 'false'.
  {
    Expected<int> A = make_error<CustomError>(42);
    EXPECT_FALSE(!!A) << "Expected with error value doesn't convert to 'false'";
    consumeError(A.takeError());
  }

  // Check that error values are accessible via takeError().
  {
    Expected<int> A = make_error<CustomError>(42);
    Error E = A.takeError();
    EXPECT_TRUE(E.isA<CustomError>()) << "Incorrect Expected error value";
    consumeError(std::move(E));
  }

// Check that an Expected instance with an error value doesn't allow access to
// operator*.
// Test runs in debug mode only.
#ifndef NDEBUG
  {
    Expected<int> A = make_error<CustomError>(42);
    EXPECT_DEATH(*A, "!HasError && \"Cannot get value when an error exists!\"")
        << "Incorrect Expected error value";
    consumeError(A.takeError());
  }
#endif

// Check that an Expected instance with an error triggers an abort if
// unhandled.
// Test runs in debug mode only.
#ifndef NDEBUG
  EXPECT_DEATH({ Expected<int> A = make_error<CustomError>(42); },
               "Program aborted due to an unhandled Error:")
      << "Unchecked Expected<T> failure value did not cause an abort()";
#endif

  // Test covariance of Expected.
  {
    class B {};
    class D : public B {};

    Expected<B *> A1(Expected<D *>(nullptr));
    A1 = Expected<D *>(nullptr);

    Expected<std::unique_ptr<B>> A2(Expected<std::unique_ptr<D>>(nullptr));
    A2 = Expected<std::unique_ptr<D>>(nullptr);
  }
}

TEST(Error, ECError) {

  // Round-trip a success value to check that it converts correctly.
  EXPECT_EQ(errorToErrorCode(errorCodeToError(std::error_code())),
            std::error_code())
      << "std::error_code() should round-trip via Error conversions";

  // Round-trip an error value to check that it converts correctly.
  EXPECT_EQ(errorToErrorCode(errorCodeToError(errc::invalid_argument)),
            errc::invalid_argument)
      << "std::error_code error value should round-trip via Error "
         "conversions";
}

} // end anon namespace
