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
#include "llvm/Support/ErrorHandling.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {

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

  std::error_code convertToErrorCode() const override {
    llvm_unreachable("CustomError doesn't support ECError conversion");
  }

  // Used by ErrorInfo::classID.
  static char ID;

protected:
  // This error is subclassed below, but we can't use inheriting constructors
  // yet, so we can't propagate the constructors through ErrorInfo. Instead
  // we have to have a default constructor and have the subclass initialize all
  // fields.
  CustomError() : Info(0) {}

  int Info;
};

char CustomError::ID = 0;

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

  std::error_code convertToErrorCode() const override {
    llvm_unreachable("CustomSubError doesn't support ECError conversion");
  }

  // Used by ErrorInfo::classID.
  static char ID;

protected:
  int ExtraInfo;
};

char CustomSubError::ID = 0;

static Error handleCustomError(const CustomError &CE) { return Error(); }

static void handleCustomErrorVoid(const CustomError &CE) {}

static Error handleCustomErrorUP(std::unique_ptr<CustomError> CE) {
  return Error();
}

static void handleCustomErrorUPVoid(std::unique_ptr<CustomError> CE) {}

// Test that success values implicitly convert to false, and don't cause crashes
// once they've been implicitly converted.
TEST(Error, CheckedSuccess) {
  Error E;
  EXPECT_FALSE(E) << "Unexpected error while testing Error 'Success'";
}

// Test that unchecked succes values cause an abort.
#ifndef NDEBUG
TEST(Error, UncheckedSuccess) {
  EXPECT_DEATH({ Error E; }, "Program aborted due to an unhandled Error:")
      << "Unchecked Error Succes value did not cause abort()";
}
#endif

// ErrorAsOutParameter tester.
void errAsOutParamHelper(Error &Err) {
  ErrorAsOutParameter ErrAsOutParam(Err);
  // Verify that checked flag is raised - assignment should not crash.
  Err = Error::success();
  // Raise the checked bit manually - caller should still have to test the
  // error.
  (void)!!Err;
}

// Test that ErrorAsOutParameter sets the checked flag on construction.
TEST(Error, ErrorAsOutParameterChecked) {
  Error E;
  errAsOutParamHelper(E);
  (void)!!E;
}

// Test that ErrorAsOutParameter clears the checked flag on destruction.
#ifndef NDEBUG
TEST(Error, ErrorAsOutParameterUnchecked) {
  EXPECT_DEATH({ Error E; errAsOutParamHelper(E); },
               "Program aborted due to an unhandled Error:")
      << "ErrorAsOutParameter did not clear the checked flag on destruction.";
}
#endif

// Check that we abort on unhandled failure cases. (Force conversion to bool
// to make sure that we don't accidentally treat checked errors as handled).
// Test runs in debug mode only.
#ifndef NDEBUG
TEST(Error, UncheckedError) {
  auto DropUnhandledError = []() {
    Error E = make_error<CustomError>(42);
    (void)!E;
  };
  EXPECT_DEATH(DropUnhandledError(),
               "Program aborted due to an unhandled Error:")
      << "Unhandled Error failure value did not cause abort()";
}
#endif

// Check 'Error::isA<T>' method handling.
TEST(Error, IsAHandling) {
  // Check 'isA' handling.
  Error E = make_error<CustomError>(1);
  Error F = make_error<CustomSubError>(1, 2);
  Error G = Error::success();

  EXPECT_TRUE(E.isA<CustomError>());
  EXPECT_FALSE(E.isA<CustomSubError>());
  EXPECT_TRUE(F.isA<CustomError>());
  EXPECT_TRUE(F.isA<CustomSubError>());
  EXPECT_FALSE(G.isA<CustomError>());

  consumeError(std::move(E));
  consumeError(std::move(F));
  consumeError(std::move(G));
}

// Check that we can handle a custom error.
TEST(Error, HandleCustomError) {
  int CaughtErrorInfo = 0;
  handleAllErrors(make_error<CustomError>(42), [&](const CustomError &CE) {
    CaughtErrorInfo = CE.getInfo();
  });

  EXPECT_TRUE(CaughtErrorInfo == 42) << "Wrong result from CustomError handler";
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
TEST(Error, HandlerTypeDeduction) {

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
}

// Test that we can handle errors with custom base classes.
TEST(Error, HandleCustomErrorWithCustomBaseClass) {
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
TEST(Error, FirstHandlerOnly) {
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
TEST(Error, HandlerShadowing) {
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

  EXPECT_TRUE(CaughtErrorInfo == 42 && DummyInfo == 0 && DummyExtraInfo == 0)
      << "General Error handler did not shadow specific handler";
}

// Test joinErrors.
TEST(Error, CheckJoinErrors) {
  int CustomErrorInfo1 = 0;
  int CustomErrorInfo2 = 0;
  int CustomErrorExtraInfo = 0;
  Error E =
      joinErrors(make_error<CustomError>(7), make_error<CustomSubError>(42, 7));

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

// Test that we can consume success values.
TEST(Error, ConsumeSuccess) {
  Error E;
  consumeError(std::move(E));
}

TEST(Error, ConsumeError) {
  Error E = make_error<CustomError>(7);
  consumeError(std::move(E));
}

// Test that handleAllUnhandledErrors crashes if an error is not caught.
// Test runs in debug mode only.
#ifndef NDEBUG
TEST(Error, FailureToHandle) {
  auto FailToHandle = []() {
    handleAllErrors(make_error<CustomError>(7), [&](const CustomSubError &SE) {
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
TEST(Error, FailureFromHandler) {
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
TEST(Error, CatchErrorFromHandler) {
  int ErrorInfo = 0;

  Error E = handleErrors(
      make_error<CustomError>(7),
      [&](std::unique_ptr<CustomError> CE) { return Error(std::move(CE)); });

  handleAllErrors(std::move(E),
                  [&](const CustomError &CE) { ErrorInfo = CE.getInfo(); });

  EXPECT_EQ(ErrorInfo, 7)
      << "Failed to handle Error returned from handleErrors.";
}

// Test that the ExitOnError utility works as expected.
TEST(Error, ExitOnError) {
  ExitOnError ExitOnErr;
  ExitOnErr.setBanner("Error in tool:");
  ExitOnErr.setExitCodeMapper([](const Error &E) {
    if (E.isA<CustomSubError>())
      return 2;
    return 1;
  });

  // Make sure we don't bail on success.
  ExitOnErr(Error::success());
  EXPECT_EQ(ExitOnErr(Expected<int>(7)), 7)
      << "exitOnError returned an invalid value for Expected";

  // Exit tests.
  EXPECT_EXIT(ExitOnErr(make_error<CustomError>(7)),
              ::testing::ExitedWithCode(1), "Error in tool:")
      << "exitOnError returned an unexpected error result";

  EXPECT_EXIT(ExitOnErr(Expected<int>(make_error<CustomSubError>(0, 0))),
              ::testing::ExitedWithCode(2), "Error in tool:")
      << "exitOnError returned an unexpected error result";
}

// Test Checked Expected<T> in success mode.
TEST(Error, CheckedExpectedInSuccessMode) {
  Expected<int> A = 7;
  EXPECT_TRUE(!!A) << "Expected with non-error value doesn't convert to 'true'";
  // Access is safe in second test, since we checked the error in the first.
  EXPECT_EQ(*A, 7) << "Incorrect Expected non-error value";
}

// Test Unchecked Expected<T> in success mode.
// We expect this to blow up the same way Error would.
// Test runs in debug mode only.
#ifndef NDEBUG
TEST(Error, UncheckedExpectedInSuccessModeDestruction) {
  EXPECT_DEATH({ Expected<int> A = 7; },
               "Expected<T> must be checked before access or destruction.")
    << "Unchecekd Expected<T> success value did not cause an abort().";
}
#endif

// Test Unchecked Expected<T> in success mode.
// We expect this to blow up the same way Error would.
// Test runs in debug mode only.
#ifndef NDEBUG
TEST(Error, UncheckedExpectedInSuccessModeAccess) {
  EXPECT_DEATH({ Expected<int> A = 7; *A; },
               "Expected<T> must be checked before access or destruction.")
    << "Unchecekd Expected<T> success value did not cause an abort().";
}
#endif

// Test Unchecked Expected<T> in success mode.
// We expect this to blow up the same way Error would.
// Test runs in debug mode only.
#ifndef NDEBUG
TEST(Error, UncheckedExpectedInSuccessModeAssignment) {
  EXPECT_DEATH({ Expected<int> A = 7; A = 7; },
               "Expected<T> must be checked before access or destruction.")
    << "Unchecekd Expected<T> success value did not cause an abort().";
}
#endif

// Test Expected<T> in failure mode.
TEST(Error, ExpectedInFailureMode) {
  Expected<int> A = make_error<CustomError>(42);
  EXPECT_FALSE(!!A) << "Expected with error value doesn't convert to 'false'";
  Error E = A.takeError();
  EXPECT_TRUE(E.isA<CustomError>()) << "Incorrect Expected error value";
  consumeError(std::move(E));
}

// Check that an Expected instance with an error value doesn't allow access to
// operator*.
// Test runs in debug mode only.
#ifndef NDEBUG
TEST(Error, AccessExpectedInFailureMode) {
  Expected<int> A = make_error<CustomError>(42);
  EXPECT_DEATH(*A, "Expected<T> must be checked before access or destruction.")
      << "Incorrect Expected error value";
  consumeError(A.takeError());
}
#endif

// Check that an Expected instance with an error triggers an abort if
// unhandled.
// Test runs in debug mode only.
#ifndef NDEBUG
TEST(Error, UnhandledExpectedInFailureMode) {
  EXPECT_DEATH({ Expected<int> A = make_error<CustomError>(42); },
               "Expected<T> must be checked before access or destruction.")
      << "Unchecked Expected<T> failure value did not cause an abort()";
}
#endif

// Test covariance of Expected.
TEST(Error, ExpectedCovariance) {
  class B {};
  class D : public B {};

  Expected<B *> A1(Expected<D *>(nullptr));
  // Check A1 by converting to bool before assigning to it.
  (void)!!A1;
  A1 = Expected<D *>(nullptr);
  // Check A1 again before destruction.
  (void)!!A1;

  Expected<std::unique_ptr<B>> A2(Expected<std::unique_ptr<D>>(nullptr));
  // Check A2 by converting to bool before assigning to it.
  (void)!!A2;
  A2 = Expected<std::unique_ptr<D>>(nullptr);
  // Check A2 again before destruction.
  (void)!!A2;
}

TEST(Error, ErrorCodeConversions) {
  // Round-trip a success value to check that it converts correctly.
  EXPECT_EQ(errorToErrorCode(errorCodeToError(std::error_code())),
            std::error_code())
      << "std::error_code() should round-trip via Error conversions";

  // Round-trip an error value to check that it converts correctly.
  EXPECT_EQ(errorToErrorCode(errorCodeToError(errc::invalid_argument)),
            errc::invalid_argument)
      << "std::error_code error value should round-trip via Error "
         "conversions";

  // Round-trip a success value through ErrorOr/Expected to check that it
  // converts correctly.
  {
    auto Orig = ErrorOr<int>(42);
    auto RoundTripped =
      expectedToErrorOr(errorOrToExpected(ErrorOr<int>(42)));
    EXPECT_EQ(*Orig, *RoundTripped)
      << "ErrorOr<T> success value should round-trip via Expected<T> "
         "conversions.";
  }

  // Round-trip a failure value through ErrorOr/Expected to check that it
  // converts correctly.
  {
    auto Orig = ErrorOr<int>(errc::invalid_argument);
    auto RoundTripped =
      expectedToErrorOr(
          errorOrToExpected(ErrorOr<int>(errc::invalid_argument)));
    EXPECT_EQ(Orig.getError(), RoundTripped.getError())
      << "ErrorOr<T> failure value should round-trip via Expected<T> "
         "conversions.";
  }
}

} // end anon namespace
