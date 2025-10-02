//===- ErrorTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime.
//
// Note:
//   This unit test was adapted from
//   llvm/unittests/Support/ErrorTest.cpp
//
//===----------------------------------------------------------------------===//

#include "orc-rt/Error.h"
#include "gtest/gtest.h"

using namespace orc_rt;

namespace {

class CustomError : public RTTIExtends<CustomError, ErrorInfoBase> {
public:
  CustomError(int Info) : Info(Info) {}
  std::string toString() const override {
    return "CustomError (" + std::to_string(Info) + ")";
  }
  int getInfo() const { return Info; }

protected:
  int Info;
};

class CustomSubError : public RTTIExtends<CustomSubError, CustomError> {
public:
  CustomSubError(int Info, std::string ExtraInfo)
      : RTTIExtends<CustomSubError, CustomError>(Info),
        ExtraInfo(std::move(ExtraInfo)) {}

  std::string toString() const override {
    return "CustomSubError (" + std::to_string(Info) + ", " + ExtraInfo + ")";
  }
  const std::string &getExtraInfo() const { return ExtraInfo; }

protected:
  std::string ExtraInfo;
};

static Error handleCustomError(const CustomError &CE) {
  return Error::success();
}

static void handleCustomErrorVoid(const CustomError &CE) {}

static Error handleCustomErrorUP(std::unique_ptr<CustomError> CE) {
  return Error::success();
}

static void handleCustomErrorUPVoid(std::unique_ptr<CustomError> CE) {}

} // end anonymous namespace

// Test that a checked success value doesn't cause any issues.
TEST(ErrorTest, CheckedSuccess) {
  Error E = Error::success();
  EXPECT_FALSE(E) << "Unexpected error while testing Error 'Success'";
}

// Check that a consumed success value doesn't cause any issues.
TEST(ErrorTest, ConsumeSuccess) { consumeError(Error::success()); }

TEST(ErrorTest, ConsumeError) {
  Error E = make_error<CustomError>(42);
  if (E) {
    consumeError(std::move(E));
  } else
    ADD_FAILURE() << "Error failure value should convert to true";
}

// Test that unchecked success values cause an abort.
TEST(ErrorTest, UncheckedSuccess) {
  EXPECT_DEATH(
      { Error E = Error::success(); },
      "Error must be checked prior to destruction")
      << "Unchecked Error Succes value did not cause abort()";
}

// Test that a checked but unhandled error causes an abort.
TEST(ErrorTest, CheckedButUnhandledError) {
  auto DropUnhandledError = []() {
    Error E = make_error<CustomError>(42);
    (void)!E;
  };
  EXPECT_DEATH(DropUnhandledError(),
               "Error must be checked prior to destruction")
      << "Unhandled Error failure value did not cause an abort()";
}

// Check that we can handle a custom error.
TEST(ErrorTest, HandleCustomError) {
  int CaughtErrorInfo = 0;
  handleAllErrors(make_error<CustomError>(42), [&](const CustomError &CE) {
    CaughtErrorInfo = CE.getInfo();
  });

  EXPECT_EQ(CaughtErrorInfo, 42) << "Wrong result from CustomError handler";
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
TEST(ErrorTest, HandlerTypeDeduction) {

  handleAllErrors(make_error<CustomError>(42), [](const CustomError &CE) {});

  handleAllErrors(
      make_error<CustomError>(42),
      [](const CustomError &CE) mutable -> Error { return Error::success(); });

  handleAllErrors(make_error<CustomError>(42),
                  [](const CustomError &CE) mutable {});

  handleAllErrors(make_error<CustomError>(42),
                  [](CustomError &CE) -> Error { return Error::success(); });

  handleAllErrors(make_error<CustomError>(42), [](CustomError &CE) {});

  handleAllErrors(
      make_error<CustomError>(42),
      [](CustomError &CE) mutable -> Error { return Error::success(); });

  handleAllErrors(make_error<CustomError>(42), [](CustomError &CE) mutable {});

  handleAllErrors(make_error<CustomError>(42),
                  [](std::unique_ptr<CustomError> CE) -> Error {
                    return Error::success();
                  });

  handleAllErrors(make_error<CustomError>(42),
                  [](std::unique_ptr<CustomError> CE) {});

  handleAllErrors(make_error<CustomError>(42),
                  [](std::unique_ptr<CustomError> CE) mutable -> Error {
                    return Error::success();
                  });

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
TEST(ErrorTest, HandleCustomErrorWithCustomBaseClass) {
  int CaughtErrorInfo = 0;
  std::string CaughtErrorExtraInfo;
  handleAllErrors(make_error<CustomSubError>(42, "foo"),
                  [&](const CustomSubError &SE) {
                    CaughtErrorInfo = SE.getInfo();
                    CaughtErrorExtraInfo = SE.getExtraInfo();
                  });

  EXPECT_EQ(CaughtErrorInfo, 42) << "Wrong result from CustomSubError handler";
  EXPECT_EQ(CaughtErrorExtraInfo, "foo")
      << "Wrong result from CustomSubError handler";
}

// Check that we trigger only the first handler that applies.
TEST(ErrorTest, FirstHandlerOnly) {
  int DummyInfo = 0;
  int CaughtErrorInfo = 0;
  std::string CaughtErrorExtraInfo;

  handleAllErrors(
      make_error<CustomSubError>(42, "foo"),
      [&](const CustomSubError &SE) {
        CaughtErrorInfo = SE.getInfo();
        CaughtErrorExtraInfo = SE.getExtraInfo();
      },
      [&](const CustomError &CE) { DummyInfo = CE.getInfo(); });

  EXPECT_EQ(CaughtErrorInfo, 42) << "Activated the wrong Error handler(s)";
  EXPECT_EQ(CaughtErrorExtraInfo, "foo")
      << "Activated the wrong Error handler(s)";
  EXPECT_EQ(DummyInfo, 0) << "Activated the wrong Error handler(s)";
}

// Check that general handlers shadow specific ones.
TEST(ErrorTest, HandlerShadowing) {
  int CaughtErrorInfo = 0;
  int DummyInfo = 0;
  std::string DummyExtraInfo;

  handleAllErrors(
      make_error<CustomSubError>(42, "foo"),
      [&](const CustomError &CE) { CaughtErrorInfo = CE.getInfo(); },
      [&](const CustomSubError &SE) {
        DummyInfo = SE.getInfo();
        DummyExtraInfo = SE.getExtraInfo();
      });

  EXPECT_EQ(CaughtErrorInfo, 42)
      << "General Error handler did not shadow specific handler";
  EXPECT_EQ(DummyInfo, 0)
      << "General Error handler did not shadow specific handler";
  EXPECT_EQ(DummyExtraInfo, "")
      << "General Error handler did not shadow specific handler";
}

// ErrorAsOutParameter tester.
static void errAsOutParamHelper(Error &Err) {
  ErrorAsOutParameter ErrAsOutParam(&Err);
  // Verify that checked flag is raised - assignment should not crash.
  Err = Error::success();
  // Raise the checked bit manually - caller should still have to test the
  // error.
  (void)!!Err;
}

// Test that ErrorAsOutParameter sets the checked flag on construction.
TEST(ErrorTest, ErrorAsOutParameterChecked) {
  Error E = Error::success();
  errAsOutParamHelper(E);
  (void)!!E;
}

// Test that ErrorAsOutParameter clears the checked flag on destruction.
TEST(ErrorTest, ErrorAsOutParameterUnchecked) {
  EXPECT_DEATH(
      {
        Error E = Error::success();
        errAsOutParamHelper(E);
      },
      "Error must be checked prior to destruction")
      << "ErrorAsOutParameter did not clear the checked flag on destruction.";
}

// Check 'Error::isA<T>' method handling.
TEST(ErrorTest, IsAHandling) {
  // Check 'isA' handling.
  Error E = make_error<CustomError>(42);
  Error F = make_error<CustomSubError>(42, "foo");
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

TEST(ErrorTest, StringError) {
  auto E = make_error<StringError>("foo");
  if (E.isA<StringError>())
    EXPECT_EQ(toString(std::move(E)), "foo") << "Unexpected StringError value";
  else
    ADD_FAILURE() << "Expected StringError value";
}

// Test Checked Expected<T> in success mode.
TEST(ErrorTest, CheckedExpectedInSuccessMode) {
  Expected<int> A = 7;
  EXPECT_TRUE(!!A) << "Expected with non-error value doesn't convert to 'true'";
  // Access is safe in second test, since we checked the error in the first.
  EXPECT_EQ(*A, 7) << "Incorrect Expected non-error value";
}

// Test Expected with reference type.
TEST(ErrorTest, ExpectedWithReferenceType) {
  int A = 7;
  Expected<int &> B = A;
  // 'Check' B.
  (void)!!B;
  int &C = *B;
  EXPECT_EQ(&A, &C) << "Expected failed to propagate reference";
}

// Test Unchecked Expected<T> in success mode.
// We expect this to blow up the same way Error would.
// Test runs in debug mode only.
TEST(ErrorTest, UncheckedExpectedInSuccessModeDestruction) {
  EXPECT_DEATH(
      { Expected<int> A = 7; },
      "Expected<T> must be checked before access or destruction.")
      << "Unchecekd Expected<T> success value did not cause an abort().";
}

// Test Unchecked Expected<T> in success mode.
// We expect this to blow up the same way Error would.
// Test runs in debug mode only.
TEST(ErrorTest, UncheckedExpectedInSuccessModeAccess) {
  EXPECT_DEATH(
      {
        Expected<int> A = 7;
        *A;
      },
      "Expected<T> must be checked before access or destruction.")
      << "Unchecekd Expected<T> success value did not cause an abort().";
}

// Test Unchecked Expected<T> in success mode.
// We expect this to blow up the same way Error would.
// Test runs in debug mode only.
TEST(ErrorTest, UncheckedExpectedInSuccessModeAssignment) {
  EXPECT_DEATH(
      {
        Expected<int> A = 7;
        A = 7;
      },
      "Expected<T> must be checked before access or destruction.")
      << "Unchecekd Expected<T> success value did not cause an abort().";
}

// Test Expected<T> in failure mode.
TEST(ErrorTest, ExpectedInFailureMode) {
  Expected<int> A = make_error<CustomError>(42);
  EXPECT_FALSE(!!A) << "Expected with error value doesn't convert to 'false'";
  Error E = A.takeError();
  EXPECT_TRUE(E.isA<CustomError>()) << "Incorrect Expected error value";
  consumeError(std::move(E));
}

// Check that an Expected instance with an error value doesn't allow access to
// operator*.
// Test runs in debug mode only.
TEST(ErrorTest, AccessExpectedInFailureMode) {
  Expected<int> A = make_error<CustomError>(42);
  EXPECT_DEATH(*A, "Expected<T> must be checked before access or destruction.")
      << "Incorrect Expected error value";
  consumeError(A.takeError());
}

// Check that an Expected instance with an error triggers an abort if
// unhandled.
// Test runs in debug mode only.
TEST(ErrorTest, UnhandledExpectedInFailureMode) {
  EXPECT_DEATH(
      { Expected<int> A = make_error<CustomError>(42); },
      "Expected<T> must be checked before access or destruction.")
      << "Unchecked Expected<T> failure value did not cause an abort()";
}

// Test covariance of Expected.
TEST(ErrorTest, ExpectedCovariance) {
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

// Test that Expected<Error> works as expected.
TEST(ErrorTest, ExpectedError) {
  {
    // Test success-success case.
    Expected<Error> E(Error::success(), ForceExpectedSuccessValue());
    EXPECT_TRUE(!!E);
    cantFail(E.takeError());
    auto Err = std::move(*E);
    EXPECT_FALSE(!!Err);
  }

  {
    // Test "failure" success case.
    Expected<Error> E(make_error<StringError>("foo"),
                      ForceExpectedSuccessValue());
    EXPECT_TRUE(!!E);
    cantFail(E.takeError());
    auto Err = std::move(*E);
    EXPECT_TRUE(!!Err);
    EXPECT_EQ(toString(std::move(Err)), "foo");
  }
}

// Test that Expected<Expected<T>> works as expected.
TEST(ErrorTest, ExpectedExpected) {
  {
    // Test success-success case.
    Expected<Expected<int>> E(Expected<int>(42), ForceExpectedSuccessValue());
    EXPECT_TRUE(!!E);
    cantFail(E.takeError());
    auto EI = std::move(*E);
    EXPECT_TRUE(!!EI);
    cantFail(EI.takeError());
    EXPECT_EQ(*EI, 42);
  }

  {
    // Test "failure" success case.
    Expected<Expected<int>> E(Expected<int>(make_error<StringError>("foo")),
                              ForceExpectedSuccessValue());
    EXPECT_TRUE(!!E);
    cantFail(E.takeError());
    auto EI = std::move(*E);
    EXPECT_FALSE(!!EI);
    EXPECT_EQ(toString(EI.takeError()), "foo");
  }
}

// Test that the ExitOnError utility works as expected.
TEST(ErrorTest, CantFailSuccess) {
  cantFail(Error::success());

  int X = cantFail(Expected<int>(42));
  EXPECT_EQ(X, 42) << "Expected value modified by cantFail";

  int Dummy = 42;
  int &Y = cantFail(Expected<int &>(Dummy));
  EXPECT_EQ(&Dummy, &Y) << "Reference mangled by cantFail";
}

// Test that cantFail results in a crash if you pass it a failure value.
TEST(ErrorTest, CantFailDeath) {
  EXPECT_DEATH(cantFail(make_error<StringError>("foo")), "")
      << "cantFail(Error) did not cause an abort for failure value";

  EXPECT_DEATH(cantFail(Expected<int>(make_error<StringError>("foo"))), "")
      << "cantFail(Expected<int>) did not cause an abort for failure value";
}
