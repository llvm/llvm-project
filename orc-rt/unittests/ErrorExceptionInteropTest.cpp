//===- ErrorExceptionInterorTest.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test interoperability between errors and exceptions.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-c/config.h"
#include "orc-rt/Error.h"
#include "gtest/gtest.h"

#include <system_error>

using namespace orc_rt;

namespace {

class CustomError : public ErrorExtends<CustomError, ErrorInfoBase> {
public:
  std::string toString() const noexcept override { return "CustomError"; }
};

} // anonymous namespace

#if ORC_RT_ENABLE_EXCEPTIONS
#define EXCEPTION_TEST(X)                                                      \
  do {                                                                         \
    X;                                                                         \
  } while (false)
#else
#define EXCEPTION_TEST(X) GTEST_SKIP() << "Exceptions disabled"
#endif

TEST(ErrorExceptionInterorTest, SuccessDoesntThrow) {
  // Test that Error::success values don't throw when throwOnFailure is called.
  EXCEPTION_TEST({
    try {
      auto E = Error::success();
      E.throwOnFailure();
    } catch (...) {
      ADD_FAILURE() << "Unexpected throw";
    }
  });
}

TEST(ErrorExceptionInteropTest, VoidReturnSuccess) {
  // Test that runCapturingExceptions returns Error::success for void()
  // function objects.
  EXCEPTION_TEST({
    bool Executed = false;
    auto Result = runCapturingExceptions([&]() { Executed = true; });
    static_assert(std::is_same_v<decltype(Result), Error>,
                  "Expected return type to be Error");
    EXPECT_FALSE(Result); // Error::success() evaluates to false
    EXPECT_TRUE(Executed);
  });
}

TEST(ErrorExceptionInteropTest, ErrorReturnPreserved) {
  // Test that plain Errors returned from runCapturingExceptions are returned
  // as expected.
  EXCEPTION_TEST({
    auto Result = runCapturingExceptions(
        []() -> Error { return make_error<StringError>("original error"); });
    EXPECT_TRUE(!!Result);
    EXPECT_EQ(toString(std::move(Result)), "original error");
  });
}

TEST(ErrorExceptionInteropTest, NonFallibleTReturnWrapped) {
  // Test that; for function types returning a non-Error, non-Expected type T;
  // runCapturingExceptions returns an Expected<T>.
  EXCEPTION_TEST({
    auto Result = runCapturingExceptions([]() { return 42; });
    static_assert(std::is_same_v<decltype(Result), Expected<int>>,
                  "Expected return type to be Expected<int>");
    EXPECT_TRUE(!!Result);
    EXPECT_EQ(*Result, 42);
  });
}

TEST(ErrorExceptionInteropTest, ExpectedReturnPreserved) {
  // Test that Expected success values are returned as expected.
  EXCEPTION_TEST({
    auto Result = runCapturingExceptions([]() -> Expected<int> { return 42; });
    EXPECT_TRUE(!!Result);
    EXPECT_EQ(*Result, 42);
  });
}

TEST(ErrorExceptionInteropTest, CatchThrownInt) {
  // Check that we can capture a thrown exception into an Error and recognize it
  // as a thrown exception.
  EXCEPTION_TEST({
    auto E = runCapturingExceptions([]() { throw 42; });
    EXPECT_TRUE(!!E);
    EXPECT_TRUE(E.isA<ExceptionError>());
    consumeError(std::move(E));
  });
}

TEST(ErrorExceptionInteropTest, RoundTripExceptionThroughError) {
  // Check that we can rethrow an exception that has been captured into an
  // error without affecting the dynamic type or value (e.g. we don't actually
  // rethrow the wrong type / value).
  EXCEPTION_TEST({
    int Result = 0;
    try {
      auto E = runCapturingExceptions([]() { throw 42; });
      EXPECT_TRUE(!!E);
      E.throwOnFailure();
    } catch (int N) {
      Result = N;
    } catch (...) {
      ADD_FAILURE() << "Caught unexpected error type";
    }
    EXPECT_EQ(Result, 42);
  });
}

static std::string peekAtErrorMessage(Error &Err) {
  std::string Msg;
  Err = handleErrors(std::move(Err), [&](std::unique_ptr<ErrorInfoBase> EIB) {
    Msg = EIB->toString();
    return make_error(std::move(EIB));
  });
  return Msg;
}

TEST(ErrorExceptionInteropTest, RoundTripErrorThroughException) {
  // Test Error → Exception → Error preserves type and message
  EXCEPTION_TEST({
    auto OriginalErr = make_error<StringError>("hello, error!");
    std::string OriginalMsg = peekAtErrorMessage(OriginalErr);

    Error RecoveredErr = Error::success();
    try {
      OriginalErr.throwOnFailure();
    } catch (ErrorInfoBase &EIB) {
      ErrorAsOutParameter _(RecoveredErr);
      RecoveredErr = restore_error(std::move(EIB));
    } catch (...) {
      ADD_FAILURE() << "Caught unexpected error type";
    }

    EXPECT_TRUE(RecoveredErr.isA<StringError>());
    EXPECT_EQ(toString(std::move(RecoveredErr)), OriginalMsg);
  });
}

TEST(ErrorExceptionInteropTest, ThrowErrorAndCatchAsException) {
  // Check that we can create an Error value, throw it as an exception, and
  // match its dynamic type to a catch handler.
  EXCEPTION_TEST({
    bool HandlerRan = false;
    std::string Msg;
    try {
      auto E = make_error<CustomError>();
      E.throwOnFailure();
    } catch (CustomError &E) {
      HandlerRan = true;
    } catch (ErrorInfoBase &E) {
      ADD_FAILURE() << "Failed to downcase error to dynamic type";
    } catch (...) {
      ADD_FAILURE() << "Caught unexpected error type";
    }
  });
}

TEST(ErrorExceptionInteropTest, ErrorExceptionToString) {
  /// Check that exceptions can be converted to Strings as exepcted.
  EXCEPTION_TEST({

    {
      // std::exception should be converted by calling `.what()`;
      class MyException : public std::exception {
      public:
        ~MyException() override {}
        const char *what() const noexcept override { return "what"; }
      };

      EXPECT_EQ(toString(runCapturingExceptions([]() { throw MyException(); })),
                "what");
    }

    {
      // std::error_code should be converted by calling `.message()`.
      auto EC = std::make_error_code(std::errc::cross_device_link);
      std::string ECErrMsg = EC.message();
      EXPECT_EQ(toString(runCapturingExceptions([&]() { throw EC; })),
                ECErrMsg);
    }

    {
      // std::string should be converted by copying its value.
      std::string ErrMsg = "foo";
      EXPECT_EQ(toString(runCapturingExceptions([&]() { throw ErrMsg; })),
                ErrMsg);
    }

    {
      // Check that exceptions of other types produce the expected
      // "unrecognized type" error message:
      EXPECT_EQ(toString(runCapturingExceptions([]() { throw 42; })),
                "C++ exception of unknown type");
    }
  });
}
