//===- ErrorCAPITest.cpp - Tests for Error C API --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests the C API for ORC runtime errors defined in orc-rt-c/Error.h.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-c/Error.h"
#include "orc-rt/Error.h"
#include "gtest/gtest.h"

#include <cstring>

using namespace orc_rt;

namespace {

// Test that wrapping a success value produces null.
TEST(ErrorCAPITest, WrapSuccess) {
  orc_rt_ErrorRef ErrRef = wrap(Error::success());
  EXPECT_EQ(ErrRef, orc_rt_ErrorSuccess);
}

// Test that wrap/unwrap round-trips correctly for error values.
TEST(ErrorCAPITest, WrapUnwrapRoundTrip) {
  Error Original = make_error<StringError>("test error");
  orc_rt_ErrorRef ErrRef = wrap(std::move(Original));

  EXPECT_NE(ErrRef, orc_rt_ErrorSuccess);

  Error Restored = unwrap(ErrRef);
  EXPECT_TRUE(Restored.isA<StringError>());
  EXPECT_EQ(toString(std::move(Restored)), "test error");
}

// Test that unwrapping null produces a success value.
TEST(ErrorCAPITest, UnwrapSuccess) {
  Error E = unwrap(orc_rt_ErrorSuccess);
  EXPECT_FALSE(E) << "Unwrapping null should produce success";
}

// Test orc_rt_Error_getTypeId returns the correct type ID.
TEST(ErrorCAPITest, GetTypeId) {
  orc_rt_ErrorRef ErrRef = orc_rt_StringError_create("test");
  orc_rt_Error_TypeId TypeId = orc_rt_Error_getTypeId(ErrRef);

  EXPECT_EQ(TypeId, orc_rt_StringError_getTypeId());

  orc_rt_Error_consume(ErrRef);
}

// Test orc_rt_Error_consume properly disposes of an error.
TEST(ErrorCAPITest, Consume) {
  orc_rt_ErrorRef ErrRef = orc_rt_StringError_create("test");
  EXPECT_NE(ErrRef, orc_rt_ErrorSuccess);

  // Should not crash or leak.
  orc_rt_Error_consume(ErrRef);
}

// Test orc_rt_Error_cantFail with success value.
TEST(ErrorCAPITest, CantFailSuccess) {
  // Should not crash.
  orc_rt_Error_cantFail(orc_rt_ErrorSuccess);
}

// Test orc_rt_Error_cantFail aborts on failure value.
TEST(ErrorCAPITest, CantFailFailure) {
  EXPECT_DEATH(
      { orc_rt_Error_cantFail(orc_rt_StringError_create("test")); }, "")
      << "orc_rt_Error_cantFail did not abort on failure value";
}

// Test orc_rt_Error_toString returns the error message and consumes the error.
TEST(ErrorCAPITest, ToString) {
  orc_rt_ErrorRef ErrRef = orc_rt_StringError_create("hello world");
  char *Msg = orc_rt_Error_toString(ErrRef);

  EXPECT_STREQ(Msg, "hello world");

  orc_rt_Error_freeErrorMessage(Msg);
}

// Test orc_rt_StringError_create creates an error with the correct message.
TEST(ErrorCAPITest, StringErrorCreate) {
  const char *TestMsg = "custom error message";
  orc_rt_ErrorRef ErrRef = orc_rt_StringError_create(TestMsg);

  EXPECT_NE(ErrRef, orc_rt_ErrorSuccess);

  // Verify it's a StringError.
  EXPECT_EQ(orc_rt_Error_getTypeId(ErrRef), orc_rt_StringError_getTypeId());

  // Verify the message.
  char *Msg = orc_rt_Error_toString(ErrRef);
  EXPECT_STREQ(Msg, TestMsg);
  orc_rt_Error_freeErrorMessage(Msg);
}

// Test orc_rt_StringError_getTypeId returns a consistent value.
TEST(ErrorCAPITest, StringErrorTypeIdConsistent) {
  orc_rt_Error_TypeId TypeId1 = orc_rt_StringError_getTypeId();
  orc_rt_Error_TypeId TypeId2 = orc_rt_StringError_getTypeId();

  EXPECT_EQ(TypeId1, TypeId2);
  EXPECT_NE(TypeId1, nullptr);
}

// Test that C API type ID matches C++ StringError class ID.
TEST(ErrorCAPITest, StringErrorTypeIdMatchesCpp) {
  orc_rt_Error_TypeId CTypeId = orc_rt_StringError_getTypeId();
  const void *CppTypeId = StringError::classID();

  EXPECT_EQ(CTypeId, CppTypeId);
}

// Test creating and consuming multiple errors.
TEST(ErrorCAPITest, MultipleErrors) {
  orc_rt_ErrorRef Err1 = orc_rt_StringError_create("error 1");
  orc_rt_ErrorRef Err2 = orc_rt_StringError_create("error 2");
  orc_rt_ErrorRef Err3 = orc_rt_StringError_create("error 3");

  EXPECT_NE(Err1, orc_rt_ErrorSuccess);
  EXPECT_NE(Err2, orc_rt_ErrorSuccess);
  EXPECT_NE(Err3, orc_rt_ErrorSuccess);

  char *Msg1 = orc_rt_Error_toString(Err1);
  char *Msg2 = orc_rt_Error_toString(Err2);
  char *Msg3 = orc_rt_Error_toString(Err3);

  EXPECT_STREQ(Msg1, "error 1");
  EXPECT_STREQ(Msg2, "error 2");
  EXPECT_STREQ(Msg3, "error 3");

  orc_rt_Error_freeErrorMessage(Msg1);
  orc_rt_Error_freeErrorMessage(Msg2);
  orc_rt_Error_freeErrorMessage(Msg3);
}

// Test wrapping a custom C++ error type and checking its type via C API.
class CustomCAPITestError
    : public ErrorExtends<CustomCAPITestError, ErrorInfoBase> {
public:
  CustomCAPITestError(int Code) : Code(Code) {}
  std::string toString() const noexcept override {
    return "CustomCAPITestError: " + std::to_string(Code);
  }
  int getCode() const { return Code; }

private:
  int Code;
};

TEST(ErrorCAPITest, CustomErrorTypeId) {
  Error CppError = make_error<CustomCAPITestError>(42);
  orc_rt_ErrorRef ErrRef = wrap(std::move(CppError));

  orc_rt_Error_TypeId TypeId = orc_rt_Error_getTypeId(ErrRef);

  // Should not be a StringError.
  EXPECT_NE(TypeId, orc_rt_StringError_getTypeId());

  // Should match the C++ class ID.
  EXPECT_EQ(TypeId, CustomCAPITestError::classID());

  char *Msg = orc_rt_Error_toString(ErrRef);
  EXPECT_STREQ(Msg, "CustomCAPITestError: 42");
  orc_rt_Error_freeErrorMessage(Msg);
}

} // end anonymous namespace
