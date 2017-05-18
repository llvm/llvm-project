//===- llvm/unittest/DebugInfo/PDB/TypeServerHandlerTest.cpp --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ErrorChecking.h"

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeRecordMapping.h"
#include "llvm/DebugInfo/CodeView/TypeSerializer.h"
#include "llvm/DebugInfo/CodeView/TypeServerHandler.h"
#include "llvm/DebugInfo/CodeView/TypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

namespace {

constexpr uint8_t Guid[] = {0x2a, 0x2c, 0x1c, 0x2a, 0xcb, 0x9e, 0x48, 0x18,
                            0x82, 0x82, 0x7a, 0x87, 0xc3, 0xfe, 0x16, 0xe8};
StringRef GuidStr(reinterpret_cast<const char *>(Guid),
                  llvm::array_lengthof(Guid));

constexpr const char *Name = "Test Name";
constexpr int Age = 1;

class MockTypeServerHandler : public TypeServerHandler {
public:
  explicit MockTypeServerHandler(bool HandleAlways)
      : HandleAlways(HandleAlways) {}

  Expected<bool> handle(TypeServer2Record &TS,
                        TypeVisitorCallbacks &Callbacks) override {
    if (TS.Age != Age || TS.Guid != GuidStr || TS.Name != Name)
      return make_error<CodeViewError>(cv_error_code::corrupt_record,
                                       "Invalid TypeServer record!");

    if (Handled && !HandleAlways)
      return false;

    Handled = true;
    return true;
  }

  bool Handled = false;
  bool HandleAlways;
};

class MockTypeVisitorCallbacks : public TypeVisitorCallbacks {
public:
  enum class State {
    Ready,
    VisitTypeBegin,
    VisitKnownRecord,
    VisitTypeEnd,
  };
  Error visitTypeBegin(CVType &CVT) override {
    if (S != State::Ready)
      return make_error<CodeViewError>(cv_error_code::unspecified,
                                       "Invalid visitor state!");

    S = State::VisitTypeBegin;
    return Error::success();
  }

  Error visitKnownRecord(CVType &CVT, TypeServer2Record &TS) override {
    if (S != State::VisitTypeBegin)
      return make_error<CodeViewError>(cv_error_code::unspecified,
                                       "Invalid visitor state!");

    S = State::VisitKnownRecord;
    return Error::success();
  }

  Error visitTypeEnd(CVType &CVT) override {
    if (S != State::VisitKnownRecord)
      return make_error<CodeViewError>(cv_error_code::unspecified,
                                       "Invalid visitor state!");

    S = State::VisitTypeEnd;
    return Error::success();
  }

  State S = State::Ready;
};

class TypeServerHandlerTest : public testing::Test {
public:
  void SetUp() override {
    TypeServer2Record R(TypeRecordKind::TypeServer2);
    R.Age = Age;
    R.Guid = GuidStr;
    R.Name = Name;

    TypeTableBuilder Builder(Allocator);
    Builder.writeKnownType(R);
    TypeServerRecord.RecordData = Builder.records().front();
    TypeServerRecord.Type = TypeLeafKind::LF_TYPESERVER2;
  }

protected:
  BumpPtrAllocator Allocator;
  CVType TypeServerRecord;
};

// Test that when no type server handler is registered, it gets handled by the
// normal
// visitor callbacks.
TEST_F(TypeServerHandlerTest, VisitRecordNoTypeServer) {
  MockTypeVisitorCallbacks C2;
  MockTypeVisitorCallbacks C1;
  TypeVisitorCallbackPipeline Pipeline;

  Pipeline.addCallbackToPipeline(C1);
  Pipeline.addCallbackToPipeline(C2);

  EXPECT_NO_ERROR(codeview::visitTypeRecord(TypeServerRecord, Pipeline));

  EXPECT_EQ(MockTypeVisitorCallbacks::State::VisitTypeEnd, C1.S);
  EXPECT_EQ(MockTypeVisitorCallbacks::State::VisitTypeEnd, C2.S);
}

// Test that when a TypeServerHandler is registered, it gets consumed by the
// handler if and only if the handler returns true.
TEST_F(TypeServerHandlerTest, VisitRecordWithTypeServerOnce) {
  MockTypeServerHandler Handler(false);

  MockTypeVisitorCallbacks C1;

  // Our mock server returns true the first time.
  EXPECT_NO_ERROR(codeview::visitTypeRecord(
      TypeServerRecord, C1, codeview::VDS_BytesExternal, &Handler));
  EXPECT_TRUE(Handler.Handled);
  EXPECT_EQ(MockTypeVisitorCallbacks::State::Ready, C1.S);

  // And false the second time.
  EXPECT_NO_ERROR(codeview::visitTypeRecord(
      TypeServerRecord, C1, codeview::VDS_BytesExternal, &Handler));
  EXPECT_TRUE(Handler.Handled);
  EXPECT_EQ(MockTypeVisitorCallbacks::State::VisitTypeEnd, C1.S);
}

// Test that when a type server handler is registered, if the handler keeps
// returning true, it will keep getting consumed by the handler and not go
// to the default processor.
TEST_F(TypeServerHandlerTest, VisitRecordWithTypeServerAlways) {
  MockTypeServerHandler Handler(true);

  MockTypeVisitorCallbacks C1;

  EXPECT_NO_ERROR(codeview::visitTypeRecord(
      TypeServerRecord, C1, codeview::VDS_BytesExternal, &Handler));
  EXPECT_TRUE(Handler.Handled);
  EXPECT_EQ(MockTypeVisitorCallbacks::State::Ready, C1.S);

  EXPECT_NO_ERROR(codeview::visitTypeRecord(
      TypeServerRecord, C1, codeview::VDS_BytesExternal, &Handler));
  EXPECT_TRUE(Handler.Handled);
  EXPECT_EQ(MockTypeVisitorCallbacks::State::Ready, C1.S);
}

} // end anonymous namespace
