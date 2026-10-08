//===- TestTargetTest.cpp - Unit tests for the test ABI target -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/ABI/Targets/Test/TestTarget.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::abi;
using namespace mlir::abi::test;

namespace {

class TestTargetClassifyTest : public ::testing::Test {
protected:
  TestTargetClassifyTest()
      : module(ModuleOp::create(UnknownLoc::get(&context))), dl(*module) {
    context.loadDialect<DLTIDialect>();
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  DataLayout dl;
};

TEST_F(TestTargetClassifyTest, IgnoresNoneType) {
  auto noneTy = NoneType::get(&context);
  FunctionClassification fc = classify({}, noneTy, dl);
  EXPECT_EQ(fc.returnInfo.kind, ArgKind::Ignore);
}

TEST_F(TestTargetClassifyTest, ExtendsNarrowSignedInteger) {
  auto i8 = IntegerType::get(&context, 8, IntegerType::Signed);
  FunctionClassification fc = classify({i8}, NoneType::get(&context), dl);
  ASSERT_EQ(fc.argInfos.size(), 1u);
  EXPECT_EQ(fc.argInfos[0].kind, ArgKind::Extend);
  EXPECT_TRUE(fc.argInfos[0].signExtend);
  auto coerced = dyn_cast<IntegerType>(fc.argInfos[0].coercedType);
  ASSERT_TRUE(coerced);
  EXPECT_EQ(coerced.getWidth(), 32u);
}

TEST_F(TestTargetClassifyTest, ExtendsNarrowSignlessIntegerAsZeroExt) {
  auto i8 = IntegerType::get(&context, 8);
  FunctionClassification fc = classify({i8}, NoneType::get(&context), dl);
  ASSERT_EQ(fc.argInfos.size(), 1u);
  EXPECT_EQ(fc.argInfos[0].kind, ArgKind::Extend);
  EXPECT_FALSE(fc.argInfos[0].signExtend);
}

TEST_F(TestTargetClassifyTest, RegisterSizedIntegerIsDirect) {
  auto i32 = IntegerType::get(&context, 32);
  auto i64 = IntegerType::get(&context, 64);
  FunctionClassification fc = classify({i32, i64}, NoneType::get(&context), dl);
  ASSERT_EQ(fc.argInfos.size(), 2u);
  EXPECT_EQ(fc.argInfos[0].kind, ArgKind::Direct);
  EXPECT_EQ(fc.argInfos[1].kind, ArgKind::Direct);
}

TEST_F(TestTargetClassifyTest, IndexTypeIsDirect) {
  // The default DataLayout reports IndexType as 64 bits, which is at or above
  // the extension threshold and should classify as Direct.
  auto idx = IndexType::get(&context);
  FunctionClassification fc = classify({idx}, idx, dl);
  EXPECT_EQ(fc.returnInfo.kind, ArgKind::Direct);
  ASSERT_EQ(fc.argInfos.size(), 1u);
  EXPECT_EQ(fc.argInfos[0].kind, ArgKind::Direct);
}

TEST_F(TestTargetClassifyTest, FloatIsDirect) {
  auto f32 = Float32Type::get(&context);
  FunctionClassification fc = classify({f32}, f32, dl);
  EXPECT_EQ(fc.returnInfo.kind, ArgKind::Direct);
  EXPECT_EQ(fc.argInfos[0].kind, ArgKind::Direct);
}

TEST_F(TestTargetClassifyTest, FunctionLevelReturnAndArgsClassifiedTogether) {
  auto i32 = IntegerType::get(&context, 32);
  auto f64 = Float64Type::get(&context);
  FunctionClassification fc = classify({i32, f64}, i32, dl);
  EXPECT_EQ(fc.returnInfo.kind, ArgKind::Direct);
  ASSERT_EQ(fc.argInfos.size(), 2u);
  EXPECT_EQ(fc.argInfos[0].kind, ArgKind::Direct);
  EXPECT_EQ(fc.argInfos[1].kind, ArgKind::Direct);
}

TEST_F(TestTargetClassifyTest,
       TypeWithoutDataLayoutInterfaceClassifiedAsIgnore) {
  // FunctionType does not implement DataLayoutTypeInterface.  The classifier
  // must treat it as Ignore rather than crashing in dl.getTypeSizeInBits().
  // This guards against the same crash for dialect-specific void / sentinel
  // types (e.g. cir::VoidType) used as a function's "no return value" marker.
  auto i32 = IntegerType::get(&context, 32);
  auto fnTy = FunctionType::get(&context, {i32}, {i32});
  FunctionClassification fc = classify({}, fnTy, dl);
  EXPECT_EQ(fc.returnInfo.kind, ArgKind::Ignore);
}

class TestTargetParseTest : public ::testing::Test {
protected:
  TestTargetParseTest() : builder(&context) {
    // Suppress diagnostic printing during tests; capture into lastError
    // for assertions instead.
    context.getDiagEngine().registerHandler([this](Diagnostic &diag) {
      lastError = diag.str();
      return success();
    });
  }

  /// Convenience: parse and assert success, returning the result.
  FunctionClassification parseOk(DictionaryAttr attr) {
    auto loc = UnknownLoc::get(&context);
    auto result =
        parseClassificationAttr(attr, [&]() { return mlir::emitError(loc); });
    EXPECT_TRUE(result.has_value())
        << "parseClassificationAttr failed: " << lastError;
    return result.value_or(FunctionClassification{});
  }

  /// Convenience: parse and assert failure with a substring match.
  void parseError(DictionaryAttr attr, StringRef expectedSubstr) {
    auto loc = UnknownLoc::get(&context);
    lastError.clear();
    auto result =
        parseClassificationAttr(attr, [&]() { return mlir::emitError(loc); });
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(lastError.find(expectedSubstr.str()), std::string::npos)
        << "expected error containing '" << expectedSubstr << "' but got '"
        << lastError << "'";
  }

  DictionaryAttr makeArg(ArrayRef<NamedAttribute> entries) {
    return DictionaryAttr::get(&context, entries);
  }

  MLIRContext context;
  OpBuilder builder;
  std::string lastError;
};

TEST_F(TestTargetParseTest, ParsesDirectReturnAndOneDirectArg) {
  auto direct =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("direct"))});
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("return", direct),
      builder.getNamedAttr("args", builder.getArrayAttr({direct})),
  });

  auto fc = parseOk(attr);
  EXPECT_EQ(fc.returnInfo.kind, ArgKind::Direct);
  ASSERT_EQ(fc.argInfos.size(), 1u);
  EXPECT_EQ(fc.argInfos[0].kind, ArgKind::Direct);
}

TEST_F(TestTargetParseTest, ParsesExtendWithCoercedTypeAndSignExtend) {
  auto i32 = IntegerType::get(&context, 32);
  auto extend = makeArg({
      builder.getNamedAttr("kind", builder.getStringAttr("extend")),
      builder.getNamedAttr("coerced_type", TypeAttr::get(i32)),
      builder.getNamedAttr("sign_extend", builder.getBoolAttr(true)),
  });
  auto direct =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("direct"))});
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("return", direct),
      builder.getNamedAttr("args", builder.getArrayAttr({extend})),
  });

  auto fc = parseOk(attr);
  ASSERT_EQ(fc.argInfos.size(), 1u);
  EXPECT_EQ(fc.argInfos[0].kind, ArgKind::Extend);
  EXPECT_TRUE(fc.argInfos[0].signExtend);
  EXPECT_EQ(fc.argInfos[0].coercedType, i32);
}

TEST_F(TestTargetParseTest, ParsesIndirectWithAlignAndByval) {
  auto direct =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("direct"))});
  auto indirect = makeArg({
      builder.getNamedAttr("kind", builder.getStringAttr("indirect")),
      builder.getNamedAttr("indirect_align", builder.getI64IntegerAttr(16)),
      builder.getNamedAttr("byval", builder.getBoolAttr(false)),
  });
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("return", direct),
      builder.getNamedAttr("args", builder.getArrayAttr({indirect})),
  });

  auto fc = parseOk(attr);
  ASSERT_EQ(fc.argInfos.size(), 1u);
  EXPECT_EQ(fc.argInfos[0].kind, ArgKind::Indirect);
  EXPECT_EQ(fc.argInfos[0].indirectAlign, llvm::Align(16));
  EXPECT_FALSE(fc.argInfos[0].byVal);
}

TEST_F(TestTargetParseTest, ParsesIgnoreAndExpand) {
  auto ignore =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("ignore"))});
  auto expand =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("expand"))});
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("return", ignore),
      builder.getNamedAttr("args", builder.getArrayAttr({expand, ignore})),
  });

  auto fc = parseOk(attr);
  EXPECT_EQ(fc.returnInfo.kind, ArgKind::Ignore);
  ASSERT_EQ(fc.argInfos.size(), 2u);
  EXPECT_EQ(fc.argInfos[0].kind, ArgKind::Expand);
  EXPECT_EQ(fc.argInfos[1].kind, ArgKind::Ignore);
}

TEST_F(TestTargetParseTest, RejectsMissingReturn) {
  auto direct =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("direct"))});
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("args", builder.getArrayAttr({direct})),
  });
  parseError(attr, "missing required 'return'");
}

TEST_F(TestTargetParseTest, RejectsMissingArgs) {
  auto direct =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("direct"))});
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("return", direct),
  });
  parseError(attr, "missing required 'args'");
}

TEST_F(TestTargetParseTest, RejectsUnknownTopLevelKey) {
  auto direct =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("direct"))});
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("return", direct),
      builder.getNamedAttr("args", builder.getArrayAttr({})),
      builder.getNamedAttr("future_field", builder.getStringAttr("hello")),
  });
  parseError(attr, "unknown top-level key 'future_field'");
}

TEST_F(TestTargetParseTest, RejectsUnknownArgKey) {
  auto badArg = makeArg({
      builder.getNamedAttr("kind", builder.getStringAttr("direct")),
      builder.getNamedAttr("future_field", builder.getBoolAttr(true)),
  });
  auto direct =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("direct"))});
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("return", direct),
      builder.getNamedAttr("args", builder.getArrayAttr({badArg})),
  });
  parseError(attr, "unknown key 'future_field'");
}

TEST_F(TestTargetParseTest, RejectsExtendWithoutCoercedType) {
  auto badExtend =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("extend"))});
  auto direct =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("direct"))});
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("return", direct),
      builder.getNamedAttr("args", builder.getArrayAttr({badExtend})),
  });
  parseError(attr, "kind='extend' requires 'coerced_type'");
}

TEST_F(TestTargetParseTest, RejectsIndirectWithoutAlign) {
  auto badIndirect = makeArg(
      {builder.getNamedAttr("kind", builder.getStringAttr("indirect"))});
  auto direct =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("direct"))});
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("return", direct),
      builder.getNamedAttr("args", builder.getArrayAttr({badIndirect})),
  });
  parseError(attr, "kind='indirect' requires 'indirect_align'");
}

TEST_F(TestTargetParseTest, RejectsIndirectWithNonPowerOfTwoAlign) {
  auto badIndirect = makeArg({
      builder.getNamedAttr("kind", builder.getStringAttr("indirect")),
      builder.getNamedAttr("indirect_align", builder.getI64IntegerAttr(7)),
  });
  auto direct =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("direct"))});
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("return", direct),
      builder.getNamedAttr("args", builder.getArrayAttr({badIndirect})),
  });
  parseError(attr, "must be a positive power of 2");
}

TEST_F(TestTargetParseTest, RejectsUnknownKind) {
  auto bad = makeArg(
      {builder.getNamedAttr("kind", builder.getStringAttr("invalid_kind"))});
  auto direct =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("direct"))});
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("return", direct),
      builder.getNamedAttr("args", builder.getArrayAttr({bad})),
  });
  parseError(attr, "unknown kind='invalid_kind'");
}

TEST_F(TestTargetParseTest, RejectsMissingKind) {
  auto bad = makeArg({});
  auto direct =
      makeArg({builder.getNamedAttr("kind", builder.getStringAttr("direct"))});
  auto attr = builder.getDictionaryAttr({
      builder.getNamedAttr("return", direct),
      builder.getNamedAttr("args", builder.getArrayAttr({bad})),
  });
  parseError(attr, "missing required 'kind'");
}

} // namespace
