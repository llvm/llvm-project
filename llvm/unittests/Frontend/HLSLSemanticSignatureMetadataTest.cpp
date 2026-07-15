//===- HLSLSemanticSignatureMetadataTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/SemanticSignatures.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::hlsl;

namespace {

class HLSLSemanticSignatureMetadataTest : public testing::Test {
protected:
  LLVMContext Ctx;

  Metadata *getI32(uint32_t Val) {
    return ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(Ctx), Val));
  }

  Metadata *getI8(uint8_t Val) {
    return ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt8Ty(Ctx), Val));
  }

  Metadata *getStr(StringRef Val) { return MDString::get(Ctx, Val); }

  MDNode *getIndices(ArrayRef<uint32_t> Indices) {
    SmallVector<Metadata *> Ops;
    for (uint32_t I : Indices)
      Ops.push_back(getI32(I));
    return MDNode::get(Ctx, Ops);
  }
};

TEST_F(HLSLSemanticSignatureMetadataTest, StructHelpers) {
  SemanticSignatureElement Elem;
  EXPECT_FALSE(Elem.isAllocated());

  Elem.Cols = 4;
  Elem.StartRow = 0;
  Elem.StartCol = 0;
  EXPECT_TRUE(Elem.isAllocated());
  EXPECT_EQ(Elem.getDeclaredMask(), 0xF);
}

} // namespace
