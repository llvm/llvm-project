//===- TargetTransformInfoTest.cpp - TargetTransformInfo tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {
class FloorConversionTTI final : public TargetTransformInfoImplBase {
public:
  explicit FloorConversionTTI(const DataLayout &DL)
      : TargetTransformInfoImplBase(DL) {}

  bool canFoldFloorIntoFPToSI(Type *SrcTy, Type *DestTy) const override {
    return SrcTy->isDoubleTy() && DestTy->isIntegerTy(32);
  }
};

TEST(TargetTransformInfoTest, FloorConversionDefaultsToUnsupported) {
  LLVMContext Context;
  DataLayout DL("");
  TargetTransformInfo TTI(DL);
  EXPECT_FALSE(TTI.canFoldFloorIntoFPToSI(Type::getDoubleTy(Context),
                                          Type::getInt32Ty(Context)));
}

TEST(TargetTransformInfoTest, FloorConversionUsesTargetImplementation) {
  LLVMContext Context;
  DataLayout DL("");
  TargetTransformInfo TTI(std::make_unique<FloorConversionTTI>(DL));
  EXPECT_TRUE(TTI.canFoldFloorIntoFPToSI(Type::getDoubleTy(Context),
                                         Type::getInt32Ty(Context)));
  EXPECT_FALSE(TTI.canFoldFloorIntoFPToSI(Type::getFloatTy(Context),
                                          Type::getInt32Ty(Context)));
  EXPECT_FALSE(TTI.canFoldFloorIntoFPToSI(Type::getDoubleTy(Context),
                                          Type::getInt64Ty(Context)));
}
} // namespace
