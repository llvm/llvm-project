//===- AArch64TargetInfoTest.cpp - AArch64 ABI unit tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

namespace {

using ABIType = llvm::abi::Type;
using llvm::abi::AArch64ABIKind;
using llvm::abi::ArgEntry;
using llvm::abi::ArgInfo;
using llvm::abi::createAArch64TargetInfo;
using llvm::abi::FunctionInfo;
using llvm::abi::TargetInfo;
using llvm::abi::TypeBuilder;

class AArch64TargetInfoTest : public ::testing::Test {
protected:
  llvm::BumpPtrAllocator Alloc;
  TypeBuilder TB;
  const ABIType *I32;
  const ABIType *F64;
  const ABIType *Ptr;
  const ABIType *Void;

  AArch64TargetInfoTest()
      : TB(Alloc), I32(TB.getIntegerType(32, llvm::Align(4), /*Signed=*/true)),
        F64(TB.getFloatType(llvm::APFloat::IEEEdouble(), llvm::Align(8))),
        Ptr(TB.getPointerType(64, llvm::Align(8))), Void(TB.getVoidType()) {}
};

static void expectAllDirect(const FunctionInfo &FI) {
  EXPECT_TRUE(FI.getReturnInfo().isDirect());
  EXPECT_EQ(FI.getReturnInfo().getCoerceToType(), nullptr);
  for (const ArgEntry &Arg : FI.arguments()) {
    EXPECT_TRUE(Arg.Info.isDirect());
    EXPECT_EQ(Arg.Info.getCoerceToType(), nullptr);
  }
}

TEST_F(AArch64TargetInfoTest, ComputeInfoMarksReturnAndArgsDirect) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIKind::AAPCS);
  std::unique_ptr<FunctionInfo> FI =
      FunctionInfo::create(llvm::CallingConv::C, I32, {I32, F64, Ptr});

  // Poison the defaults so a no-op would fail the assertions below.
  FI->getReturnInfo() = ArgInfo::getIgnore();
  for (ArgEntry &Arg : FI->arguments())
    Arg.Info = ArgInfo::getIgnore();

  TI->computeInfo(*FI);
  expectAllDirect(*FI);
}

TEST_F(AArch64TargetInfoTest, ComputeInfoVoidReturnNoArgs) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIKind::DarwinPCS);
  std::unique_ptr<FunctionInfo> FI =
      FunctionInfo::create(llvm::CallingConv::C, Void, {});

  FI->getReturnInfo() = ArgInfo::getIgnore();
  TI->computeInfo(*FI);

  EXPECT_TRUE(FI->getReturnInfo().isDirect());
  EXPECT_TRUE(FI->arguments().empty());
}

TEST_F(AArch64TargetInfoTest, ComputeInfoAAPCSSoftSameAsStub) {
  std::unique_ptr<TargetInfo> TI =
      createAArch64TargetInfo(TB, AArch64ABIKind::AAPCSSoft);
  std::unique_ptr<FunctionInfo> FI =
      FunctionInfo::create(llvm::CallingConv::C, F64, {I32});

  FI->getReturnInfo() = ArgInfo::getIgnore();
  FI->getArgInfo(0).Info = ArgInfo::getIgnore();

  TI->computeInfo(*FI);
  expectAllDirect(*FI);
}

} // namespace
