//===- RISCVBaseInfoTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

std::unique_ptr<MCSubtargetInfo> createSTI(StringRef TripleName,
                                           StringRef FeatureStr) {
  LLVMInitializeRISCVTargetInfo();
  LLVMInitializeRISCVTarget();
  LLVMInitializeRISCVTargetMC();
  std::string Error;
  Triple TT(TripleName);
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
  return std::unique_ptr<MCSubtargetInfo>(
      TheTarget->createMCSubtargetInfo(TT, /*CPU=*/"", FeatureStr));
}

RISCVABI::ABI computeTargetABI(StringRef TripleName, StringRef FeatureStr,
                               StringRef ABIName = "") {
  auto STI = createSTI(TripleName, FeatureStr);
  return RISCVABI::computeTargetABI(*STI, ABIName);
}

TEST(ComputeTargetABI, SelectsExpectedABI) {
  EXPECT_EQ(computeTargetABI("riscv32", ""), RISCVABI::ABI_ILP32);
  EXPECT_EQ(computeTargetABI("riscv32", "+f"), RISCVABI::ABI_ILP32F);
  EXPECT_EQ(computeTargetABI("riscv32", "+f,+d"), RISCVABI::ABI_ILP32D);
  EXPECT_EQ(computeTargetABI("riscv64", ""), RISCVABI::ABI_LP64);
  EXPECT_EQ(computeTargetABI("riscv64", "+f"), RISCVABI::ABI_LP64F);
  EXPECT_EQ(computeTargetABI("riscv64", "+f,+d"), RISCVABI::ABI_LP64D);

  // CHERIoT always selects the cheriot ABI by default.
  EXPECT_EQ(computeTargetABI("riscv32", "+xcheriot"), RISCVABI::ABI_CHERIOT);
}

} // namespace
