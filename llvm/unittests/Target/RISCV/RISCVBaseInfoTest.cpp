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
  return cantFail(RISCVABI::computeTargetABI(*STI, ABIName));
}

// Returns the error message for a StringRef/FeatureStr/ABIName combination
// that is expected to be rejected by computeTargetABI.
std::string computeTargetABIError(StringRef TripleName, StringRef FeatureStr,
                                  StringRef ABIName) {
  auto STI = createSTI(TripleName, FeatureStr);
  Expected<RISCVABI::ABI> Result = RISCVABI::computeTargetABI(*STI, ABIName);
  if (Result) {
    ADD_FAILURE() << "expected an error for -target-abi " << ABIName;
    return {};
  }
  return toString(Result.takeError());
}

TEST(ComputeTargetABI, SelectsExpectedABI) {
  EXPECT_EQ(computeTargetABI("riscv32", ""), RISCVABI::ABI_ILP32);
  EXPECT_EQ(computeTargetABI("riscv32", "+f"), RISCVABI::ABI_ILP32F);
  EXPECT_EQ(computeTargetABI("riscv32", "+f,+d"), RISCVABI::ABI_ILP32D);
  EXPECT_EQ(computeTargetABI("riscv64", ""), RISCVABI::ABI_LP64);
  EXPECT_EQ(computeTargetABI("riscv64", "+f"), RISCVABI::ABI_LP64F);
  EXPECT_EQ(computeTargetABI("riscv64", "+f,+d"), RISCVABI::ABI_LP64D);

  // With the Y extension enabled and no explicit -target-abi, the capability
  // ABI is selected by default.
  EXPECT_EQ(computeTargetABI("riscv32", "+experimental-y"),
            RISCVABI::ABI_IL32PC64);
  EXPECT_EQ(computeTargetABI("riscv32", "+experimental-y,+f"),
            RISCVABI::ABI_IL32PC64F);
  EXPECT_EQ(computeTargetABI("riscv32", "+experimental-y,+f,+d"),
            RISCVABI::ABI_IL32PC64D);
  EXPECT_EQ(computeTargetABI("riscv64", "+experimental-y"),
            RISCVABI::ABI_L64PC128);
  EXPECT_EQ(computeTargetABI("riscv64", "+experimental-y,+f"),
            RISCVABI::ABI_L64PC128F);
  EXPECT_EQ(computeTargetABI("riscv64", "+experimental-y,+f,+d"),
            RISCVABI::ABI_L64PC128D);

  // An explicitly requested ABI is unaffected by the Y-based default: even
  // with Y (and F/D) enabled, asking for the plain integer ABI still gives
  // the integer ABI rather than the capability one.
  EXPECT_EQ(computeTargetABI("riscv32", "+experimental-y", /*ABIName=*/"ilp32"),
            RISCVABI::ABI_ILP32);
  EXPECT_EQ(computeTargetABI("riscv64", "+experimental-y,+f,+d",
                             /*ABIName=*/"lp64d"),
            RISCVABI::ABI_LP64D);

  // CHERIoT always selects the cheriot ABI by default.
  EXPECT_EQ(computeTargetABI("riscv32", "+xcheriot"), RISCVABI::ABI_CHERIOT);
}

TEST(ComputeTargetABI, ReportsInvalidExplicitABI) {
  EXPECT_EQ(computeTargetABIError("riscv32", "", "foo"),
            "'foo' is not a recognized ABI for this target");
  EXPECT_EQ(computeTargetABIError("riscv64", "", "ilp32"),
            "32-bit ABIs are not supported for 64-bit targets");
  EXPECT_EQ(computeTargetABIError("riscv32", "", "lp64"),
            "64-bit ABIs are not supported for 32-bit targets");
  EXPECT_EQ(computeTargetABIError("riscv32", "", "ilp32f"),
            "hard-float 'f' ABI can't be used for a target that doesn't "
            "support the F instruction set extension");
  EXPECT_EQ(computeTargetABIError("riscv64", "", "lp64f"),
            "hard-float 'f' ABI can't be used for a target that doesn't "
            "support the F instruction set extension");
  EXPECT_EQ(computeTargetABIError("riscv32", "", "ilp32d"),
            "hard-float 'd' ABI can't be used for a target that doesn't "
            "support the D instruction set extension");
  EXPECT_EQ(computeTargetABIError("riscv32", "+f", "ilp32d"),
            "hard-float 'd' ABI can't be used for a target that doesn't "
            "support the D instruction set extension");
  EXPECT_EQ(computeTargetABIError("riscv32", "+e", "ilp32"),
            "only the ilp32e ABI is supported for RV32E");
  EXPECT_EQ(computeTargetABIError("riscv32", "+e,+xcheriot", "ilp32e"),
            "only the cheriot ABI is supported for XCheriot");
  EXPECT_EQ(computeTargetABIError("riscv64", "+e", "lp64"),
            "only the lp64e ABI is supported for RV64E");
  EXPECT_EQ(
      computeTargetABIError("riscv32", "+experimental-y,+f,+d", "l64pc128d"),
      "64-bit ABIs are not supported for 32-bit targets");
  EXPECT_EQ(
      computeTargetABIError("riscv64", "+experimental-y,+f,+d", "il32pc64f"),
      "32-bit ABIs are not supported for 64-bit targets");
  EXPECT_EQ(computeTargetABIError("riscv64", "+f,+d", "l64pc128d"),
            "'l64pc128d' ABI is only supported for RVY targets");
  EXPECT_EQ(computeTargetABIError(
                "riscv64", "+experimental-y,+f,+d,+xllvmrvyipm", "l64pc128d"),
            "'l64pc128d' ABI is only supported for RVY targets");
}

} // namespace
