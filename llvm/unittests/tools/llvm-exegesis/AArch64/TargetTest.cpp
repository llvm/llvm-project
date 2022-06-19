//===-- TargetTest.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Target.h"

#include <cassert>
#include <memory>

#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

void InitializeAArch64ExegesisTarget();

namespace {

using testing::Gt;
using testing::IsEmpty;
using testing::Not;
using testing::NotNull;

constexpr const char kTriple[] = "aarch64-unknown-linux";

class AArch64TargetTest : public ::testing::Test {
protected:
  AArch64TargetTest()
      : ExegesisTarget_(ExegesisTarget::lookup(Triple(kTriple))) {
    EXPECT_THAT(ExegesisTarget_, NotNull());
    std::string error;
    Target_ = TargetRegistry::lookupTarget(kTriple, error);
    EXPECT_THAT(Target_, NotNull());
    STI_.reset(
        Target_->createMCSubtargetInfo(kTriple, "generic", /*no features*/ ""));
  }

  static void SetUpTestCase() {
    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
    InitializeAArch64ExegesisTarget();
  }

  std::vector<MCInst> setRegTo(unsigned Reg, const APInt &Value) {
    return ExegesisTarget_->setRegTo(*STI_, Reg, Value);
  }

  const Target *Target_;
  const ExegesisTarget *const ExegesisTarget_;
  std::unique_ptr<MCSubtargetInfo> STI_;
};

TEST_F(AArch64TargetTest, SetRegToConstant) {
  // The AArch64 target currently doesn't know how to set register values.
  const auto Insts = setRegTo(AArch64::X0, APInt());
  EXPECT_THAT(Insts, Not(IsEmpty()));
}

TEST_F(AArch64TargetTest, DefaultPfmCounters) {
  const std::string Expected = "CPU_CYCLES";
  EXPECT_EQ(ExegesisTarget_->getPfmCounters("").CycleCounter, Expected);
  EXPECT_EQ(ExegesisTarget_->getPfmCounters("unknown_cpu").CycleCounter,
            Expected);
}

} // namespace
} // namespace exegesis
} // namespace llvm
