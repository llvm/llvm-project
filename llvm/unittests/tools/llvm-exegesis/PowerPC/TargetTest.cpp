//===-- TargetTest.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Target.h"

#include <cassert>
#include <memory>

#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm{
namespace exegesis {

void InitializePowerPCExegesisTarget();

namespace {

using testing::NotNull;
using testing::IsEmpty;
using testing::Not;

constexpr const char kTriple[] = "powerpc64le-unknown-linux";

class PowerPCTargetTest : public ::testing::Test {
protected:
  PowerPCTargetTest()
      : ExegesisTarget_(ExegesisTarget::lookup(llvm::Triple(kTriple))) {
    EXPECT_THAT(ExegesisTarget_, NotNull());
    std::string error;
    Target_ = llvm::TargetRegistry::lookupTarget(kTriple, error);
    EXPECT_THAT(Target_, NotNull());
  }
  static void SetUpTestCase() {
    LLVMInitializePowerPCTargetInfo();
    LLVMInitializePowerPCTarget();
    LLVMInitializePowerPCTargetMC();
    InitializePowerPCExegesisTarget();
  }

  const llvm::Target *Target_;
  const ExegesisTarget *const ExegesisTarget_;
};

TEST_F(PowerPCTargetTest, SetRegToConstant) {
  const std::unique_ptr<llvm::MCSubtargetInfo> STI(
      Target_->createMCSubtargetInfo(kTriple, "generic", ""));
  const auto Insts =
      ExegesisTarget_->setRegTo(*STI, llvm::PPC::X0, llvm::APInt());
  EXPECT_THAT(Insts, Not(IsEmpty()));
}

TEST_F(PowerPCTargetTest, DefaultPfmCounters) {
  const std::string Expected = "CYCLES";
  EXPECT_EQ(ExegesisTarget_->getPfmCounters("").CycleCounter, Expected);
  EXPECT_EQ(ExegesisTarget_->getPfmCounters("unknown_cpu").CycleCounter,
            Expected);
}

} // namespace
} // namespace exegesis
} // namespace llvm
