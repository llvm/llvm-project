//===-- RegisterAliasingTest.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "RegisterAliasing.h"

#include <cassert>
#include <memory>

#include "TestBase.h"
#include "X86InstrInfo.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {
namespace {

class RegisterAliasingTest : public X86TestBase {};

TEST_F(RegisterAliasingTest, TrackSimpleRegister) {
  const auto &RegInfo = State.getRegInfo();
  const RegisterAliasingTracker tracker(RegInfo, llvm::X86::EAX);
  std::set<llvm::MCPhysReg> ActualAliasedRegisters;
  for (unsigned I : tracker.aliasedBits().set_bits())
    ActualAliasedRegisters.insert(static_cast<llvm::MCPhysReg>(I));
  const std::set<llvm::MCPhysReg> ExpectedAliasedRegisters = {
      llvm::X86::AL,  llvm::X86::AH,  llvm::X86::AX,
      llvm::X86::EAX, llvm::X86::HAX, llvm::X86::RAX};
  ASSERT_THAT(ActualAliasedRegisters, ExpectedAliasedRegisters);
  for (llvm::MCPhysReg aliased : ExpectedAliasedRegisters) {
    ASSERT_THAT(tracker.getOrigin(aliased), llvm::X86::EAX);
  }
}

TEST_F(RegisterAliasingTest, TrackRegisterClass) {
  // The alias bits for GR8_ABCD_LRegClassID are the union of the alias bits for
  // AL, BL, CL and DL.
  const auto &RegInfo = State.getRegInfo();
  const llvm::BitVector NoReservedReg(RegInfo.getNumRegs());

  const RegisterAliasingTracker RegClassTracker(
      RegInfo, NoReservedReg,
      RegInfo.getRegClass(llvm::X86::GR8_ABCD_LRegClassID));

  llvm::BitVector sum(RegInfo.getNumRegs());
  sum |= RegisterAliasingTracker(RegInfo, llvm::X86::AL).aliasedBits();
  sum |= RegisterAliasingTracker(RegInfo, llvm::X86::BL).aliasedBits();
  sum |= RegisterAliasingTracker(RegInfo, llvm::X86::CL).aliasedBits();
  sum |= RegisterAliasingTracker(RegInfo, llvm::X86::DL).aliasedBits();

  ASSERT_THAT(RegClassTracker.aliasedBits(), sum);
}

TEST_F(RegisterAliasingTest, TrackRegisterClassCache) {
  // Fetching twice the same tracker yields the same pointers.
  const auto &RegInfo = State.getRegInfo();
  const llvm::BitVector NoReservedReg(RegInfo.getNumRegs());
  RegisterAliasingTrackerCache Cache(RegInfo, NoReservedReg);
  ASSERT_THAT(&Cache.getRegister(llvm::X86::AX),
              &Cache.getRegister(llvm::X86::AX));

  ASSERT_THAT(&Cache.getRegisterClass(llvm::X86::GR8_ABCD_LRegClassID),
              &Cache.getRegisterClass(llvm::X86::GR8_ABCD_LRegClassID));
}

} // namespace
} // namespace exegesis
} // namespace llvm
