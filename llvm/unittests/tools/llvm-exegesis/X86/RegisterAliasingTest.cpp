//===-- X86RegisterAliasingTest.cpp --------------------------------*- C++ -*-===//
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
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {
namespace {

class X86RegisterAliasingTest : public X86TestBase {};

TEST_F(X86RegisterAliasingTest, TrackSimpleRegister) {
  const auto &RegInfo = State.getRegInfo();
  const RegisterAliasingTracker tracker(RegInfo, X86::EAX);
  std::set<MCPhysReg> ActualAliasedRegisters;
  for (unsigned I : tracker.aliasedBits().set_bits())
    ActualAliasedRegisters.insert(static_cast<MCPhysReg>(I));
  const std::set<MCPhysReg> ExpectedAliasedRegisters = {
      X86::AL, X86::AH, X86::AX, X86::EAX, X86::HAX, X86::RAX};
  ASSERT_THAT(ActualAliasedRegisters, ExpectedAliasedRegisters);
  for (MCPhysReg aliased : ExpectedAliasedRegisters) {
    ASSERT_THAT(tracker.getOrigin(aliased), X86::EAX);
  }
}

TEST_F(X86RegisterAliasingTest, TrackRegisterClass) {
  // The alias bits for GR8_ABCD_LRegClassID are the union of the alias bits for
  // AL, BL, CL and DL.
  const auto &RegInfo = State.getRegInfo();
  const BitVector NoReservedReg(RegInfo.getNumRegs());

  const RegisterAliasingTracker RegClassTracker(
      RegInfo, NoReservedReg, RegInfo.getRegClass(X86::GR8_ABCD_LRegClassID));

  BitVector sum(RegInfo.getNumRegs());
  sum |= RegisterAliasingTracker(RegInfo, X86::AL).aliasedBits();
  sum |= RegisterAliasingTracker(RegInfo, X86::BL).aliasedBits();
  sum |= RegisterAliasingTracker(RegInfo, X86::CL).aliasedBits();
  sum |= RegisterAliasingTracker(RegInfo, X86::DL).aliasedBits();

  ASSERT_THAT(RegClassTracker.aliasedBits(), sum);
}

TEST_F(X86RegisterAliasingTest, TrackRegisterClassCache) {
  // Fetching twice the same tracker yields the same pointers.
  const auto &RegInfo = State.getRegInfo();
  const BitVector NoReservedReg(RegInfo.getNumRegs());
  RegisterAliasingTrackerCache Cache(RegInfo, NoReservedReg);
  ASSERT_THAT(&Cache.getRegister(X86::AX), &Cache.getRegister(X86::AX));

  ASSERT_THAT(&Cache.getRegisterClass(X86::GR8_ABCD_LRegClassID),
              &Cache.getRegisterClass(X86::GR8_ABCD_LRegClassID));
}

} // namespace
} // namespace exegesis
} // namespace llvm
