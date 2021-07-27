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

#include "MipsInstrInfo.h"
#include "TestBase.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {
namespace {

class MipsRegisterAliasingTest : public MipsTestBase {};

TEST_F(MipsRegisterAliasingTest, TrackSimpleRegister) {
  const auto &RegInfo = State.getRegInfo();
  const RegisterAliasingTracker tracker(RegInfo, Mips::T0_64);
  std::set<MCPhysReg> ActualAliasedRegisters;
  for (unsigned I : tracker.aliasedBits().set_bits())
    ActualAliasedRegisters.insert(static_cast<MCPhysReg>(I));
  const std::set<MCPhysReg> ExpectedAliasedRegisters = {Mips::T0, Mips::T0_64};
  ASSERT_THAT(ActualAliasedRegisters, ExpectedAliasedRegisters);
  for (MCPhysReg aliased : ExpectedAliasedRegisters) {
    ASSERT_THAT(tracker.getOrigin(aliased), Mips::T0_64);
  }
}

TEST_F(MipsRegisterAliasingTest, TrackRegisterClass) {
  // The alias bits for
  // GPR64_with_sub_32_in_GPRMM16MoveP_and_GPRMM16ZeroRegClassID
  // are the union of the alias bits for ZERO_64, V0_64, V1_64 and S1_64.
  const auto &RegInfo = State.getRegInfo();
  const BitVector NoReservedReg(RegInfo.getNumRegs());

  const RegisterAliasingTracker RegClassTracker(
      RegInfo, NoReservedReg,
      RegInfo.getRegClass(
          Mips::GPR64_with_sub_32_in_GPRMM16MoveP_and_GPRMM16ZeroRegClassID));

  BitVector sum(RegInfo.getNumRegs());
  sum |= RegisterAliasingTracker(RegInfo, Mips::ZERO_64).aliasedBits();
  sum |= RegisterAliasingTracker(RegInfo, Mips::V0_64).aliasedBits();
  sum |= RegisterAliasingTracker(RegInfo, Mips::V1_64).aliasedBits();
  sum |= RegisterAliasingTracker(RegInfo, Mips::S1_64).aliasedBits();

  ASSERT_THAT(RegClassTracker.aliasedBits(), sum);
}

TEST_F(MipsRegisterAliasingTest, TrackRegisterClassCache) {
  // Fetching the same tracker twice yields the same pointers.
  const auto &RegInfo = State.getRegInfo();
  const BitVector NoReservedReg(RegInfo.getNumRegs());
  RegisterAliasingTrackerCache Cache(RegInfo, NoReservedReg);
  ASSERT_THAT(&Cache.getRegister(Mips::T0), &Cache.getRegister(Mips::T0));

  ASSERT_THAT(&Cache.getRegisterClass(Mips::ACC64RegClassID),
              &Cache.getRegisterClass(Mips::ACC64RegClassID));
}

} // namespace
} // namespace exegesis
} // namespace llvm
