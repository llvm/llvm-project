//===-- RegisterAliasingTracker.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RegisterAliasing.h"

namespace exegesis {

llvm::BitVector getAliasedBits(const llvm::MCRegisterInfo &RegInfo,
                               const llvm::BitVector &SourceBits) {
  llvm::BitVector AliasedBits(RegInfo.getNumRegs());
  for (const size_t PhysReg : SourceBits.set_bits()) {
    using RegAliasItr = llvm::MCRegAliasIterator;
    for (auto Itr = RegAliasItr(PhysReg, &RegInfo, true); Itr.isValid();
         ++Itr) {
      AliasedBits.set(*Itr);
    }
  }
  return AliasedBits;
}

RegisterAliasingTracker::RegisterAliasingTracker(
    const llvm::MCRegisterInfo &RegInfo)
    : SourceBits(RegInfo.getNumRegs()), AliasedBits(RegInfo.getNumRegs()),
      Origins(RegInfo.getNumRegs()) {}

RegisterAliasingTracker::RegisterAliasingTracker(
    const llvm::MCRegisterInfo &RegInfo, const llvm::BitVector &ReservedReg,
    const llvm::MCRegisterClass &RegClass)
    : RegisterAliasingTracker(RegInfo) {
  for (llvm::MCPhysReg PhysReg : RegClass)
    if (!ReservedReg[PhysReg]) // Removing reserved registers.
      SourceBits.set(PhysReg);
  FillOriginAndAliasedBits(RegInfo, SourceBits);
}

RegisterAliasingTracker::RegisterAliasingTracker(
    const llvm::MCRegisterInfo &RegInfo, const llvm::MCPhysReg PhysReg)
    : RegisterAliasingTracker(RegInfo) {
  SourceBits.set(PhysReg);
  FillOriginAndAliasedBits(RegInfo, SourceBits);
}

void RegisterAliasingTracker::FillOriginAndAliasedBits(
    const llvm::MCRegisterInfo &RegInfo, const llvm::BitVector &SourceBits) {
  using RegAliasItr = llvm::MCRegAliasIterator;
  for (const size_t PhysReg : SourceBits.set_bits()) {
    for (auto Itr = RegAliasItr(PhysReg, &RegInfo, true); Itr.isValid();
         ++Itr) {
      AliasedBits.set(*Itr);
      Origins[*Itr] = PhysReg;
    }
  }
}

RegisterAliasingTrackerCache::RegisterAliasingTrackerCache(
    const llvm::MCRegisterInfo &RegInfo, const llvm::BitVector &ReservedReg)
    : RegInfo(RegInfo), ReservedReg(ReservedReg),
      EmptyRegisters(RegInfo.getNumRegs()) {}

const RegisterAliasingTracker &
RegisterAliasingTrackerCache::getRegister(llvm::MCPhysReg PhysReg) const {
  auto &Found = Registers[PhysReg];
  if (!Found)
    Found.reset(new RegisterAliasingTracker(RegInfo, PhysReg));
  return *Found;
}

const RegisterAliasingTracker &
RegisterAliasingTrackerCache::getRegisterClass(unsigned RegClassIndex) const {
  auto &Found = RegisterClasses[RegClassIndex];
  const auto &RegClass = RegInfo.getRegClass(RegClassIndex);
  if (!Found)
    Found.reset(new RegisterAliasingTracker(RegInfo, ReservedReg, RegClass));
  return *Found;
}

} // namespace exegesis
