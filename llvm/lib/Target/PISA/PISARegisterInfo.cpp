//===-- PISARegisterInfo.cpp - PISA Register Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISARegisterInfo.h"
#include "PISA.h"
#include "PISASubtarget.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringTable.h"
#include "llvm/CodeGen/MachineFunction.h"

#define GET_REGINFO_TARGET_DESC
#include "PISAGenRegisterInfo.inc"

using namespace llvm;

namespace {
struct SpecialRegEntry {
  StringTable::Offset Name;
};

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define GET_SpecialRegNames_IMPL
#include "PISAGenSearchableTables.inc"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
} // anonymous namespace

static_assert(PISA::NUM_TARGET_SUBREGS == 63, "updated needed!");
DenseMap<unsigned, PISARegisterInfo::SwizzleDesc> PISARegisterInfo::SwizzleMap =
    {
        {PISA::NoSubRegister, {PISA::Swizzle::NONE, nullptr}},
        {PISA::sub8_0, {PISA::Swizzle::X, ".x"}},
        {PISA::sub8_1, {PISA::Swizzle::Y, ".y"}},
        {PISA::sub8_2, {PISA::Swizzle::Z, ".z"}},
        {PISA::sub8_3, {PISA::Swizzle::W, ".w"}},
        {PISA::sub16_0, {PISA::Swizzle::X, ".x"}},
        {PISA::sub16_1, {PISA::Swizzle::Y, ".y"}},
        {PISA::sub16_2, {PISA::Swizzle::Z, ".z"}},
        {PISA::sub16_3, {PISA::Swizzle::W, ".w"}},
        {PISA::sub32_0, {PISA::Swizzle::X, ".x"}},
        {PISA::sub32_1, {PISA::Swizzle::Y, ".y"}},
        {PISA::sub32_2, {PISA::Swizzle::Z, ".z"}},
        {PISA::sub32_3, {PISA::Swizzle::W, ".w"}},
        {PISA::sub64_0, {PISA::Swizzle::X, ".x"}},
        {PISA::sub64_1, {PISA::Swizzle::Y, ".y"}},
        {PISA::sub64_2, {PISA::Swizzle::Z, ".z"}},
        {PISA::sub64_3, {PISA::Swizzle::W, ".w"}},
        {PISA::sub8_xy, {PISA::Swizzle::XY, ".xy"}},
        {PISA::sub8_zw, {PISA::Swizzle::ZW, ".zw"}},
        {PISA::sub16_xy, {PISA::Swizzle::XY, ".xy"}},
        {PISA::sub16_zw, {PISA::Swizzle::ZW, ".zw"}},
        {PISA::sub32_xy, {PISA::Swizzle::XY, ".xy"}},
        {PISA::sub32_zw, {PISA::Swizzle::ZW, ".zw"}},
        {PISA::sub64_xy, {PISA::Swizzle::XY, ".xy"}},
        {PISA::sub64_zw, {PISA::Swizzle::ZW, ".zw"}},
};

bool PISARegisterInfo::shouldCoalesce(
    MachineInstr *MI, const TargetRegisterClass *SrcRC, unsigned SubReg,
    const TargetRegisterClass *DstRC, unsigned DstSubReg,
    const TargetRegisterClass *NewRC, LiveIntervals &LIS) const {

  if (!MI->isCopy())
    return false;

  auto IsLegalSwizzle = [&](unsigned Subreg) {
    auto Swizzle = SwizzleMap.find(Subreg);
    return Swizzle != SwizzleMap.end();
  };

  return IsLegalSwizzle(SubReg) && IsLegalSwizzle(DstSubReg);
}

PISARegisterInfo::PISARegisterInfo() : PISAGenRegisterInfo(PISA::DummyReg) {
  // Tablegen can sometimes synthesize register classes if you don't set
  // subregs explicitly on some regs. For now at least, we probably want to
  // be explicit about what regclasses exist. If you added a register class
  // explicitly, go ahead and update this number. If not, you might want to
  // figure out what happened.
  // New reg classes must also be reflected in PISATargetLowering constructor
  static_assert(std::size(PISAMCRegisterClassStorage.Classes) == 30);
  unsigned NumRCs = getNumRegClasses();
  for (unsigned I = 0; I < NumRCs; I++) {
    auto *RC = getRegClass(I);
    auto RCD = std::make_unique<RegClassDescription>();
    if (RC == &PISA::RegV64_32bRegClass) {
      // RegV64_32b has no per-element sub-register structure (LLVM's
      // LaneBitmask cannot represent 64 lanes), so derive the metadata
      // from the known type rather than from lane masks.
      RCD->NumElements = 64;
      RCD->ScalarBitSize = 32;
    } else {
      RCD->NumElements = RC->LaneMask.getNumLanes();
      RCD->ScalarBitSize = getScalarBitSize(RC, RCD->NumElements);
    }
    auto Key = std::make_pair(RCD->NumElements, RCD->ScalarBitSize);
    if (!VecRegClassMap[Key])
      VecRegClassMap[std::make_pair(RCD->NumElements, RCD->ScalarBitSize)] = RC;
    RegClassMap[RC] = std::move(RCD);
  }

  // Precompute special register BitVector for O(1) lookup.
  SpecialRegs.resize(getNumRegs());
  for (unsigned I = 1, E = getNumRegs(); I < E; ++I) {
    MCRegister Reg(I);
    if (lookupSpecialRegByName(getName(Reg)))
      SpecialRegs.set(I);
  }
}

BitVector PISARegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  // Reserve DummyReg (used as a sentinel in machine instructions) and all
  // special registers (hardware-defined values like %localid, %groupid, etc.
  // that appear in machine instructions without explicit definitions).
  //
  // General-purpose physical registers are left unreserved so that
  // RegisterClassInfo::getNumAllocatableRegs returns non-zero values for
  // every register class. This is required because the register coalescer
  // refuses to coalesce into register classes with zero allocatable
  // registers. PISA works entirely with virtual registers and disables
  // physical register allocation (createTargetRegisterAllocator returns
  // nullptr), so unreserved physical registers are never actually allocated.
  BitVector Reserved(getNumRegs());
  Reserved.set(PISA::DummyReg);
  Reserved |= SpecialRegs;
  return Reserved;
}

const MCPhysReg *
PISARegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const MCPhysReg CalleeSavedReg = {0};
  return &CalleeSavedReg;
}

unsigned PISARegisterInfo::getScalarBitSize(const TargetRegisterClass *RC,
                                            unsigned NumElts) const {
  unsigned RegSizeInBits = getRegSizeInBits(*RC);
  assert((RegSizeInBits % NumElts == 0) && "not divisible?");
  return RegSizeInBits / NumElts;
}

unsigned PISARegisterInfo::getSubRegIdx(unsigned Size, unsigned Idx) const {
  assert(Idx < 4 && "Only 4 sub-registers supported!");

  static unsigned Subregs[4][4] = {
      {PISA::sub8_0, PISA::sub8_1, PISA::sub8_2, PISA::sub8_3},
      {PISA::sub16_0, PISA::sub16_1, PISA::sub16_2, PISA::sub16_3},
      {PISA::sub32_0, PISA::sub32_1, PISA::sub32_2, PISA::sub32_3},
      {PISA::sub64_0, PISA::sub64_1, PISA::sub64_2, PISA::sub64_3}};

  switch (Size) {
  case 8:
    return Subregs[0][Idx];
  case 16:
    return Subregs[1][Idx];
  case 32:
    return Subregs[2][Idx];
  case 64:
    return Subregs[3][Idx];
  default:
    assert(0 && "unknown type size!");
    break;
  }
  return 0;
}

unsigned PISARegisterInfo::getCompositeSubRegIdx(unsigned Size, unsigned Base,
                                                 unsigned Count) const {
  // Only nameable 2-element pairs: .xy (base 0) and .zw (base 2).
  if (Count != 2 || (Base != 0 && Base != 2))
    return 0;
  bool Low = (Base == 0);
  switch (Size) {
  case 8:
    return Low ? PISA::sub8_xy : PISA::sub8_zw;
  case 16:
    return Low ? PISA::sub16_xy : PISA::sub16_zw;
  case 32:
    return Low ? PISA::sub32_xy : PISA::sub32_zw;
  case 64:
    return Low ? PISA::sub64_xy : PISA::sub64_zw;
  default:
    return 0;
  }
}

PISA::Swizzle PISARegisterInfo::getSwizzle(unsigned SubReg) const {
  auto Swizzle = SwizzleMap.find(SubReg);
  assert((Swizzle != SwizzleMap.end()) && "invalid swizzle!");
  return Swizzle->second.Swizzle;
}

const char *PISARegisterInfo::getSwizzleName(unsigned SubReg) const {
  auto Swizzle = SwizzleMap.find(SubReg);
  assert((Swizzle != SwizzleMap.end()) && "invalid swizzle!");
  return Swizzle->second.SwizzleName;
}

bool PISARegisterInfo::isSelectorSwizzle(PISA::Swizzle Swizzle) {
  switch (Swizzle) {
  case PISA::Swizzle::X:
  case PISA::Swizzle::Y:
  case PISA::Swizzle::Z:
  case PISA::Swizzle::W:
    return true;
  default:
    return false;
  }
}

const TargetRegisterClass *PISARegisterInfo::getRegClassFromLLT(LLT Ty) const {
  if (Ty.isScalar() || Ty.isPointer()) {
    switch (Ty.getSizeInBits()) {
    case 1:
      return &PISA::PredRegClass;
    case 8:
      return &PISA::Reg8bRegClass;
    case 16:
      return &PISA::Reg16bRegClass;
    case 32:
      return &PISA::Reg32bRegClass;
    case 64:
      return &PISA::Reg64bRegClass;
    case 128:
      return &PISA::Reg128bRegClass;
    default:
      break;
    }
  } else if (Ty.isVector()) {
    unsigned NumElts = Ty.getNumElements();
    unsigned BitSize = Ty.getScalarSizeInBits();
    return getVectorRegClass(NumElts, BitSize);
  }

  llvm_unreachable("unhandled LLT!");
}

unsigned
PISARegisterInfo::getNumEltsFromRegClass(const TargetRegisterClass *RC) const {
  auto I = RegClassMap.find(RC);
  assert(I != RegClassMap.end());
  auto &RCD = I->second;
  return RCD->NumElements;
}

unsigned
PISARegisterInfo::getBitSizeFromRegClass(const TargetRegisterClass *RC) const {
  auto I = RegClassMap.find(RC);
  assert(I != RegClassMap.end());
  auto &RCD = I->second;
  return RCD->ScalarBitSize;
}

const TargetRegisterClass *
PISARegisterInfo::getVectorRegClass(unsigned NumElts, unsigned BitSize) const {
  auto P = std::make_pair(NumElts, BitSize);
  auto I = VecRegClassMap.find(P);
  assert(I != VecRegClassMap.end());
  return I->second;
}

bool PISARegisterInfo::isSpecialReg(Register Reg) const {
  return Reg.isPhysical() && SpecialRegs.test(Reg.asMCReg());
}

const TargetRegisterClass *
PISARegisterInfo::getMatchingSuperRegClass(const TargetRegisterClass *A,
                                           const TargetRegisterClass *B,
                                           unsigned SubIdx) const {
  switch (SubIdx) {
  default:
    return PISAGenRegisterInfo::getMatchingSuperRegClass(A, B, SubIdx);
  // generic code (above) is unable to find 'hi' subreg
  // index within 2-element vector ('.zw' within v2b?).
  case PISA::sub8_zw:
  case PISA::sub16_zw:
  case PISA::sub32_zw:
  case PISA::sub64_zw:
    return PISAGenRegisterInfo::getSubClassWithSubReg(A, SubIdx);
  }
}
