//===--- RDFRegisters.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RDFRegisters.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFunction.h"

using namespace llvm;
using namespace rdf;

PhysicalRegisterInfo::PhysicalRegisterInfo(const TargetRegisterInfo &tri,
      const MachineFunction &mf)
    : TRI(tri) {
  RegInfos.resize(TRI.getNumRegs());

  BitVector BadRC(TRI.getNumRegs());
  for (const TargetRegisterClass *RC : TRI.regclasses()) {
    for (MCPhysReg R : *RC) {
      RegInfo &RI = RegInfos[R];
      if (RI.RegClass != nullptr && !BadRC[R]) {
        if (RC->LaneMask != RI.RegClass->LaneMask) {
          BadRC.set(R);
          RI.RegClass = nullptr;
        }
      } else
        RI.RegClass = RC;
    }
  }

  auto HasPartialOverlaps = [this] (uint32_t Reg) -> bool {
    for (MCRegAliasIterator A(Reg, &TRI, false); A.isValid(); ++A)
      if (!TRI.isSubRegister(Reg, *A) && !TRI.isSubRegister(*A, Reg))
        return true;
    return false;
  };

  for (MCPhysReg R = 1, NR = TRI.getNumRegs(); R != NR; ++R)
    RegInfos[R].Partial = HasPartialOverlaps(R);

  for (MCPhysReg R = 1, NR = TRI.getNumRegs(); R != NR; ++R) {
    MCPhysReg SuperR = R;
    for (MCSuperRegIterator S(R, &TRI, false); S.isValid(); ++S)
      if (!RegInfos[*S].Partial)
        SuperR = *S;
    RegInfos[R].MaxSuper = SuperR;
  }

  for (const uint32_t *RM : TRI.getRegMasks())
    RegMasks.insert(RM);
  for (const MachineBasicBlock &B : mf)
    for (const MachineInstr &In : B)
      for (const MachineOperand &Op : In.operands())
        if (Op.isRegMask())
          RegMasks.insert(Op.getRegMask());
}

RegisterRef PhysicalRegisterInfo::normalize(RegisterRef RR) const {
  if (PhysicalRegisterInfo::isRegMaskId(RR.Reg))
    return RR;
  RegisterId SuperReg = RegInfos[RR.Reg].MaxSuper;
  if (RR.Reg == SuperReg)
    return RR;

  const TargetRegisterClass *RC = RegInfos[RR.Reg].RegClass;
  LaneBitmask RCMask = RC != nullptr ? RC->LaneMask : LaneBitmask(0x00000001);
  LaneBitmask Common = RR.Mask & RCMask;

// Ex: IP/EIP/RIP
//  assert(RC != nullptr || RR.Reg == SuperReg);
  uint32_t Sub = TRI.getSubRegIndex(SuperReg, RR.Reg);
  LaneBitmask SuperMask = TRI.composeSubRegIndexLaneMask(Sub, Common);
  assert(RR.Mask.none() || SuperMask.any());
  return RegisterRef(SuperReg, SuperMask);
}

std::set<RegisterId> PhysicalRegisterInfo::getAliasSet(RegisterId Reg) const {
  // Do not include RR in the alias set.
  std::set<RegisterId> AS;
  assert(isRegMaskId(Reg) || TargetRegisterInfo::isPhysicalRegister(Reg));
  if (isRegMaskId(Reg)) {
    // XXX SLOW
    const uint32_t *MB = getRegMaskBits(Reg);
    for (unsigned i = 1, e = TRI.getNumRegs(); i != e; ++i) {
      if (MB[i/32] & (1u << (i%32)))
        continue;
      AS.insert(i);
    }
    for (const uint32_t *RM : RegMasks) {
      RegisterId MI = getRegMaskId(RM);
      if (MI != Reg && aliasMM(RegisterRef(Reg), RegisterRef(MI)))
        AS.insert(MI);
    }
    return AS;
  }

  for (MCRegAliasIterator AI(Reg, &TRI, false); AI.isValid(); ++AI)
    AS.insert(*AI);
  for (const uint32_t *RM : RegMasks) {
    RegisterId MI = getRegMaskId(RM);
    if (aliasRM(RegisterRef(Reg), RegisterRef(MI)))
      AS.insert(MI);
  }
  return AS;
}

bool PhysicalRegisterInfo::aliasRR(RegisterRef RA, RegisterRef RB) const {
  assert(TargetRegisterInfo::isPhysicalRegister(RA.Reg));
  assert(TargetRegisterInfo::isPhysicalRegister(RB.Reg));

  MCRegUnitMaskIterator UMA(RA.Reg, &TRI);
  MCRegUnitMaskIterator UMB(RB.Reg, &TRI);
  // Reg units are returned in the numerical order.
  while (UMA.isValid() && UMB.isValid()) {
    // Skip units that are masked off in RA.
    std::pair<RegisterId,LaneBitmask> PA = *UMA;
    if (PA.second.any() && (PA.second & RA.Mask).none()) {
      ++UMA;
      continue;
    }
    // Skip units that are masked off in RB.
    std::pair<RegisterId,LaneBitmask> PB = *UMB;
    if (PB.second.any() && (PB.second & RB.Mask).none()) {
      ++UMB;
      continue;
    }

    if (PA.first == PB.first)
      return true;
    if (PA.first < PB.first)
      ++UMA;
    else if (PB.first < PA.first)
      ++UMB;
  }
  return false;
}

bool PhysicalRegisterInfo::aliasRM(RegisterRef RR, RegisterRef RM) const {
  assert(TargetRegisterInfo::isPhysicalRegister(RR.Reg) && isRegMaskId(RM.Reg));
  const uint32_t *MB = getRegMaskBits(RM.Reg);
  bool Preserved = MB[RR.Reg/32] & (1u << (RR.Reg%32));
  // If the lane mask information is "full", e.g. when the given lane mask
  // is a superset of the lane mask from the register class, check the regmask
  // bit directly.
  if (RR.Mask == LaneBitmask::getAll())
    return !Preserved;
  const TargetRegisterClass *RC = RegInfos[RR.Reg].RegClass;
  if (RC != nullptr && (RR.Mask & RC->LaneMask) == RC->LaneMask)
    return !Preserved;

  // Otherwise, check all subregisters whose lane mask overlaps the given
  // mask. For each such register, if it is preserved by the regmask, then
  // clear the corresponding bits in the given mask. If at the end, all
  // bits have been cleared, the register does not alias the regmask (i.e.
  // is it preserved by it).
  LaneBitmask M = RR.Mask;
  for (MCSubRegIndexIterator SI(RR.Reg, &TRI); SI.isValid(); ++SI) {
    LaneBitmask SM = TRI.getSubRegIndexLaneMask(SI.getSubRegIndex());
    if ((SM & RR.Mask).none())
      continue;
    unsigned SR = SI.getSubReg();
    if (!(MB[SR/32] & (1u << (SR%32))))
      continue;
    // The subregister SR is preserved.
    M &= ~SM;
    if (M.none())
      return false;
  }

  return true;
}

bool PhysicalRegisterInfo::aliasMM(RegisterRef RM, RegisterRef RN) const {
  assert(isRegMaskId(RM.Reg) && isRegMaskId(RN.Reg));
  unsigned NumRegs = TRI.getNumRegs();
  const uint32_t *BM = getRegMaskBits(RM.Reg);
  const uint32_t *BN = getRegMaskBits(RN.Reg);

  for (unsigned w = 0, nw = NumRegs/32; w != nw; ++w) {
    // Intersect the negations of both words. Disregard reg=0,
    // i.e. 0th bit in the 0th word.
    uint32_t C = ~BM[w] & ~BN[w];
    if (w == 0)
      C &= ~1;
    if (C)
      return true;
  }

  // Check the remaining registers in the last word.
  unsigned TailRegs = NumRegs % 32;
  if (TailRegs == 0)
    return false;
  unsigned TW = NumRegs / 32;
  uint32_t TailMask = (1u << TailRegs) - 1;
  if (~BM[TW] & ~BN[TW] & TailMask)
    return true;

  return false;
}


bool RegisterAggr::hasAliasOf(RegisterRef RR) const {
  if (PhysicalRegisterInfo::isRegMaskId(RR.Reg)) {
    // XXX SLOW
    const uint32_t *MB = PRI.getRegMaskBits(RR.Reg);
    for (unsigned i = 1, e = PRI.getTRI().getNumRegs(); i != e; ++i) {
      if (MB[i/32] & (1u << (i%32)))
        continue;
      if (hasAliasOf(RegisterRef(i, LaneBitmask::getAll())))
        return true;
    }
    return false;
  }

  RegisterRef NR = PRI.normalize(RR);
  auto F = Masks.find(NR.Reg);
  if (F != Masks.end()) {
    if ((F->second & NR.Mask).any())
      return true;
  }
  if (CheckUnits || PRI.hasPartialOverlaps(NR.Reg)) {
    for (MCRegUnitMaskIterator U(RR.Reg, &PRI.getTRI()); U.isValid(); ++U) {
      std::pair<RegisterId,LaneBitmask> P = *U;
      if (P.second.none() || (P.second & RR.Mask).any())
        if (ExpUnits.test(P.first))
          return true;
    }
  }
  return false;
}

bool RegisterAggr::hasCoverOf(RegisterRef RR) const {
  if (PhysicalRegisterInfo::isRegMaskId(RR.Reg)) {
    // XXX SLOW
    const uint32_t *MB = PRI.getRegMaskBits(RR.Reg);
    for (unsigned i = 1, e = PRI.getTRI().getNumRegs(); i != e; ++i) {
      if (MB[i/32] & (1u << (i%32)))
        continue;
      if (!hasCoverOf(RegisterRef(i, LaneBitmask::getAll())))
        return false;
    }
    return true;
  }

  // Always have a cover for empty lane mask.
  RegisterRef NR = PRI.normalize(RR);
  if (NR.Mask.none())
    return true;
  auto F = Masks.find(NR.Reg);
  if (F != Masks.end()) {
    if ((NR.Mask & F->second) == NR.Mask)
      return true;
  }
  if (CheckUnits || PRI.hasPartialOverlaps(NR.Reg)) {
    for (MCRegUnitMaskIterator U(RR.Reg, &PRI.getTRI()); U.isValid(); ++U) {
      std::pair<RegisterId,LaneBitmask> P = *U;
      if (P.second.none() || (P.second & RR.Mask).any())
        if (!ExpUnits.test(P.first))
          return false;
    }
    return true;
  }
  return false;
}

RegisterAggr &RegisterAggr::insert(RegisterRef RR) {
  if (PhysicalRegisterInfo::isRegMaskId(RR.Reg)) {
    // XXX SLOW
    const uint32_t *MB = PRI.getRegMaskBits(RR.Reg);
    for (unsigned i = 1, e = PRI.getTRI().getNumRegs(); i != e; ++i) {
      if (MB[i/32] & (1u << (i%32)))
        continue;
      insert(RegisterRef(i, LaneBitmask::getAll()));
    }
    return *this;
  }

  RegisterRef NR = PRI.normalize(RR);
  auto F = Masks.find(NR.Reg);
  if (F == Masks.end())
    Masks.insert({NR.Reg, NR.Mask});
  else
    F->second |= NR.Mask;

  // If the register has any partial overlaps, the mask will not be sufficient
  // to accurately represent aliasing/covering information. Add all units to
  // the bit vector.
  if (PRI.hasPartialOverlaps(NR.Reg)) {
    for (MCRegUnitMaskIterator U(RR.Reg, &PRI.getTRI()); U.isValid(); ++U) {
      std::pair<RegisterId,LaneBitmask> P = *U;
      if (P.second.none() || (P.second & RR.Mask).none())
        continue;
      ExpUnits.set(P.first);
      CheckUnits = true;
    }
  }
  return *this;
}

RegisterAggr &RegisterAggr::insert(const RegisterAggr &RG) {
  for (std::pair<RegisterId,LaneBitmask> P : RG.Masks)
    insert(RegisterRef(P.first, P.second));
  return *this;
}

RegisterAggr &RegisterAggr::clear(RegisterRef RR) {
  if (PhysicalRegisterInfo::isRegMaskId(RR.Reg)) {
    // XXX SLOW
    const uint32_t *MB = PRI.getRegMaskBits(RR.Reg);
    for (unsigned i = 1, e = PRI.getTRI().getNumRegs(); i != e; ++i) {
      if (MB[i/32] & (1u << (i%32)))
        continue;
      clear(RegisterRef(i, LaneBitmask::getAll()));
    }
    return *this;
  }

  RegisterRef NR = PRI.normalize(RR);
  auto F = Masks.find(NR.Reg);
  if (F == Masks.end())
    return *this;
  LaneBitmask NewM = F->second & ~NR.Mask;
  if (NewM.none())
    Masks.erase(F);
  else
    F->second = NewM;
  return *this;
}

RegisterAggr &RegisterAggr::clear(const RegisterAggr &RG) {
  for (std::pair<RegisterId,LaneBitmask> P : RG.Masks)
    clear(RegisterRef(P.first, P.second));
  return *this;
}

RegisterRef RegisterAggr::clearIn(RegisterRef RR) const {
  RegisterAggr T(PRI);
  T.insert(RR).clear(*this);
  if (T.empty())
    return RegisterRef();
  return RegisterRef(T.begin()->first, T.begin()->second);
}

void RegisterAggr::print(raw_ostream &OS) const {
  OS << '{';
  for (auto I : Masks)
    OS << ' ' << PrintReg(I.first, &PRI.getTRI())
       << PrintLaneMaskOpt(I.second);
  OS << " }";
}

