//==- TargetRegisterInfo.cpp - Target Register Information Implementation --==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetRegisterInfo interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Printable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <cassert>
#include <utility>

#define DEBUG_TYPE "target-reg-info"

using namespace llvm;

TargetRegisterInfo::TargetRegisterInfo(const TargetRegisterInfoDesc *ID,
                             regclass_iterator RCB, regclass_iterator RCE,
                             const char *const *SRINames,
                             const LaneBitmask *SRILaneMasks,
                             LaneBitmask SRICoveringLanes,
                             const RegClassInfo *const RCIs,
                             unsigned Mode)
  : InfoDesc(ID), SubRegIndexNames(SRINames),
    SubRegIndexLaneMasks(SRILaneMasks),
    RegClassBegin(RCB), RegClassEnd(RCE),
    CoveringLanes(SRICoveringLanes),
    RCInfos(RCIs), HwMode(Mode) {
}

TargetRegisterInfo::~TargetRegisterInfo() = default;

void TargetRegisterInfo::markSuperRegs(BitVector &RegisterSet, unsigned Reg)
    const {
  for (MCSuperRegIterator AI(Reg, this, true); AI.isValid(); ++AI)
    RegisterSet.set(*AI);
}

bool TargetRegisterInfo::checkAllSuperRegsMarked(const BitVector &RegisterSet,
    ArrayRef<MCPhysReg> Exceptions) const {
  // Check that all super registers of reserved regs are reserved as well.
  BitVector Checked(getNumRegs());
  for (unsigned Reg : RegisterSet.set_bits()) {
    if (Checked[Reg])
      continue;
    for (MCSuperRegIterator SR(Reg, this); SR.isValid(); ++SR) {
      if (!RegisterSet[*SR] && !is_contained(Exceptions, Reg)) {
        dbgs() << "Error: Super register " << PrintReg(*SR, this)
               << " of reserved register " << PrintReg(Reg, this)
               << " is not reserved.\n";
        return false;
      }

      // We transitively check superregs. So we can remember this for later
      // to avoid compiletime explosion in deep register hierarchies.
      Checked.set(*SR);
    }
  }
  return true;
}

namespace llvm {

Printable PrintReg(unsigned Reg, const TargetRegisterInfo *TRI,
                   unsigned SubIdx) {
  return Printable([Reg, TRI, SubIdx](raw_ostream &OS) {
    if (!Reg)
      OS << "%noreg";
    else if (TargetRegisterInfo::isStackSlot(Reg))
      OS << "SS#" << TargetRegisterInfo::stackSlot2Index(Reg);
    else if (TargetRegisterInfo::isVirtualRegister(Reg))
      OS << "%vreg" << TargetRegisterInfo::virtReg2Index(Reg);
    else if (TRI && Reg < TRI->getNumRegs())
      OS << '%' << TRI->getName(Reg);
    else
      OS << "%physreg" << Reg;
    if (SubIdx) {
      if (TRI)
        OS << ':' << TRI->getSubRegIndexName(SubIdx);
      else
        OS << ":sub(" << SubIdx << ')';
    }
  });
}

Printable PrintRegUnit(unsigned Unit, const TargetRegisterInfo *TRI) {
  return Printable([Unit, TRI](raw_ostream &OS) {
    // Generic printout when TRI is missing.
    if (!TRI) {
      OS << "Unit~" << Unit;
      return;
    }

    // Check for invalid register units.
    if (Unit >= TRI->getNumRegUnits()) {
      OS << "BadUnit~" << Unit;
      return;
    }

    // Normal units have at least one root.
    MCRegUnitRootIterator Roots(Unit, TRI);
    assert(Roots.isValid() && "Unit has no roots.");
    OS << TRI->getName(*Roots);
    for (++Roots; Roots.isValid(); ++Roots)
      OS << '~' << TRI->getName(*Roots);
  });
}

Printable PrintVRegOrUnit(unsigned Unit, const TargetRegisterInfo *TRI) {
  return Printable([Unit, TRI](raw_ostream &OS) {
    if (TRI && TRI->isVirtualRegister(Unit)) {
      OS << "%vreg" << TargetRegisterInfo::virtReg2Index(Unit);
    } else {
      OS << PrintRegUnit(Unit, TRI);
    }
  });
}

} // end namespace llvm

/// getAllocatableClass - Return the maximal subclass of the given register
/// class that is alloctable, or NULL.
const TargetRegisterClass *
TargetRegisterInfo::getAllocatableClass(const TargetRegisterClass *RC) const {
  if (!RC || RC->isAllocatable())
    return RC;

  for (BitMaskClassIterator It(RC->getSubClassMask(), *this); It.isValid();
       ++It) {
    const TargetRegisterClass *SubRC = getRegClass(It.getID());
    if (SubRC->isAllocatable())
      return SubRC;
  }
  return nullptr;
}

/// getMinimalPhysRegClass - Returns the Register Class of a physical
/// register of the given type, picking the most sub register class of
/// the right type that contains this physreg.
const TargetRegisterClass *
TargetRegisterInfo::getMinimalPhysRegClass(unsigned reg, MVT VT) const {
  assert(isPhysicalRegister(reg) && "reg must be a physical register");

  // Pick the most sub register class of the right type that contains
  // this physreg.
  const TargetRegisterClass* BestRC = nullptr;
  for (const TargetRegisterClass* RC : regclasses()) {
    if ((VT == MVT::Other || isTypeLegalForClass(*RC, VT)) &&
        RC->contains(reg) && (!BestRC || BestRC->hasSubClass(RC)))
      BestRC = RC;
  }

  assert(BestRC && "Couldn't find the register class");
  return BestRC;
}

/// getAllocatableSetForRC - Toggle the bits that represent allocatable
/// registers for the specific register class.
static void getAllocatableSetForRC(const MachineFunction &MF,
                                   const TargetRegisterClass *RC, BitVector &R){
  assert(RC->isAllocatable() && "invalid for nonallocatable sets");
  ArrayRef<MCPhysReg> Order = RC->getRawAllocationOrder(MF);
  for (unsigned i = 0; i != Order.size(); ++i)
    R.set(Order[i]);
}

BitVector TargetRegisterInfo::getAllocatableSet(const MachineFunction &MF,
                                          const TargetRegisterClass *RC) const {
  BitVector Allocatable(getNumRegs());
  if (RC) {
    // A register class with no allocatable subclass returns an empty set.
    const TargetRegisterClass *SubClass = getAllocatableClass(RC);
    if (SubClass)
      getAllocatableSetForRC(MF, SubClass, Allocatable);
  } else {
    for (const TargetRegisterClass *C : regclasses())
      if (C->isAllocatable())
        getAllocatableSetForRC(MF, C, Allocatable);
  }

  // Mask out the reserved registers
  BitVector Reserved = getReservedRegs(MF);
  Allocatable &= Reserved.flip();

  return Allocatable;
}

static inline
const TargetRegisterClass *firstCommonClass(const uint32_t *A,
                                            const uint32_t *B,
                                            const TargetRegisterInfo *TRI,
                                            const MVT::SimpleValueType SVT =
                                            MVT::SimpleValueType::Any) {
  const MVT VT(SVT);
  for (unsigned I = 0, E = TRI->getNumRegClasses(); I < E; I += 32)
    if (unsigned Common = *A++ & *B++) {
      const TargetRegisterClass *RC =
          TRI->getRegClass(I + countTrailingZeros(Common));
      if (SVT == MVT::SimpleValueType::Any || TRI->isTypeLegalForClass(*RC, VT))
        return RC;
    }
  return nullptr;
}

const TargetRegisterClass *
TargetRegisterInfo::getCommonSubClass(const TargetRegisterClass *A,
                                      const TargetRegisterClass *B,
                                      const MVT::SimpleValueType SVT) const {
  // First take care of the trivial cases.
  if (A == B)
    return A;
  if (!A || !B)
    return nullptr;

  // Register classes are ordered topologically, so the largest common
  // sub-class it the common sub-class with the smallest ID.
  return firstCommonClass(A->getSubClassMask(), B->getSubClassMask(), this, SVT);
}

const TargetRegisterClass *
TargetRegisterInfo::getMatchingSuperRegClass(const TargetRegisterClass *A,
                                             const TargetRegisterClass *B,
                                             unsigned Idx) const {
  assert(A && B && "Missing register class");
  assert(Idx && "Bad sub-register index");

  // Find Idx in the list of super-register indices.
  for (SuperRegClassIterator RCI(B, this); RCI.isValid(); ++RCI)
    if (RCI.getSubReg() == Idx)
      // The bit mask contains all register classes that are projected into B
      // by Idx. Find a class that is also a sub-class of A.
      return firstCommonClass(RCI.getMask(), A->getSubClassMask(), this);
  return nullptr;
}

const TargetRegisterClass *TargetRegisterInfo::
getCommonSuperRegClass(const TargetRegisterClass *RCA, unsigned SubA,
                       const TargetRegisterClass *RCB, unsigned SubB,
                       unsigned &PreA, unsigned &PreB) const {
  assert(RCA && SubA && RCB && SubB && "Invalid arguments");

  // Search all pairs of sub-register indices that project into RCA and RCB
  // respectively. This is quadratic, but usually the sets are very small. On
  // most targets like X86, there will only be a single sub-register index
  // (e.g., sub_16bit projecting into GR16).
  //
  // The worst case is a register class like DPR on ARM.
  // We have indices dsub_0..dsub_7 projecting into that class.
  //
  // It is very common that one register class is a sub-register of the other.
  // Arrange for RCA to be the larger register so the answer will be found in
  // the first iteration. This makes the search linear for the most common
  // case.
  const TargetRegisterClass *BestRC = nullptr;
  unsigned *BestPreA = &PreA;
  unsigned *BestPreB = &PreB;
  if (getRegSizeInBits(*RCA) < getRegSizeInBits(*RCB)) {
    std::swap(RCA, RCB);
    std::swap(SubA, SubB);
    std::swap(BestPreA, BestPreB);
  }

  // Also terminate the search one we have found a register class as small as
  // RCA.
  unsigned MinSize = getRegSizeInBits(*RCA);

  for (SuperRegClassIterator IA(RCA, this, true); IA.isValid(); ++IA) {
    unsigned FinalA = composeSubRegIndices(IA.getSubReg(), SubA);
    for (SuperRegClassIterator IB(RCB, this, true); IB.isValid(); ++IB) {
      // Check if a common super-register class exists for this index pair.
      const TargetRegisterClass *RC =
        firstCommonClass(IA.getMask(), IB.getMask(), this);
      if (!RC || getRegSizeInBits(*RC) < MinSize)
        continue;

      // The indexes must compose identically: PreA+SubA == PreB+SubB.
      unsigned FinalB = composeSubRegIndices(IB.getSubReg(), SubB);
      if (FinalA != FinalB)
        continue;

      // Is RC a better candidate than BestRC?
      if (BestRC && getRegSizeInBits(*RC) >= getRegSizeInBits(*BestRC))
        continue;

      // Yes, RC is the smallest super-register seen so far.
      BestRC = RC;
      *BestPreA = IA.getSubReg();
      *BestPreB = IB.getSubReg();

      // Bail early if we reached MinSize. We won't find a better candidate.
      if (getRegSizeInBits(*BestRC) == MinSize)
        return BestRC;
    }
  }
  return BestRC;
}

/// \brief Check if the registers defined by the pair (RegisterClass, SubReg)
/// share the same register file.
static bool shareSameRegisterFile(const TargetRegisterInfo &TRI,
                                  const TargetRegisterClass *DefRC,
                                  unsigned DefSubReg,
                                  const TargetRegisterClass *SrcRC,
                                  unsigned SrcSubReg) {
  // Same register class.
  if (DefRC == SrcRC)
    return true;

  // Both operands are sub registers. Check if they share a register class.
  unsigned SrcIdx, DefIdx;
  if (SrcSubReg && DefSubReg) {
    return TRI.getCommonSuperRegClass(SrcRC, SrcSubReg, DefRC, DefSubReg,
                                      SrcIdx, DefIdx) != nullptr;
  }

  // At most one of the register is a sub register, make it Src to avoid
  // duplicating the test.
  if (!SrcSubReg) {
    std::swap(DefSubReg, SrcSubReg);
    std::swap(DefRC, SrcRC);
  }

  // One of the register is a sub register, check if we can get a superclass.
  if (SrcSubReg)
    return TRI.getMatchingSuperRegClass(SrcRC, DefRC, SrcSubReg) != nullptr;

  // Plain copy.
  return TRI.getCommonSubClass(DefRC, SrcRC) != nullptr;
}

bool TargetRegisterInfo::shouldRewriteCopySrc(const TargetRegisterClass *DefRC,
                                              unsigned DefSubReg,
                                              const TargetRegisterClass *SrcRC,
                                              unsigned SrcSubReg) const {
  // If this source does not incur a cross register bank copy, use it.
  return shareSameRegisterFile(*this, DefRC, DefSubReg, SrcRC, SrcSubReg);
}

// Compute target-independent register allocator hints to help eliminate copies.
void
TargetRegisterInfo::getRegAllocationHints(unsigned VirtReg,
                                          ArrayRef<MCPhysReg> Order,
                                          SmallVectorImpl<MCPhysReg> &Hints,
                                          const MachineFunction &MF,
                                          const VirtRegMap *VRM,
                                          const LiveRegMatrix *Matrix) const {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  std::pair<unsigned, unsigned> Hint = MRI.getRegAllocationHint(VirtReg);

  // Hints with HintType != 0 were set by target-dependent code.
  // Such targets must provide their own implementation of
  // TRI::getRegAllocationHints to interpret those hint types.
  assert(Hint.first == 0 && "Target must implement TRI::getRegAllocationHints");

  // Target-independent hints are either a physical or a virtual register.
  unsigned Phys = Hint.second;
  if (VRM && isVirtualRegister(Phys))
    Phys = VRM->getPhys(Phys);

  // Check that Phys is a valid hint in VirtReg's register class.
  if (!isPhysicalRegister(Phys))
    return;
  if (MRI.isReserved(Phys))
    return;
  // Check that Phys is in the allocation order. We shouldn't heed hints
  // from VirtReg's register class if they aren't in the allocation order. The
  // target probably has a reason for removing the register.
  if (!is_contained(Order, Phys))
    return;

  // All clear, tell the register allocator to prefer this register.
  Hints.push_back(Phys);
}

bool TargetRegisterInfo::canRealignStack(const MachineFunction &MF) const {
  return !MF.getFunction()->hasFnAttribute("no-realign-stack");
}

bool TargetRegisterInfo::needsStackRealignment(
    const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const Function *F = MF.getFunction();
  unsigned StackAlign = TFI->getStackAlignment();
  bool requiresRealignment = ((MFI.getMaxAlignment() > StackAlign) ||
                              F->hasFnAttribute(Attribute::StackAlignment));
  if (MF.getFunction()->hasFnAttribute("stackrealign") || requiresRealignment) {
    if (canRealignStack(MF))
      return true;
    DEBUG(dbgs() << "Can't realign function's stack: " << F->getName() << "\n");
  }
  return false;
}

bool TargetRegisterInfo::regmaskSubsetEqual(const uint32_t *mask0,
                                            const uint32_t *mask1) const {
  unsigned N = (getNumRegs()+31) / 32;
  for (unsigned I = 0; I < N; ++I)
    if ((mask0[I] & mask1[I]) != mask0[I])
      return false;
  return true;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
void TargetRegisterInfo::dumpReg(unsigned Reg, unsigned SubRegIndex,
                                 const TargetRegisterInfo *TRI) {
  dbgs() << PrintReg(Reg, TRI, SubRegIndex) << "\n";
}
#endif
