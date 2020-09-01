//===- MachineCopyPropagation.cpp - Machine Copy Propagation Pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is an extremely simple MachineInstr-level copy propagation pass.
//
// This pass forwards the source of COPYs to the users of their destinations
// when doing so is legal.  For example:
//
//   %reg1 = COPY %reg0
//   ...
//   ... = OP %reg1
//
// If
//   - %reg0 has not been clobbered by the time of the use of %reg1
//   - the register class constraints are satisfied
//   - the COPY def is the only value that reaches OP
// then this pass replaces the above with:
//
//   %reg1 = COPY %reg0
//   ...
//   ... = OP %reg0
//
// This pass also removes some redundant COPYs.  For example:
//
//    %R1 = COPY %R0
//    ... // No clobber of %R1
//    %R0 = COPY %R1 <<< Removed
//
// or
//
//    %R1 = COPY %R0
//    ... // No clobber of %R0
//    %R1 = COPY %R0 <<< Removed
//
// or
//
//    $R0 = OP ...
//    ... // No read/clobber of $R0 and $R1
//    $R1 = COPY $R0 // $R0 is killed
// Replace $R0 with $R1 and remove the COPY
//    $R1 = OP ...
//    ...
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <iterator>

using namespace llvm;

#define DEBUG_TYPE "machine-cp"

STATISTIC(NumDeletes, "Number of dead copies deleted");
STATISTIC(NumCopyForwards, "Number of copy uses forwarded");
STATISTIC(NumCopyBackwardPropagated, "Number of copy defs backward propagated");
DEBUG_COUNTER(FwdCounter, "machine-cp-fwd",
              "Controls which register COPYs are forwarded");

namespace {

class CopyTracker {
  struct CopyInfo {
    MachineInstr *MI;
    SmallVector<unsigned, 4> DefRegs;
    bool Avail;
  };

  DenseMap<unsigned, CopyInfo> Copies;

public:
  /// Mark all of the given registers and their subregisters as unavailable for
  /// copying.
  void markRegsUnavailable(ArrayRef<unsigned> Regs,
                           const TargetRegisterInfo &TRI) {
    for (unsigned Reg : Regs) {
      // Source of copy is no longer available for propagation.
      for (MCRegUnitIterator RUI(Reg, &TRI); RUI.isValid(); ++RUI) {
        auto CI = Copies.find(*RUI);
        if (CI != Copies.end())
          CI->second.Avail = false;
      }
    }
  }

  /// Remove register from copy maps.
  void invalidateRegister(unsigned Reg, const TargetRegisterInfo &TRI) {
    // Since Reg might be a subreg of some registers, only invalidate Reg is not
    // enough. We have to find the COPY defines Reg or registers defined by Reg
    // and invalidate all of them.
    SmallSet<unsigned, 8> RegsToInvalidate;
    RegsToInvalidate.insert(Reg);
    for (MCRegUnitIterator RUI(Reg, &TRI); RUI.isValid(); ++RUI) {
      auto I = Copies.find(*RUI);
      if (I != Copies.end()) {
        if (MachineInstr *MI = I->second.MI) {
          RegsToInvalidate.insert(MI->getOperand(0).getReg());
          RegsToInvalidate.insert(MI->getOperand(1).getReg());
        }
        RegsToInvalidate.insert(I->second.DefRegs.begin(),
                                I->second.DefRegs.end());
      }
    }
    for (unsigned InvalidReg : RegsToInvalidate)
      for (MCRegUnitIterator RUI(InvalidReg, &TRI); RUI.isValid(); ++RUI)
        Copies.erase(*RUI);
  }

  /// Clobber a single register, removing it from the tracker's copy maps.
  void clobberRegister(unsigned Reg, const TargetRegisterInfo &TRI) {
    for (MCRegUnitIterator RUI(Reg, &TRI); RUI.isValid(); ++RUI) {
      auto I = Copies.find(*RUI);
      if (I != Copies.end()) {
        // When we clobber the source of a copy, we need to clobber everything
        // it defined.
        markRegsUnavailable(I->second.DefRegs, TRI);
        // When we clobber the destination of a copy, we need to clobber the
        // whole register it defined.
        if (MachineInstr *MI = I->second.MI)
          markRegsUnavailable({MI->getOperand(0).getReg()}, TRI);
        // Now we can erase the copy.
        Copies.erase(I);
      }
    }
  }

  /// Add this copy's registers into the tracker's copy maps.
  void trackCopy(MachineInstr *MI, const TargetRegisterInfo &TRI) {
    assert(MI->isCopy() && "Tracking non-copy?");

    Register Def = MI->getOperand(0).getReg();
    Register Src = MI->getOperand(1).getReg();

    // Remember Def is defined by the copy.
    for (MCRegUnitIterator RUI(Def, &TRI); RUI.isValid(); ++RUI)
      Copies[*RUI] = {MI, {}, true};

    // Remember source that's copied to Def. Once it's clobbered, then
    // it's no longer available for copy propagation.
    for (MCRegUnitIterator RUI(Src, &TRI); RUI.isValid(); ++RUI) {
      auto I = Copies.insert({*RUI, {nullptr, {}, false}});
      auto &Copy = I.first->second;
      if (!is_contained(Copy.DefRegs, Def))
        Copy.DefRegs.push_back(Def);
    }
  }

  bool hasAnyCopies() {
    return !Copies.empty();
  }

  MachineInstr *findCopyForUnit(unsigned RegUnit, const TargetRegisterInfo &TRI,
                         bool MustBeAvailable = false) {
    auto CI = Copies.find(RegUnit);
    if (CI == Copies.end())
      return nullptr;
    if (MustBeAvailable && !CI->second.Avail)
      return nullptr;
    return CI->second.MI;
  }

  MachineInstr *findCopyDefViaUnit(unsigned RegUnit,
                                    const TargetRegisterInfo &TRI) {
    auto CI = Copies.find(RegUnit);
    if (CI == Copies.end())
      return nullptr;
    if (CI->second.DefRegs.size() != 1)
      return nullptr;
    MCRegUnitIterator RUI(CI->second.DefRegs[0], &TRI);
    return findCopyForUnit(*RUI, TRI, true);
  }

  MachineInstr *findAvailBackwardCopy(MachineInstr &I, unsigned Reg,
                                      const TargetRegisterInfo &TRI) {
    MCRegUnitIterator RUI(Reg, &TRI);
    MachineInstr *AvailCopy = findCopyDefViaUnit(*RUI, TRI);
    if (!AvailCopy ||
        !TRI.isSubRegisterEq(AvailCopy->getOperand(1).getReg(), Reg))
      return nullptr;

    Register AvailSrc = AvailCopy->getOperand(1).getReg();
    Register AvailDef = AvailCopy->getOperand(0).getReg();
    for (const MachineInstr &MI :
         make_range(AvailCopy->getReverseIterator(), I.getReverseIterator()))
      for (const MachineOperand &MO : MI.operands())
        if (MO.isRegMask())
          // FIXME: Shall we simultaneously invalidate AvailSrc or AvailDef?
          if (MO.clobbersPhysReg(AvailSrc) || MO.clobbersPhysReg(AvailDef))
            return nullptr;

    return AvailCopy;
  }

  MachineInstr *findAvailCopy(MachineInstr &DestCopy, unsigned Reg,
                              const TargetRegisterInfo &TRI) {
    // We check the first RegUnit here, since we'll only be interested in the
    // copy if it copies the entire register anyway.
    MCRegUnitIterator RUI(Reg, &TRI);
    MachineInstr *AvailCopy =
        findCopyForUnit(*RUI, TRI, /*MustBeAvailable=*/true);
    if (!AvailCopy ||
        !TRI.isSubRegisterEq(AvailCopy->getOperand(0).getReg(), Reg))
      return nullptr;

    // Check that the available copy isn't clobbered by any regmasks between
    // itself and the destination.
    Register AvailSrc = AvailCopy->getOperand(1).getReg();
    Register AvailDef = AvailCopy->getOperand(0).getReg();
    for (const MachineInstr &MI :
         make_range(AvailCopy->getIterator(), DestCopy.getIterator()))
      for (const MachineOperand &MO : MI.operands())
        if (MO.isRegMask())
          if (MO.clobbersPhysReg(AvailSrc) || MO.clobbersPhysReg(AvailDef))
            return nullptr;

    return AvailCopy;
  }

  void clear() {
    Copies.clear();
  }
};

class MachineCopyPropagation : public MachineFunctionPass {
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  const MachineRegisterInfo *MRI;

public:
  static char ID; // Pass identification, replacement for typeid

  MachineCopyPropagation() : MachineFunctionPass(ID) {
    initializeMachineCopyPropagationPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  typedef enum { DebugUse = false, RegularUse = true } DebugType;

  void ClobberRegister(unsigned Reg);
  void ReadRegister(unsigned Reg, MachineInstr &Reader,
                    DebugType DT);
  void ForwardCopyPropagateBlock(MachineBasicBlock &MBB);
  void BackwardCopyPropagateBlock(MachineBasicBlock &MBB);
  bool eraseIfRedundant(MachineInstr &Copy, unsigned Src, unsigned Def);
  void forwardUses(MachineInstr &MI);
  void propagateDefs(MachineInstr &MI);
  bool isForwardableRegClassCopy(const MachineInstr &Copy,
                                 const MachineInstr &UseI, unsigned UseIdx);
  bool isBackwardPropagatableRegClassCopy(const MachineInstr &Copy,
                                          const MachineInstr &UseI,
                                          unsigned UseIdx);
  bool hasImplicitOverlap(const MachineInstr &MI, const MachineOperand &Use);

  /// Candidates for deletion.
  SmallSetVector<MachineInstr *, 8> MaybeDeadCopies;

  /// Multimap tracking debug users in current BB
  DenseMap<MachineInstr*, SmallVector<MachineInstr*, 2>> CopyDbgUsers;

  CopyTracker Tracker;

  bool Changed;
};

} // end anonymous namespace

char MachineCopyPropagation::ID = 0;

char &llvm::MachineCopyPropagationID = MachineCopyPropagation::ID;

INITIALIZE_PASS(MachineCopyPropagation, DEBUG_TYPE,
                "Machine Copy Propagation Pass", false, false)

void MachineCopyPropagation::ReadRegister(unsigned Reg, MachineInstr &Reader,
                                          DebugType DT) {
  // If 'Reg' is defined by a copy, the copy is no longer a candidate
  // for elimination. If a copy is "read" by a debug user, record the user
  // for propagation.
  for (MCRegUnitIterator RUI(Reg, TRI); RUI.isValid(); ++RUI) {
    if (MachineInstr *Copy = Tracker.findCopyForUnit(*RUI, *TRI)) {
      if (DT == RegularUse) {
        LLVM_DEBUG(dbgs() << "MCP: Copy is used - not dead: "; Copy->dump());
        MaybeDeadCopies.remove(Copy);
      } else {
        CopyDbgUsers[Copy].push_back(&Reader);
      }
    }
  }
}

/// Return true if \p PreviousCopy did copy register \p Src to register \p Def.
/// This fact may have been obscured by sub register usage or may not be true at
/// all even though Src and Def are subregisters of the registers used in
/// PreviousCopy. e.g.
/// isNopCopy("ecx = COPY eax", AX, CX) == true
/// isNopCopy("ecx = COPY eax", AH, CL) == false
static bool isNopCopy(const MachineInstr &PreviousCopy, unsigned Src,
                      unsigned Def, const TargetRegisterInfo *TRI) {
  Register PreviousSrc = PreviousCopy.getOperand(1).getReg();
  Register PreviousDef = PreviousCopy.getOperand(0).getReg();
  if (Src == PreviousSrc && Def == PreviousDef)
    return true;
  if (!TRI->isSubRegister(PreviousSrc, Src))
    return false;
  unsigned SubIdx = TRI->getSubRegIndex(PreviousSrc, Src);
  return SubIdx == TRI->getSubRegIndex(PreviousDef, Def);
}

/// Remove instruction \p Copy if there exists a previous copy that copies the
/// register \p Src to the register \p Def; This may happen indirectly by
/// copying the super registers.
bool MachineCopyPropagation::eraseIfRedundant(MachineInstr &Copy, unsigned Src,
                                              unsigned Def) {
  // Avoid eliminating a copy from/to a reserved registers as we cannot predict
  // the value (Example: The sparc zero register is writable but stays zero).
  if (MRI->isReserved(Src) || MRI->isReserved(Def))
    return false;

  // Search for an existing copy.
  MachineInstr *PrevCopy = Tracker.findAvailCopy(Copy, Def, *TRI);
  if (!PrevCopy)
    return false;

  // Check that the existing copy uses the correct sub registers.
  if (PrevCopy->getOperand(0).isDead())
    return false;
  if (!isNopCopy(*PrevCopy, Src, Def, TRI))
    return false;

  LLVM_DEBUG(dbgs() << "MCP: copy is a NOP, removing: "; Copy.dump());

  // Copy was redundantly redefining either Src or Def. Remove earlier kill
  // flags between Copy and PrevCopy because the value will be reused now.
  assert(Copy.isCopy());
  Register CopyDef = Copy.getOperand(0).getReg();
  assert(CopyDef == Src || CopyDef == Def);
  for (MachineInstr &MI :
       make_range(PrevCopy->getIterator(), Copy.getIterator()))
    MI.clearRegisterKills(CopyDef, TRI);

  Copy.eraseFromParent();
  Changed = true;
  ++NumDeletes;
  return true;
}

bool MachineCopyPropagation::isBackwardPropagatableRegClassCopy(
    const MachineInstr &Copy, const MachineInstr &UseI, unsigned UseIdx) {
  Register Def = Copy.getOperand(0).getReg();

  if (const TargetRegisterClass *URC =
          UseI.getRegClassConstraint(UseIdx, TII, TRI))
    return URC->contains(Def);

  // We don't process further if UseI is a COPY, since forward copy propagation
  // should handle that.
  return false;
}

/// Decide whether we should forward the source of \param Copy to its use in
/// \param UseI based on the physical register class constraints of the opcode
/// and avoiding introducing more cross-class COPYs.
bool MachineCopyPropagation::isForwardableRegClassCopy(const MachineInstr &Copy,
                                                       const MachineInstr &UseI,
                                                       unsigned UseIdx) {

  Register CopySrcReg = Copy.getOperand(1).getReg();

  // If the new register meets the opcode register constraints, then allow
  // forwarding.
  if (const TargetRegisterClass *URC =
          UseI.getRegClassConstraint(UseIdx, TII, TRI))
    return URC->contains(CopySrcReg);

  if (!UseI.isCopy())
    return false;

  /// COPYs don't have register class constraints, so if the user instruction
  /// is a COPY, we just try to avoid introducing additional cross-class
  /// COPYs.  For example:
  ///
  ///   RegClassA = COPY RegClassB  // Copy parameter
  ///   ...
  ///   RegClassB = COPY RegClassA  // UseI parameter
  ///
  /// which after forwarding becomes
  ///
  ///   RegClassA = COPY RegClassB
  ///   ...
  ///   RegClassB = COPY RegClassB
  ///
  /// so we have reduced the number of cross-class COPYs and potentially
  /// introduced a nop COPY that can be removed.
  const TargetRegisterClass *UseDstRC =
      TRI->getMinimalPhysRegClass(UseI.getOperand(0).getReg());

  const TargetRegisterClass *SuperRC = UseDstRC;
  for (TargetRegisterClass::sc_iterator SuperRCI = UseDstRC->getSuperClasses();
       SuperRC; SuperRC = *SuperRCI++)
    if (SuperRC->contains(CopySrcReg))
      return true;

  return false;
}

/// Check that \p MI does not have implicit uses that overlap with it's \p Use
/// operand (the register being replaced), since these can sometimes be
/// implicitly tied to other operands.  For example, on AMDGPU:
///
/// V_MOVRELS_B32_e32 %VGPR2, %M0<imp-use>, %EXEC<imp-use>, %VGPR2_VGPR3_VGPR4_VGPR5<imp-use>
///
/// the %VGPR2 is implicitly tied to the larger reg operand, but we have no
/// way of knowing we need to update the latter when updating the former.
bool MachineCopyPropagation::hasImplicitOverlap(const MachineInstr &MI,
                                                const MachineOperand &Use) {
  for (const MachineOperand &MIUse : MI.uses())
    if (&MIUse != &Use && MIUse.isReg() && MIUse.isImplicit() &&
        MIUse.isUse() && TRI->regsOverlap(Use.getReg(), MIUse.getReg()))
      return true;

  return false;
}

/// Look for available copies whose destination register is used by \p MI and
/// replace the use in \p MI with the copy's source register.
void MachineCopyPropagation::forwardUses(MachineInstr &MI) {
  if (!Tracker.hasAnyCopies())
    return;

  // Look for non-tied explicit vreg uses that have an active COPY
  // instruction that defines the physical register allocated to them.
  // Replace the vreg with the source of the active COPY.
  for (unsigned OpIdx = 0, OpEnd = MI.getNumOperands(); OpIdx < OpEnd;
       ++OpIdx) {
    MachineOperand &MOUse = MI.getOperand(OpIdx);
    // Don't forward into undef use operands since doing so can cause problems
    // with the machine verifier, since it doesn't treat undef reads as reads,
    // so we can end up with a live range that ends on an undef read, leading to
    // an error that the live range doesn't end on a read of the live range
    // register.
    if (!MOUse.isReg() || MOUse.isTied() || MOUse.isUndef() || MOUse.isDef() ||
        MOUse.isImplicit())
      continue;

    if (!MOUse.getReg())
      continue;

    // Check that the register is marked 'renamable' so we know it is safe to
    // rename it without violating any constraints that aren't expressed in the
    // IR (e.g. ABI or opcode requirements).
    if (!MOUse.isRenamable())
      continue;

    MachineInstr *Copy = Tracker.findAvailCopy(MI, MOUse.getReg(), *TRI);
    if (!Copy)
      continue;

    Register CopyDstReg = Copy->getOperand(0).getReg();
    const MachineOperand &CopySrc = Copy->getOperand(1);
    Register CopySrcReg = CopySrc.getReg();

    // FIXME: Don't handle partial uses of wider COPYs yet.
    if (MOUse.getReg() != CopyDstReg) {
      LLVM_DEBUG(
          dbgs() << "MCP: FIXME! Not forwarding COPY to sub-register use:\n  "
                 << MI);
      continue;
    }

    // Don't forward COPYs of reserved regs unless they are constant.
    if (MRI->isReserved(CopySrcReg) && !MRI->isConstantPhysReg(CopySrcReg))
      continue;

    if (!isForwardableRegClassCopy(*Copy, MI, OpIdx))
      continue;

    if (hasImplicitOverlap(MI, MOUse))
      continue;

    // Check that the instruction is not a copy that partially overwrites the
    // original copy source that we are about to use. The tracker mechanism
    // cannot cope with that.
    if (MI.isCopy() && MI.modifiesRegister(CopySrcReg, TRI) &&
        !MI.definesRegister(CopySrcReg)) {
      LLVM_DEBUG(dbgs() << "MCP: Copy source overlap with dest in " << MI);
      continue;
    }

    if (!DebugCounter::shouldExecute(FwdCounter)) {
      LLVM_DEBUG(dbgs() << "MCP: Skipping forwarding due to debug counter:\n  "
                        << MI);
      continue;
    }

    LLVM_DEBUG(dbgs() << "MCP: Replacing " << printReg(MOUse.getReg(), TRI)
                      << "\n     with " << printReg(CopySrcReg, TRI)
                      << "\n     in " << MI << "     from " << *Copy);

    MOUse.setReg(CopySrcReg);
    if (!CopySrc.isRenamable())
      MOUse.setIsRenamable(false);

    LLVM_DEBUG(dbgs() << "MCP: After replacement: " << MI << "\n");

    // Clear kill markers that may have been invalidated.
    for (MachineInstr &KMI :
         make_range(Copy->getIterator(), std::next(MI.getIterator())))
      KMI.clearRegisterKills(CopySrcReg, TRI);

    ++NumCopyForwards;
    Changed = true;
  }
}

void MachineCopyPropagation::ForwardCopyPropagateBlock(MachineBasicBlock &MBB) {
  LLVM_DEBUG(dbgs() << "MCP: ForwardCopyPropagateBlock " << MBB.getName()
                    << "\n");

  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ) {
    MachineInstr *MI = &*I;
    ++I;

    // Analyze copies (which don't overlap themselves).
    if (MI->isCopy() && !TRI->regsOverlap(MI->getOperand(0).getReg(),
                                          MI->getOperand(1).getReg())) {
      Register Def = MI->getOperand(0).getReg();
      Register Src = MI->getOperand(1).getReg();

      assert(!Register::isVirtualRegister(Def) &&
             !Register::isVirtualRegister(Src) &&
             "MachineCopyPropagation should be run after register allocation!");

      // The two copies cancel out and the source of the first copy
      // hasn't been overridden, eliminate the second one. e.g.
      //  %ecx = COPY %eax
      //  ... nothing clobbered eax.
      //  %eax = COPY %ecx
      // =>
      //  %ecx = COPY %eax
      //
      // or
      //
      //  %ecx = COPY %eax
      //  ... nothing clobbered eax.
      //  %ecx = COPY %eax
      // =>
      //  %ecx = COPY %eax
      if (eraseIfRedundant(*MI, Def, Src) || eraseIfRedundant(*MI, Src, Def))
        continue;

      forwardUses(*MI);

      // Src may have been changed by forwardUses()
      Src = MI->getOperand(1).getReg();

      // If Src is defined by a previous copy, the previous copy cannot be
      // eliminated.
      ReadRegister(Src, *MI, RegularUse);
      for (const MachineOperand &MO : MI->implicit_operands()) {
        if (!MO.isReg() || !MO.readsReg())
          continue;
        Register Reg = MO.getReg();
        if (!Reg)
          continue;
        ReadRegister(Reg, *MI, RegularUse);
      }

      LLVM_DEBUG(dbgs() << "MCP: Copy is a deletion candidate: "; MI->dump());

      // Copy is now a candidate for deletion.
      if (!MRI->isReserved(Def))
        MaybeDeadCopies.insert(MI);

      // If 'Def' is previously source of another copy, then this earlier copy's
      // source is no longer available. e.g.
      // %xmm9 = copy %xmm2
      // ...
      // %xmm2 = copy %xmm0
      // ...
      // %xmm2 = copy %xmm9
      Tracker.clobberRegister(Def, *TRI);
      for (const MachineOperand &MO : MI->implicit_operands()) {
        if (!MO.isReg() || !MO.isDef())
          continue;
        Register Reg = MO.getReg();
        if (!Reg)
          continue;
        Tracker.clobberRegister(Reg, *TRI);
      }

      Tracker.trackCopy(MI, *TRI);

      continue;
    }

    // Clobber any earlyclobber regs first.
    for (const MachineOperand &MO : MI->operands())
      if (MO.isReg() && MO.isEarlyClobber()) {
        Register Reg = MO.getReg();
        // If we have a tied earlyclobber, that means it is also read by this
        // instruction, so we need to make sure we don't remove it as dead
        // later.
        if (MO.isTied())
          ReadRegister(Reg, *MI, RegularUse);
        Tracker.clobberRegister(Reg, *TRI);
      }

    forwardUses(*MI);

    // Not a copy.
    SmallVector<unsigned, 2> Defs;
    const MachineOperand *RegMask = nullptr;
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isRegMask())
        RegMask = &MO;
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      if (!Reg)
        continue;

      assert(!Register::isVirtualRegister(Reg) &&
             "MachineCopyPropagation should be run after register allocation!");

      if (MO.isDef() && !MO.isEarlyClobber()) {
        Defs.push_back(Reg);
        continue;
      } else if (MO.readsReg())
        ReadRegister(Reg, *MI, MO.isDebug() ? DebugUse : RegularUse);
    }

    // The instruction has a register mask operand which means that it clobbers
    // a large set of registers.  Treat clobbered registers the same way as
    // defined registers.
    if (RegMask) {
      // Erase any MaybeDeadCopies whose destination register is clobbered.
      for (SmallSetVector<MachineInstr *, 8>::iterator DI =
               MaybeDeadCopies.begin();
           DI != MaybeDeadCopies.end();) {
        MachineInstr *MaybeDead = *DI;
        Register Reg = MaybeDead->getOperand(0).getReg();
        assert(!MRI->isReserved(Reg));

        if (!RegMask->clobbersPhysReg(Reg)) {
          ++DI;
          continue;
        }

        LLVM_DEBUG(dbgs() << "MCP: Removing copy due to regmask clobbering: ";
                   MaybeDead->dump());

        // Make sure we invalidate any entries in the copy maps before erasing
        // the instruction.
        Tracker.clobberRegister(Reg, *TRI);

        // erase() will return the next valid iterator pointing to the next
        // element after the erased one.
        DI = MaybeDeadCopies.erase(DI);
        MaybeDead->eraseFromParent();
        Changed = true;
        ++NumDeletes;
      }
    }

    // Any previous copy definition or reading the Defs is no longer available.
    for (unsigned Reg : Defs)
      Tracker.clobberRegister(Reg, *TRI);
  }

  // If MBB doesn't have successors, delete the copies whose defs are not used.
  // If MBB does have successors, then conservative assume the defs are live-out
  // since we don't want to trust live-in lists.
  if (MBB.succ_empty()) {
    for (MachineInstr *MaybeDead : MaybeDeadCopies) {
      LLVM_DEBUG(dbgs() << "MCP: Removing copy due to no live-out succ: ";
                 MaybeDead->dump());
      assert(!MRI->isReserved(MaybeDead->getOperand(0).getReg()));

      // Update matching debug values, if any.
      assert(MaybeDead->isCopy());
      unsigned SrcReg = MaybeDead->getOperand(1).getReg();
      MRI->updateDbgUsersToReg(SrcReg, CopyDbgUsers[MaybeDead]);

      MaybeDead->eraseFromParent();
      Changed = true;
      ++NumDeletes;
    }
  }

  MaybeDeadCopies.clear();
  CopyDbgUsers.clear();
  Tracker.clear();
}

static bool isBackwardPropagatableCopy(MachineInstr &MI,
                                       const MachineRegisterInfo &MRI) {
  assert(MI.isCopy() && "MI is expected to be a COPY");
  Register Def = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();

  if (!Def || !Src)
    return false;

  if (MRI.isReserved(Def) || MRI.isReserved(Src))
    return false;

  return MI.getOperand(1).isRenamable() && MI.getOperand(1).isKill();
}

void MachineCopyPropagation::propagateDefs(MachineInstr &MI) {
  if (!Tracker.hasAnyCopies())
    return;

  for (unsigned OpIdx = 0, OpEnd = MI.getNumOperands(); OpIdx != OpEnd;
       ++OpIdx) {
    MachineOperand &MODef = MI.getOperand(OpIdx);

    if (!MODef.isReg() || MODef.isUse())
      continue;

    // Ignore non-trivial cases.
    if (MODef.isTied() || MODef.isUndef() || MODef.isImplicit())
      continue;

    if (!MODef.getReg())
      continue;

    // We only handle if the register comes from a vreg.
    if (!MODef.isRenamable())
      continue;

    MachineInstr *Copy =
        Tracker.findAvailBackwardCopy(MI, MODef.getReg(), *TRI);
    if (!Copy)
      continue;

    Register Def = Copy->getOperand(0).getReg();
    Register Src = Copy->getOperand(1).getReg();

    if (MODef.getReg() != Src)
      continue;

    if (!isBackwardPropagatableRegClassCopy(*Copy, MI, OpIdx))
      continue;

    if (hasImplicitOverlap(MI, MODef))
      continue;

    LLVM_DEBUG(dbgs() << "MCP: Replacing " << printReg(MODef.getReg(), TRI)
                      << "\n     with " << printReg(Def, TRI) << "\n     in "
                      << MI << "     from " << *Copy);

    MODef.setReg(Def);
    MODef.setIsRenamable(Copy->getOperand(0).isRenamable());

    LLVM_DEBUG(dbgs() << "MCP: After replacement: " << MI << "\n");
    MaybeDeadCopies.insert(Copy);
    Changed = true;
    ++NumCopyBackwardPropagated;
  }
}

void MachineCopyPropagation::BackwardCopyPropagateBlock(
    MachineBasicBlock &MBB) {
  LLVM_DEBUG(dbgs() << "MCP: BackwardCopyPropagateBlock " << MBB.getName()
                    << "\n");

  for (MachineBasicBlock::reverse_iterator I = MBB.rbegin(), E = MBB.rend();
       I != E;) {
    MachineInstr *MI = &*I;
    ++I;

    // Ignore non-trivial COPYs.
    if (MI->isCopy() && MI->getNumOperands() == 2 &&
        !TRI->regsOverlap(MI->getOperand(0).getReg(),
                          MI->getOperand(1).getReg())) {

      Register Def = MI->getOperand(0).getReg();
      Register Src = MI->getOperand(1).getReg();

      // Unlike forward cp, we don't invoke propagateDefs here,
      // just let forward cp do COPY-to-COPY propagation.
      if (isBackwardPropagatableCopy(*MI, *MRI)) {
        Tracker.invalidateRegister(Src, *TRI);
        Tracker.invalidateRegister(Def, *TRI);
        Tracker.trackCopy(MI, *TRI);
        continue;
      }
    }

    // Invalidate any earlyclobber regs first.
    for (const MachineOperand &MO : MI->operands())
      if (MO.isReg() && MO.isEarlyClobber()) {
        Register Reg = MO.getReg();
        if (!Reg)
          continue;
        Tracker.invalidateRegister(Reg, *TRI);
      }

    propagateDefs(*MI);
    for (const MachineOperand &MO : MI->operands()) {
      if (!MO.isReg())
        continue;

      if (!MO.getReg())
        continue;

      if (MO.isDef())
        Tracker.invalidateRegister(MO.getReg(), *TRI);

      if (MO.readsReg())
        Tracker.invalidateRegister(MO.getReg(), *TRI);
    }
  }

  for (auto *Copy : MaybeDeadCopies) {
    Copy->eraseFromParent();
    ++NumDeletes;
  }

  MaybeDeadCopies.clear();
  CopyDbgUsers.clear();
  Tracker.clear();
}

bool MachineCopyPropagation::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  Changed = false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  MRI = &MF.getRegInfo();

  for (MachineBasicBlock &MBB : MF) {
    BackwardCopyPropagateBlock(MBB);
    ForwardCopyPropagateBlock(MBB);
  }

  return Changed;
}
