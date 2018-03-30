//===- MachineCopyPropagation.cpp - Machine Copy Propagation Pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
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
DEBUG_COUNTER(FwdCounter, "machine-cp-fwd",
              "Controls which register COPYs are forwarded");

namespace {

using RegList = SmallVector<unsigned, 4>;
using SourceMap = DenseMap<unsigned, RegList>;
using Reg2MIMap = DenseMap<unsigned, MachineInstr *>;

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
    void ClobberRegister(unsigned Reg);
    void ReadRegister(unsigned Reg);
    void CopyPropagateBlock(MachineBasicBlock &MBB);
    bool eraseIfRedundant(MachineInstr &Copy, unsigned Src, unsigned Def);
    void forwardUses(MachineInstr &MI);
    bool isForwardableRegClassCopy(const MachineInstr &Copy,
                                   const MachineInstr &UseI, unsigned UseIdx);
    bool hasImplicitOverlap(const MachineInstr &MI, const MachineOperand &Use);

    /// Candidates for deletion.
    SmallSetVector<MachineInstr*, 8> MaybeDeadCopies;

    /// Def -> available copies map.
    Reg2MIMap AvailCopyMap;

    /// Def -> copies map.
    Reg2MIMap CopyMap;

    /// Src -> Def map
    SourceMap SrcMap;

    bool Changed;
  };

} // end anonymous namespace

char MachineCopyPropagation::ID = 0;

char &llvm::MachineCopyPropagationID = MachineCopyPropagation::ID;

INITIALIZE_PASS(MachineCopyPropagation, DEBUG_TYPE,
                "Machine Copy Propagation Pass", false, false)

/// Remove any entry in \p Map where the register is a subregister or equal to
/// a register contained in \p Regs.
static void removeRegsFromMap(Reg2MIMap &Map, const RegList &Regs,
                              const TargetRegisterInfo &TRI) {
  for (unsigned Reg : Regs) {
    // Source of copy is no longer available for propagation.
    for (MCSubRegIterator SR(Reg, &TRI, true); SR.isValid(); ++SR)
      Map.erase(*SR);
  }
}

/// Remove any entry in \p Map that is marked clobbered in \p RegMask.
/// The map will typically have a lot fewer entries than the regmask clobbers,
/// so this is more efficient than iterating the clobbered registers and calling
/// ClobberRegister() on them.
static void removeClobberedRegsFromMap(Reg2MIMap &Map,
                                       const MachineOperand &RegMask) {
  for (Reg2MIMap::iterator I = Map.begin(), E = Map.end(), Next; I != E;
       I = Next) {
    Next = std::next(I);
    unsigned Reg = I->first;
    if (RegMask.clobbersPhysReg(Reg))
      Map.erase(I);
  }
}

void MachineCopyPropagation::ClobberRegister(unsigned Reg) {
  for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI) {
    CopyMap.erase(*AI);
    AvailCopyMap.erase(*AI);

    SourceMap::iterator SI = SrcMap.find(*AI);
    if (SI != SrcMap.end()) {
      removeRegsFromMap(AvailCopyMap, SI->second, *TRI);
      SrcMap.erase(SI);
    }
  }
}

void MachineCopyPropagation::ReadRegister(unsigned Reg) {
  // If 'Reg' is defined by a copy, the copy is no longer a candidate
  // for elimination.
  for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI) {
    Reg2MIMap::iterator CI = CopyMap.find(*AI);
    if (CI != CopyMap.end()) {
      DEBUG(dbgs() << "MCP: Copy is used - not dead: "; CI->second->dump());
      MaybeDeadCopies.remove(CI->second);
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
  unsigned PreviousSrc = PreviousCopy.getOperand(1).getReg();
  unsigned PreviousDef = PreviousCopy.getOperand(0).getReg();
  if (Src == PreviousSrc) {
    assert(Def == PreviousDef);
    return true;
  }
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
  Reg2MIMap::iterator CI = AvailCopyMap.find(Def);
  if (CI == AvailCopyMap.end())
    return false;

  // Check that the existing copy uses the correct sub registers.
  MachineInstr &PrevCopy = *CI->second;
  if (PrevCopy.getOperand(0).isDead())
    return false;
  if (!isNopCopy(PrevCopy, Src, Def, TRI))
    return false;

  DEBUG(dbgs() << "MCP: copy is a NOP, removing: "; Copy.dump());

  // Copy was redundantly redefining either Src or Def. Remove earlier kill
  // flags between Copy and PrevCopy because the value will be reused now.
  assert(Copy.isCopy());
  unsigned CopyDef = Copy.getOperand(0).getReg();
  assert(CopyDef == Src || CopyDef == Def);
  for (MachineInstr &MI :
       make_range(PrevCopy.getIterator(), Copy.getIterator()))
    MI.clearRegisterKills(CopyDef, TRI);

  Copy.eraseFromParent();
  Changed = true;
  ++NumDeletes;
  return true;
}

/// Decide whether we should forward the source of \param Copy to its use in
/// \param UseI based on the physical register class constraints of the opcode
/// and avoiding introducing more cross-class COPYs.
bool MachineCopyPropagation::isForwardableRegClassCopy(const MachineInstr &Copy,
                                                       const MachineInstr &UseI,
                                                       unsigned UseIdx) {

  unsigned CopySrcReg = Copy.getOperand(1).getReg();

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
  if (AvailCopyMap.empty())
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

    auto CI = AvailCopyMap.find(MOUse.getReg());
    if (CI == AvailCopyMap.end())
      continue;

    MachineInstr &Copy = *CI->second;
    unsigned CopyDstReg = Copy.getOperand(0).getReg();
    const MachineOperand &CopySrc = Copy.getOperand(1);
    unsigned CopySrcReg = CopySrc.getReg();

    // FIXME: Don't handle partial uses of wider COPYs yet.
    if (MOUse.getReg() != CopyDstReg) {
      DEBUG(dbgs() << "MCP: FIXME! Not forwarding COPY to sub-register use:\n  "
                   << MI);
      continue;
    }

    // Don't forward COPYs of reserved regs unless they are constant.
    if (MRI->isReserved(CopySrcReg) && !MRI->isConstantPhysReg(CopySrcReg))
      continue;

    if (!isForwardableRegClassCopy(Copy, MI, OpIdx))
      continue;

    if (hasImplicitOverlap(MI, MOUse))
      continue;

    if (!DebugCounter::shouldExecute(FwdCounter)) {
      DEBUG(dbgs() << "MCP: Skipping forwarding due to debug counter:\n  "
                   << MI);
      continue;
    }

    DEBUG(dbgs() << "MCP: Replacing " << printReg(MOUse.getReg(), TRI)
                 << "\n     with " << printReg(CopySrcReg, TRI) << "\n     in "
                 << MI << "     from " << Copy);

    MOUse.setReg(CopySrcReg);
    if (!CopySrc.isRenamable())
      MOUse.setIsRenamable(false);

    DEBUG(dbgs() << "MCP: After replacement: " << MI << "\n");

    // Clear kill markers that may have been invalidated.
    for (MachineInstr &KMI :
         make_range(Copy.getIterator(), std::next(MI.getIterator())))
      KMI.clearRegisterKills(CopySrcReg, TRI);

    ++NumCopyForwards;
    Changed = true;
  }
}

void MachineCopyPropagation::CopyPropagateBlock(MachineBasicBlock &MBB) {
  DEBUG(dbgs() << "MCP: CopyPropagateBlock " << MBB.getName() << "\n");

  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ) {
    MachineInstr *MI = &*I;
    ++I;

    // Analyze copies (which don't overlap themselves).
    if (MI->isCopy() && !TRI->regsOverlap(MI->getOperand(0).getReg(),
                                          MI->getOperand(1).getReg())) {
      unsigned Def = MI->getOperand(0).getReg();
      unsigned Src = MI->getOperand(1).getReg();

      assert(!TargetRegisterInfo::isVirtualRegister(Def) &&
             !TargetRegisterInfo::isVirtualRegister(Src) &&
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
      ReadRegister(Src);
      for (const MachineOperand &MO : MI->implicit_operands()) {
        if (!MO.isReg() || !MO.readsReg())
          continue;
        unsigned Reg = MO.getReg();
        if (!Reg)
          continue;
        ReadRegister(Reg);
      }

      DEBUG(dbgs() << "MCP: Copy is a deletion candidate: "; MI->dump());

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
      ClobberRegister(Def);
      for (const MachineOperand &MO : MI->implicit_operands()) {
        if (!MO.isReg() || !MO.isDef())
          continue;
        unsigned Reg = MO.getReg();
        if (!Reg)
          continue;
        ClobberRegister(Reg);
      }

      // Remember Def is defined by the copy.
      for (MCSubRegIterator SR(Def, TRI, /*IncludeSelf=*/true); SR.isValid();
           ++SR) {
        CopyMap[*SR] = MI;
        AvailCopyMap[*SR] = MI;
      }

      // Remember source that's copied to Def. Once it's clobbered, then
      // it's no longer available for copy propagation.
      RegList &DestList = SrcMap[Src];
      if (!is_contained(DestList, Def))
          DestList.push_back(Def);

      continue;
    }

    // Clobber any earlyclobber regs first.
    for (const MachineOperand &MO : MI->operands())
      if (MO.isReg() && MO.isEarlyClobber()) {
        unsigned Reg = MO.getReg();
        // If we have a tied earlyclobber, that means it is also read by this
        // instruction, so we need to make sure we don't remove it as dead
        // later.
        if (MO.isTied())
          ReadRegister(Reg);
        ClobberRegister(Reg);
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
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;

      assert(!TargetRegisterInfo::isVirtualRegister(Reg) &&
             "MachineCopyPropagation should be run after register allocation!");

      if (MO.isDef() && !MO.isEarlyClobber()) {
        Defs.push_back(Reg);
        continue;
      } else if (MO.readsReg())
        ReadRegister(Reg);
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
        unsigned Reg = MaybeDead->getOperand(0).getReg();
        assert(!MRI->isReserved(Reg));

        if (!RegMask->clobbersPhysReg(Reg)) {
          ++DI;
          continue;
        }

        DEBUG(dbgs() << "MCP: Removing copy due to regmask clobbering: ";
              MaybeDead->dump());

        // erase() will return the next valid iterator pointing to the next
        // element after the erased one.
        DI = MaybeDeadCopies.erase(DI);
        MaybeDead->eraseFromParent();
        Changed = true;
        ++NumDeletes;
      }

      removeClobberedRegsFromMap(AvailCopyMap, *RegMask);
      removeClobberedRegsFromMap(CopyMap, *RegMask);
      for (SourceMap::iterator I = SrcMap.begin(), E = SrcMap.end(), Next;
           I != E; I = Next) {
        Next = std::next(I);
        if (RegMask->clobbersPhysReg(I->first)) {
          removeRegsFromMap(AvailCopyMap, I->second, *TRI);
          SrcMap.erase(I);
        }
      }
    }

    // Any previous copy definition or reading the Defs is no longer available.
    for (unsigned Reg : Defs)
      ClobberRegister(Reg);
  }

  // If MBB doesn't have successors, delete the copies whose defs are not used.
  // If MBB does have successors, then conservative assume the defs are live-out
  // since we don't want to trust live-in lists.
  if (MBB.succ_empty()) {
    for (MachineInstr *MaybeDead : MaybeDeadCopies) {
      DEBUG(dbgs() << "MCP: Removing copy due to no live-out succ: ";
            MaybeDead->dump());
      assert(!MRI->isReserved(MaybeDead->getOperand(0).getReg()));
      MaybeDead->eraseFromParent();
      Changed = true;
      ++NumDeletes;
    }
  }

  MaybeDeadCopies.clear();
  AvailCopyMap.clear();
  CopyMap.clear();
  SrcMap.clear();
}

bool MachineCopyPropagation::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  Changed = false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  MRI = &MF.getRegInfo();

  for (MachineBasicBlock &MBB : MF)
    CopyPropagateBlock(MBB);

  return Changed;
}
