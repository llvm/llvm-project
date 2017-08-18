//===- MachineCopyPropagation.cpp - Machine Copy Propagation Pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a simple MachineInstr-level copy forwarding pass.  It may be run at
// two places in the codegen pipeline:
//   - After register allocation but before virtual registers have been remapped
//     to physical registers.
//   - After physical register remapping.
//
// The optimizations done vary slightly based on whether virtual registers are
// still present.  In both cases, this pass forwards the source of COPYs to the
// users of their destinations when doing so is legal.  For example:
//
//   %vreg1 = COPY %vreg0
//   ...
//   ... = OP %vreg1
//
// If
//   - the physical register assigned to %vreg0 has not been clobbered by the
//     time of the use of %vreg1
//   - the register class constraints are satisfied
//   - the COPY def is the only value that reaches OP
// then this pass replaces the above with:
//
//   %vreg1 = COPY %vreg0
//   ...
//   ... = OP %vreg0
//
// and updates the relevant state required by VirtRegMap (e.g. LiveIntervals).
// COPYs whose LiveIntervals become dead as a result of this forwarding (i.e. if
// all uses of %vreg1 are changed to %vreg0) are removed.
//
// When being run with only physical registers, this pass will also remove some
// redundant COPYs.  For example:
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

#include "LiveDebugVariables.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#define DEBUG_TYPE "machine-cp"

STATISTIC(NumDeletes, "Number of dead copies deleted");
STATISTIC(NumCopyForwards, "Number of copy uses forwarded");
DEBUG_COUNTER(FwdCounter, "machine-cp-fwd",
              "Controls which register COPYs are forwarded");

namespace {
  typedef SmallVector<unsigned, 4> RegList;
  typedef DenseMap<unsigned, RegList> SourceMap;
  typedef DenseMap<unsigned, MachineInstr*> Reg2MIMap;

  class MachineCopyPropagation : public MachineFunctionPass,
                                 private LiveRangeEdit::Delegate {
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;
    MachineRegisterInfo *MRI;
    MachineFunction *MF;
    SlotIndexes *Indexes;
    LiveIntervals *LIS;
    const VirtRegMap *VRM;
    // True if this pass being run before virtual registers are remapped to
    // physical ones.
    bool PreRegRewrite;
    bool NoSubRegLiveness;

  protected:
    MachineCopyPropagation(char &ID, bool PreRegRewrite)
        : MachineFunctionPass(ID), PreRegRewrite(PreRegRewrite) {}

  public:
    static char ID; // Pass identification, replacement for typeid
    MachineCopyPropagation() : MachineCopyPropagation(ID, false) {
      initializeMachineCopyPropagationPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      if (PreRegRewrite) {
        AU.addRequired<SlotIndexes>();
        AU.addPreserved<SlotIndexes>();
        AU.addRequired<LiveIntervals>();
        AU.addPreserved<LiveIntervals>();
        AU.addRequired<VirtRegMap>();
        AU.addPreserved<VirtRegMap>();
        AU.addPreserved<LiveDebugVariables>();
        AU.addPreserved<LiveStacks>();
      }
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

    MachineFunctionProperties getRequiredProperties() const override {
      if (PreRegRewrite)
        return MachineFunctionProperties()
            .set(MachineFunctionProperties::Property::NoPHIs)
            .set(MachineFunctionProperties::Property::TracksLiveness);
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoVRegs);
    }

  private:
    void ClobberRegister(unsigned Reg);
    void ReadRegister(unsigned Reg);
    void CopyPropagateBlock(MachineBasicBlock &MBB);
    bool eraseIfRedundant(MachineInstr &Copy, unsigned Src, unsigned Def);
    unsigned getPhysReg(unsigned Reg, unsigned SubReg);
    unsigned getPhysReg(const MachineOperand &Opnd) {
      return getPhysReg(Opnd.getReg(), Opnd.getSubReg());
    }
    unsigned getFullPhysReg(const MachineOperand &Opnd) {
      return getPhysReg(Opnd.getReg(), 0);
    }
    void forwardUses(MachineInstr &MI);
    bool isForwardableRegClassCopy(const MachineInstr &Copy,
                                   const MachineInstr &UseI);
    std::tuple<unsigned, unsigned, bool>
    checkUseSubReg(const MachineOperand &CopySrc, const MachineOperand &MOUse);
    bool hasImplicitOverlap(const MachineInstr &MI, const MachineOperand &Use);
    void narrowRegClass(const MachineInstr &MI, const MachineOperand &MOUse,
                        unsigned NewUseReg, unsigned NewUseSubReg);
    void updateForwardedCopyLiveInterval(const MachineInstr &Copy,
                                         const MachineInstr &UseMI,
                                         unsigned OrigUseReg,
                                         unsigned NewUseReg,
                                         unsigned NewUseSubReg);
    /// LiveRangeEdit callback for eliminateDeadDefs().
    void LRE_WillEraseInstruction(MachineInstr *MI) override;

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

  class MachineCopyPropagationPreRegRewrite : public MachineCopyPropagation {
  public:
    static char ID; // Pass identification, replacement for typeid
    MachineCopyPropagationPreRegRewrite()
        : MachineCopyPropagation(ID, true) {
      initializeMachineCopyPropagationPreRegRewritePass(*PassRegistry::getPassRegistry());
    }
  };
}
char MachineCopyPropagation::ID = 0;
char &llvm::MachineCopyPropagationID = MachineCopyPropagation::ID;

INITIALIZE_PASS(MachineCopyPropagation, DEBUG_TYPE,
                "Machine Copy Propagation Pass", false, false)

/// We have two separate passes that are very similar, the only difference being
/// where they are meant to be run in the pipeline.  This is done for several
/// reasons:
/// - the two passes have different dependencies
/// - some targets want to disable the later run of this pass, but not the
///   earlier one (e.g. NVPTX and WebAssembly)
/// - it allows for easier debugging via llc

char MachineCopyPropagationPreRegRewrite::ID = 0;
char &llvm::MachineCopyPropagationPreRegRewriteID = MachineCopyPropagationPreRegRewrite::ID;

INITIALIZE_PASS_BEGIN(MachineCopyPropagationPreRegRewrite,
                      "machine-cp-prerewrite",
                      "Machine Copy Propagation Pre-Register Rewrite Pass",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_END(MachineCopyPropagationPreRegRewrite,
                    "machine-cp-prerewrite",
                    "Machine Copy Propagation Pre-Register Rewrite Pass", false,
                    false)

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
  // We don't track MaybeDeadCopies when running pre-VirtRegRewriter.
  if (PreRegRewrite)
    return;

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

/// Return the physical register assigned to \p Reg if it is a virtual register,
/// otherwise just return the physical reg from the operand itself.
///
/// If \p SubReg is 0 then return the full physical register assigned to the
/// virtual register ignoring subregs.  If we aren't tracking sub-reg liveness
/// then we need to use this to be more conservative with clobbers by killing
/// all super reg and their sub reg COPYs as well.  This is to prevent COPY
/// forwarding in cases like the following:
///
///    %vreg2 = COPY %vreg1:sub1
///    %vreg3 = COPY %vreg1:sub0
///    ...    = OP1 %vreg2
///    ...    = OP2 %vreg3
///
/// After forward %vreg2 (assuming this is the last use of %vreg1) and
/// VirtRegRewriter adding kill markers we have:
///
///    %vreg3 = COPY %vreg1:sub0
///    ...    = OP1 %vreg1:sub1<kill>
///    ...    = OP2 %vreg3
///
/// If %vreg3 is assigned to a sub-reg of %vreg1, then after rewriting we have:
///
///    ...     = OP1 R0:sub1, R0<imp-use,kill>
///    ...     = OP2 R0:sub0
///
/// and the use of R0 by OP2 will not have a valid definition.
unsigned MachineCopyPropagation::getPhysReg(unsigned Reg, unsigned SubReg) {

  // Physical registers cannot have subregs.
  if (!TargetRegisterInfo::isVirtualRegister(Reg))
    return Reg;

  assert(PreRegRewrite && "Unexpected virtual register encountered");
  Reg = VRM->getPhys(Reg);
  if (SubReg && !NoSubRegLiveness)
    Reg = TRI->getSubReg(Reg, SubReg);
  return Reg;
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


/// Decide whether we should forward the destination of \param Copy to its use
/// in \param UseI based on the register class of the Copy operands.  Same-class
/// COPYs are always accepted by this function, but cross-class COPYs are only
/// accepted if they are forwarded to another COPY with the operand register
/// classes reversed.  For example:
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
/// introduced a no COPY that can be removed.
bool MachineCopyPropagation::isForwardableRegClassCopy(
    const MachineInstr &Copy, const MachineInstr &UseI) {
  auto isCross = [&](const MachineOperand &Dst, const MachineOperand &Src) {
    unsigned DstReg = Dst.getReg();
    unsigned SrcPhysReg = getPhysReg(Src);
    const TargetRegisterClass *DstRC;
    if (TargetRegisterInfo::isVirtualRegister(DstReg)) {
      DstRC = MRI->getRegClass(DstReg);
      unsigned DstSubReg = Dst.getSubReg();
      if (DstSubReg)
        SrcPhysReg = TRI->getMatchingSuperReg(SrcPhysReg, DstSubReg, DstRC);
    } else
      DstRC = TRI->getMinimalPhysRegClass(DstReg);

    return !DstRC->contains(SrcPhysReg);
  };

  const MachineOperand &CopyDst = Copy.getOperand(0);
  const MachineOperand &CopySrc = Copy.getOperand(1);

  if (!isCross(CopyDst, CopySrc))
    return true;

  if (!UseI.isCopy())
    return false;

  assert(getFullPhysReg(UseI.getOperand(1)) == getFullPhysReg(CopyDst));
  return !isCross(UseI.getOperand(0), CopySrc);
}

/// Check that the subregs on the copy source operand (\p CopySrc) and the use
/// operand to be forwarded to (\p MOUse) are compatible with doing the
/// forwarding.  Also computes the new register and subregister to be used in
/// the forwarded-to instruction.
std::tuple<unsigned, unsigned, bool> MachineCopyPropagation::checkUseSubReg(
    const MachineOperand &CopySrc, const MachineOperand &MOUse) {
  unsigned NewUseReg = CopySrc.getReg();
  unsigned NewUseSubReg;

  if (TargetRegisterInfo::isPhysicalRegister(NewUseReg)) {
    // If MOUse is a virtual reg, we need to apply it to the new physical reg
    // we're going to replace it with.
    if (MOUse.getSubReg())
      NewUseReg = TRI->getSubReg(NewUseReg, MOUse.getSubReg());
    // If the original use subreg isn't valid on the new src reg, we can't
    // forward it here.
    if (!NewUseReg)
      return std::make_tuple(0, 0, false);
    NewUseSubReg = 0;
  } else {
    // %v1 = COPY %v2:sub1
    //    USE %v1:sub2
    // The new use is %v2:sub1:sub2
    NewUseSubReg =
        TRI->composeSubRegIndices(CopySrc.getSubReg(), MOUse.getSubReg());
    // Check that NewUseSubReg is valid on NewUseReg
    if (NewUseSubReg &&
        !TRI->getSubClassWithSubReg(MRI->getRegClass(NewUseReg), NewUseSubReg))
      return std::make_tuple(0, 0, false);
  }

  return std::make_tuple(NewUseReg, NewUseSubReg, true);
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
  if (!TargetRegisterInfo::isPhysicalRegister(Use.getReg()))
    return false;

  for (const MachineOperand &MIUse : MI.uses())
    if (&MIUse != &Use && MIUse.isReg() && MIUse.isImplicit() &&
        TRI->regsOverlap(Use.getReg(), MIUse.getReg()))
      return true;

  return false;
}

/// Narrow the register class of the forwarded vreg so it matches any
/// instruction constraints.  \p MI is the instruction being forwarded to. \p
/// MOUse is the operand being replaced in \p MI (which hasn't yet been updated
/// at the time this function is called).  \p NewUseReg and \p NewUseSubReg are
/// what the \p MOUse will be changed to after forwarding.
///
/// If we are forwarding
///    A:RCA = COPY B:RCB
/// into
///    ... = OP A:RCA
///
/// then we need to narrow the register class of B so that it is a subclass
/// of RCA so that it meets the instruction register class constraints.
void MachineCopyPropagation::narrowRegClass(const MachineInstr &MI,
                                            const MachineOperand &MOUse,
                                            unsigned NewUseReg,
                                            unsigned NewUseSubReg) {
  if (!TargetRegisterInfo::isVirtualRegister(NewUseReg))
    return;

  // Make sure the virtual reg class allows the subreg.
  if (NewUseSubReg) {
    const TargetRegisterClass *CurUseRC = MRI->getRegClass(NewUseReg);
    const TargetRegisterClass *NewUseRC =
        TRI->getSubClassWithSubReg(CurUseRC, NewUseSubReg);
    if (CurUseRC != NewUseRC) {
      DEBUG(dbgs() << "MCP: Setting regclass of " << PrintReg(NewUseReg, TRI)
                   << " to " << TRI->getRegClassName(NewUseRC) << "\n");
      MRI->setRegClass(NewUseReg, NewUseRC);
    }
  }

  unsigned MOUseOpNo = &MOUse - &MI.getOperand(0);
  const TargetRegisterClass *InstRC =
      TII->getRegClass(MI.getDesc(), MOUseOpNo, TRI, *MF);
  if (InstRC) {
    const TargetRegisterClass *CurUseRC = MRI->getRegClass(NewUseReg);
    if (NewUseSubReg)
      InstRC = TRI->getMatchingSuperRegClass(CurUseRC, InstRC, NewUseSubReg);
    if (!InstRC->hasSubClassEq(CurUseRC)) {
      const TargetRegisterClass *NewUseRC =
          TRI->getCommonSubClass(InstRC, CurUseRC);
      DEBUG(dbgs() << "MCP: Setting regclass of " << PrintReg(NewUseReg, TRI)
                   << " to " << TRI->getRegClassName(NewUseRC) << "\n");
      MRI->setRegClass(NewUseReg, NewUseRC);
    }
  }
}

/// Update the LiveInterval information to reflect the destination of \p Copy
/// being forwarded to a use in \p UseMI.  \p OrigUseReg is the register being
/// forwarded through. It should be the destination register of \p Copy and has
/// already been replaced in \p UseMI at the point this function is called.  \p
/// NewUseReg and \p NewUseSubReg are the register and subregister being
/// forwarded.  They should be the source register of the \p Copy and should be
/// the value of the \p UseMI operand being forwarded at the point this function
/// is called.
void MachineCopyPropagation::updateForwardedCopyLiveInterval(
    const MachineInstr &Copy, const MachineInstr &UseMI, unsigned OrigUseReg,
    unsigned NewUseReg, unsigned NewUseSubReg) {

  assert(TRI->isSubRegisterEq(getPhysReg(OrigUseReg, 0),
                              getFullPhysReg(Copy.getOperand(0))) &&
         "OrigUseReg mismatch");
  assert(TRI->isSubRegisterEq(getFullPhysReg(Copy.getOperand(1)),
                              getPhysReg(NewUseReg, 0)) &&
         "NewUseReg mismatch");

  // Extend live range starting from COPY early-clobber slot, since that
  // is where the original src live range ends.
  SlotIndex CopyUseIdx =
      Indexes->getInstructionIndex(Copy).getRegSlot(true /*=EarlyClobber*/);
  SlotIndex UseIdx = Indexes->getInstructionIndex(UseMI).getRegSlot();
  if (TargetRegisterInfo::isVirtualRegister(NewUseReg)) {
    LiveInterval &LI = LIS->getInterval(NewUseReg);
    LI.extendInBlock(CopyUseIdx, UseIdx);
    LaneBitmask UseMask = TRI->getSubRegIndexLaneMask(NewUseSubReg);
    for (auto &S : LI.subranges())
      if ((S.LaneMask & UseMask).any() && S.find(CopyUseIdx))
        S.extendInBlock(CopyUseIdx, UseIdx);
  } else {
    assert(NewUseSubReg == 0 && "Unexpected subreg on physical register!");
    for (MCRegUnitIterator UI(NewUseReg, TRI); UI.isValid(); ++UI) {
      LiveRange &LR = LIS->getRegUnit(*UI);
      LR.extendInBlock(CopyUseIdx, UseIdx);
    }
  }

  if (!TargetRegisterInfo::isVirtualRegister(OrigUseReg))
    return;

  LiveInterval &LI = LIS->getInterval(OrigUseReg);

  // Can happen for undef uses.
  if (LI.empty())
    return;

  SlotIndex UseIndex = Indexes->getInstructionIndex(UseMI);
  const LiveRange::Segment *UseSeg = LI.getSegmentContaining(UseIndex);

  // Only shrink if forwarded use is the end of a segment.
  if (UseSeg->end != UseIndex.getRegSlot())
    return;

  SmallVector<MachineInstr *, 4> DeadInsts;
  LIS->shrinkToUses(&LI, &DeadInsts);
  if (!DeadInsts.empty()) {
    SmallVector<unsigned, 8> NewRegs;
    LiveRangeEdit(nullptr, NewRegs, *MF, *LIS, nullptr, this)
        .eliminateDeadDefs(DeadInsts);
  }
}

void MachineCopyPropagation::LRE_WillEraseInstruction(MachineInstr *MI) {
  // Remove this COPY from further consideration for forwarding.
  ClobberRegister(getFullPhysReg(MI->getOperand(0)));
  Changed = true;
}

/// Look for available copies whose destination register is used by \p MI and
/// replace the use in \p MI with the copy's source register.
void MachineCopyPropagation::forwardUses(MachineInstr &MI) {
  // We can't generally forward uses after virtual registers have been renamed
  // because some targets generate code that has implicit dependencies on the
  // physical register numbers.  For example, in PowerPC, when spilling
  // condition code registers, the following code pattern is generated:
  //
  //   %CR7 = COPY %CR0
  //   %R6 = MFOCRF %CR7
  //   %R6 = RLWINM %R6, 29, 31, 31
  //
  // where the shift amount in the RLWINM instruction depends on the source
  // register number of the MFOCRF instruction.  If we were to forward %CR0 to
  // the MFOCRF instruction, the shift amount would no longer be correct.
  //
  // FIXME: It may be possible to define a target hook that checks the register
  // class or user opcode and allows some cases, but prevents cases like the
  // above from being broken to enable later register copy forwarding.
  if (!PreRegRewrite)
    return;

  if (AvailCopyMap.empty())
    return;

  // Look for non-tied explicit vreg uses that have an active COPY
  // instruction that defines the physical register allocated to them.
  // Replace the vreg with the source of the active COPY.
  for (MachineOperand &MOUse : MI.explicit_uses()) {
    // Don't forward into undef use operands since doing so can cause problems
    // with the machine verifier, since it doesn't treat undef reads as reads,
    // so we can end up with a live range the ends on an undef read, leading to
    // an error that the live range doesn't end on a read of the live range
    // register.
    if (!MOUse.isReg() || MOUse.isTied() || MOUse.isUndef())
      continue;

    unsigned UseReg = MOUse.getReg();
    if (!UseReg)
      continue;

    // See comment above check for !PreRegRewrite regarding forwarding changing
    // physical registers.
    if (!TargetRegisterInfo::isVirtualRegister(UseReg))
      continue;

    UseReg = VRM->getPhys(UseReg);

    // Don't forward COPYs via non-allocatable regs since they can have
    // non-standard semantics.
    if (!MRI->isAllocatable(UseReg))
      continue;

    auto CI = AvailCopyMap.find(UseReg);
    if (CI == AvailCopyMap.end())
      continue;

    MachineInstr &Copy = *CI->second;
    MachineOperand &CopyDst = Copy.getOperand(0);
    MachineOperand &CopySrc = Copy.getOperand(1);

    // Don't forward COPYs that are already NOPs due to register assignment.
    if (getPhysReg(CopyDst) == getPhysReg(CopySrc))
      continue;

    // FIXME: Don't handle partial uses of wider COPYs yet.
    if (CopyDst.getSubReg() != 0 || UseReg != getPhysReg(CopyDst))
      continue;

    // Don't forward COPYs of non-allocatable regs unless they are constant.
    unsigned CopySrcReg = CopySrc.getReg();
    if (TargetRegisterInfo::isPhysicalRegister(CopySrcReg) &&
        !MRI->isAllocatable(CopySrcReg) && !MRI->isConstantPhysReg(CopySrcReg))
      continue;

    if (!isForwardableRegClassCopy(Copy, MI))
      continue;

    unsigned NewUseReg, NewUseSubReg;
    bool SubRegOK;
    std::tie(NewUseReg, NewUseSubReg, SubRegOK) =
        checkUseSubReg(CopySrc, MOUse);
    if (!SubRegOK)
      continue;

    if (hasImplicitOverlap(MI, MOUse))
      continue;

    if (!DebugCounter::shouldExecute(FwdCounter))
      continue;

    DEBUG(dbgs() << "MCP: Replacing "
          << PrintReg(MOUse.getReg(), TRI, MOUse.getSubReg())
          << "\n     with "
          << PrintReg(NewUseReg, TRI, CopySrc.getSubReg())
          << "\n     in "
          << MI
          << "     from "
          << Copy);

    narrowRegClass(MI, MOUse, NewUseReg, NewUseSubReg);

    unsigned OrigUseReg = MOUse.getReg();
    MOUse.setReg(NewUseReg);
    MOUse.setSubReg(NewUseSubReg);

    DEBUG(dbgs() << "MCP: After replacement: " << MI << "\n");

    if (PreRegRewrite)
      updateForwardedCopyLiveInterval(Copy, MI, OrigUseReg, NewUseReg,
                                      NewUseSubReg);
    else
      for (MachineInstr &KMI :
             make_range(Copy.getIterator(), std::next(MI.getIterator())))
        KMI.clearRegisterKills(NewUseReg, TRI);

    ++NumCopyForwards;
    Changed = true;
  }
}

void MachineCopyPropagation::CopyPropagateBlock(MachineBasicBlock &MBB) {
  DEBUG(dbgs() << "MCP: CopyPropagateBlock " << MBB.getName() << "\n");

  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ) {
    MachineInstr *MI = &*I;
    ++I;

    if (MI->isCopy()) {
      unsigned Def = getPhysReg(MI->getOperand(0));
      unsigned Src = getPhysReg(MI->getOperand(1));

      // The two copies cancel out and the source of the first copy
      // hasn't been overridden, eliminate the second one. e.g.
      //  %ECX<def> = COPY %EAX
      //  ... nothing clobbered EAX.
      //  %EAX<def> = COPY %ECX
      // =>
      //  %ECX<def> = COPY %EAX
      //
      // or
      //
      //  %ECX<def> = COPY %EAX
      //  ... nothing clobbered EAX.
      //  %ECX<def> = COPY %EAX
      // =>
      //  %ECX<def> = COPY %EAX
      if (!PreRegRewrite)
        if (eraseIfRedundant(*MI, Def, Src) || eraseIfRedundant(*MI, Src, Def))
          continue;

      forwardUses(*MI);

      // Src may have been changed by forwardUses()
      Src = getPhysReg(MI->getOperand(1));
      unsigned DefClobber = getFullPhysReg(MI->getOperand(0));
      unsigned SrcClobber = getFullPhysReg(MI->getOperand(1));

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
      // Only look for dead COPYs if we're not running just before
      // VirtRegRewriter, since presumably these COPYs will have already been
      // removed.
      if (!PreRegRewrite && !MRI->isReserved(Def))
        MaybeDeadCopies.insert(MI);

      // If 'Def' is previously source of another copy, then this earlier copy's
      // source is no longer available. e.g.
      // %xmm9<def> = copy %xmm2
      // ...
      // %xmm2<def> = copy %xmm0
      // ...
      // %xmm2<def> = copy %xmm9
      ClobberRegister(DefClobber);
      for (const MachineOperand &MO : MI->implicit_operands()) {
        if (!MO.isReg() || !MO.isDef())
          continue;
        unsigned Reg = getFullPhysReg(MO);
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
      RegList &DestList = SrcMap[SrcClobber];
      if (!is_contained(DestList, DefClobber))
        DestList.push_back(DefClobber);

      continue;
    }

    // Clobber any earlyclobber regs first.
    for (const MachineOperand &MO : MI->operands())
      if (MO.isReg() && MO.isEarlyClobber()) {
        unsigned Reg = getFullPhysReg(MO);
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
      unsigned Reg = getFullPhysReg(MO);
      if (!Reg)
        continue;

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
  if (skipFunction(*MF.getFunction()))
    return false;

  Changed = false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  MRI = &MF.getRegInfo();
  this->MF = &MF;
  if (PreRegRewrite) {
    Indexes = &getAnalysis<SlotIndexes>();
    LIS = &getAnalysis<LiveIntervals>();
    VRM = &getAnalysis<VirtRegMap>();
  }
  NoSubRegLiveness = !MRI->subRegLivenessEnabled();

  for (MachineBasicBlock &MBB : MF)
    CopyPropagateBlock(MBB);

  return Changed;
}
