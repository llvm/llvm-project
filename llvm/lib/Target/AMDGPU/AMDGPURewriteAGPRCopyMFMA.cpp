//===-- AMDGPURewriteAGPRCopyMFMA.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file \brief Try to replace MFMA instructions using VGPRs with MFMA
/// instructions using AGPRs. We expect MFMAs to be selected using VGPRs, and
/// only use AGPRs if it helps avoid spilling. In this case, the MFMA will have
/// copies between AGPRs and VGPRs and the AGPR variant of an MFMA pseudo. This
/// pass will attempt to delete the cross register bank copy and replace the
/// MFMA opcode.
///
/// TODO:
///  - Handle SplitKit partial copy bundles, and not just full copy instructions
///
///  - Update LiveIntervals incrementally instead of recomputing from scratch
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-rewrite-agpr-copy-mfma"

namespace {

class AMDGPURewriteAGPRCopyMFMAImpl {
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  MachineRegisterInfo &MRI;
  VirtRegMap &VRM;
  LiveRegMatrix &LRM;
  LiveIntervals &LIS;
  const RegisterClassInfo &RegClassInfo;

  bool attemptReassignmentsToAGPR(SmallSetVector<Register, 4> &InterferingRegs,
                                  MCPhysReg PrefPhysReg) const;

public:
  AMDGPURewriteAGPRCopyMFMAImpl(MachineFunction &MF, VirtRegMap &VRM,
                                LiveRegMatrix &LRM, LiveIntervals &LIS,
                                const RegisterClassInfo &RegClassInfo)
      : ST(MF.getSubtarget<GCNSubtarget>()), TII(*ST.getInstrInfo()),
        TRI(*ST.getRegisterInfo()), MRI(MF.getRegInfo()), VRM(VRM), LRM(LRM),
        LIS(LIS), RegClassInfo(RegClassInfo) {}

  bool isRewriteCandidate(const MachineInstr &MI) const {
    return TII.isMAI(MI) && AMDGPU::getMFMASrcCVDstAGPROp(MI.getOpcode()) != -1;
  }

  /// Compute the register class constraints based on the uses of \p Reg,
  /// excluding MFMA uses from which can be rewritten to change the register
  /// class constraint. This should be nearly identical to
  /// MachineRegisterInfo::recomputeRegClass.

  /// \p RewriteCandidates will collect the set of MFMA instructions that need
  /// to have the opcode mutated to perform the replacement.
  ///
  /// \p RewriteRegs will accumulate the set of register used by those MFMAs
  /// that need to have the register classes adjusted.
  const TargetRegisterClass *recomputeRegClassExceptRewritable(
      Register Reg, const TargetRegisterClass *OldRC,
      const TargetRegisterClass *NewRC,
      SmallVectorImpl<MachineInstr *> &RewriteCandidates,
      SmallSetVector<Register, 4> &RewriteRegs) const;

  bool run(MachineFunction &MF) const;
};

const TargetRegisterClass *
AMDGPURewriteAGPRCopyMFMAImpl::recomputeRegClassExceptRewritable(
    Register StartReg, const TargetRegisterClass *OldRC,
    const TargetRegisterClass *NewRC,
    SmallVectorImpl<MachineInstr *> &RewriteCandidates,
    SmallSetVector<Register, 4> &RewriteRegs) const {
  SmallVector<Register, 8> Worklist = {StartReg};

  // Recursively visit all transitive MFMA users
  while (!Worklist.empty()) {
    Register Reg = Worklist.pop_back_val();
    // Accumulate constraints from all uses.
    for (MachineOperand &MO : MRI.reg_nodbg_operands(Reg)) {
      // Apply the effect of the given operand to NewRC.
      MachineInstr *MI = MO.getParent();

      // We can swap the classes of dst + src2 as a pair to AGPR, so ignore the
      // effects of rewrite candidates. It just so happens that we can use
      // either AGPR or VGPR in src0/src1, so don't bother checking the
      // constraint effects of the individual operands.
      if (isRewriteCandidate(*MI)) {
        for (AMDGPU::OpName OpName :
             {AMDGPU::OpName::vdst, AMDGPU::OpName::src2}) {
          const MachineOperand *Op = TII.getNamedOperand(*MI, OpName);
          if (!Op->isReg())
            continue;

          Register OtherReg = Op->getReg();
          if (OtherReg != Reg) {
            if (RewriteRegs.insert(OtherReg))
              Worklist.push_back(OtherReg);
          }
        }

        LLVM_DEBUG(dbgs() << "Ignoring effects of " << *MI);

        if (!is_contained(RewriteCandidates, MI))
          RewriteCandidates.push_back(MI);

        continue;
      }

      unsigned OpNo = &MO - &MI->getOperand(0);
      NewRC = MI->getRegClassConstraintEffect(OpNo, NewRC, &TII, &TRI);
      if (!NewRC || NewRC == OldRC) {
        LLVM_DEBUG(dbgs() << "User of " << printReg(Reg, &TRI)
                          << " cannot be reassigned to AGPR: " << *MI);
        return nullptr;
      }
    }
  }

  return NewRC;
}

/// Attempt to reassign the registers in \p InterferingRegs to be AGPRs, with a
/// preference to use \p PhysReg first. Returns false if the reassignments
/// cannot be trivially performed.
bool AMDGPURewriteAGPRCopyMFMAImpl::attemptReassignmentsToAGPR(
    SmallSetVector<Register, 4> &InterferingRegs, MCPhysReg PrefPhysReg) const {
  // FIXME: The ordering may matter here, but we're just taking uselistorder
  // with the special case of ensuring to process the starting instruction
  // first. We probably should extract the priority advisor out of greedy and
  // use that ordering.
  for (Register InterferingReg : InterferingRegs) {
    LiveInterval &ReassignLI = LIS.getInterval(InterferingReg);
    const TargetRegisterClass *EquivalentAGPRRegClass =
        TRI.getEquivalentAGPRClass(MRI.getRegClass(InterferingReg));

    MCPhysReg Assignable = AMDGPU::NoRegister;
    if (EquivalentAGPRRegClass->contains(PrefPhysReg) &&
        LRM.checkInterference(ReassignLI, PrefPhysReg) ==
            LiveRegMatrix::IK_Free) {
      // First try to assign to the AGPR we were already copying to. This
      // should be the first assignment we attempt. We have to guard
      // against the use being a subregister (which doesn't have an exact
      // class match).

      // TODO: If this does happen to be a subregister use, we should
      // still try to assign to a subregister of the original copy result.
      Assignable = PrefPhysReg;
    } else {
      ArrayRef<MCPhysReg> AllocOrder =
          RegClassInfo.getOrder(EquivalentAGPRRegClass);
      for (MCPhysReg Reg : AllocOrder) {
        if (LRM.checkInterference(ReassignLI, Reg) == LiveRegMatrix::IK_Free) {
          Assignable = Reg;
          break;
        }
      }
    }

    if (!Assignable) {
      LLVM_DEBUG(dbgs() << "Unable to reassign VGPR "
                        << printReg(InterferingReg, &TRI)
                        << " to a free AGPR\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "Reassigning VGPR " << printReg(InterferingReg, &TRI)
                      << " to " << printReg(Assignable, &TRI) << '\n');
    LRM.assign(ReassignLI, Assignable);
  }

  return true;
}

bool AMDGPURewriteAGPRCopyMFMAImpl::run(MachineFunction &MF) const {
  // This only applies on subtargets that have a configurable AGPR vs. VGPR
  // allocation.
  if (!ST.hasGFX90AInsts())
    return false;

  // Early exit if no AGPRs were assigned.
  if (!LRM.isPhysRegUsed(AMDGPU::AGPR0)) {
    LLVM_DEBUG(dbgs() << "skipping function that did not allocate AGPRs\n");
    return false;
  }

  bool MadeChange = false;

  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register VReg = Register::index2VirtReg(I);
    Register PhysReg = VRM.getPhys(VReg);
    if (!PhysReg)
      continue;

    // Find AV_* registers assigned to AGPRs.
    const TargetRegisterClass *VirtRegRC = MRI.getRegClass(VReg);
    if (!TRI.hasAGPRs(VirtRegRC))
      continue;

    const TargetRegisterClass *AssignedRC = VirtRegRC;
    if (TRI.hasVGPRs(VirtRegRC)) {
      // If this is an AV register, we have to check if the actual assignment is
      // to an AGPR
      AssignedRC = TRI.getPhysRegBaseClass(PhysReg);
      if (!TRI.isAGPRClass(AssignedRC))
        continue;
    }

    LiveInterval &LI = LIS.getInterval(VReg);

    for (VNInfo *VNI : LI.vnis()) {
      if (VNI->isPHIDef() || VNI->isUnused())
        continue;

      MachineInstr *DefMI = LIS.getInstructionFromIndex(VNI->def);

      // TODO: Handle SplitKit produced copy bundles for partially defined
      // registers.
      if (!DefMI || !DefMI->isFullCopy())
        continue;

      Register MFMADstReg = DefMI->getOperand(1).getReg();
      if (!MFMADstReg.isVirtual())
        continue;

      LiveInterval &CopySrcLI = LIS.getInterval(MFMADstReg);
      LiveQueryResult LRQ = CopySrcLI.Query(VNI->def.getRegSlot());
      MachineInstr *MFMA = LIS.getInstructionFromIndex(LRQ.valueIn()->def);
      if (!MFMA || !isRewriteCandidate(*MFMA))
        continue;

      MachineOperand *Src2 = TII.getNamedOperand(*MFMA, AMDGPU::OpName::src2);
      if (!Src2->isReg())
        continue;

      Register Src2Reg = Src2->getReg();
      if (!Src2Reg.isVirtual())
        continue;

      // FIXME: getMinimalPhysRegClass returns a nonsense AV_* subclass instead
      // of an AGPR or VGPR subclass, so we can't simply use the result on the
      // assignment.

      LLVM_DEBUG({
        dbgs() << "Attempting to replace VGPR MFMA with AGPR version:"
               << " Dst=[" << printReg(VReg) << " => "
               << printReg(PhysReg, &TRI) << ']';

        if (Src2Reg) {
          Register Src2PhysReg = VRM.getPhys(Src2Reg);
          dbgs() << ", Src2=[" << printReg(Src2Reg, &TRI) << " => "
                 << printReg(Src2PhysReg, &TRI) << "]: " << *MFMA;
        }
      });

      const TargetRegisterClass *DstVirtRegRC = MRI.getRegClass(MFMADstReg);

      // src2 and dst have the same physical class constraint; try to preserve
      // the original src2 subclass if one were to exist.
      SmallVector<MachineInstr *, 4> RewriteCandidates = {MFMA};
      SmallSetVector<Register, 4> RewriteRegs;

      // Make sure we reassign the MFMA we found the copy from first. We want
      // to ensure dst ends up in the physreg we were originally copying to.
      RewriteRegs.insert(MFMADstReg);

      // We've found av = COPY (MFMA), and need to verify that we can trivially
      // rewrite src2 to use the new AGPR. If we can't trivially replace it,
      // we're going to induce as many copies as we would have emitted in the
      // first place, as well as need to assign another register, and need to
      // figure out where to put them. The live range splitting is smarter than
      // anything we're doing here, so trust it did something reasonable.
      //
      // Note recomputeRegClassExceptRewritable will consider the constraints of
      // this MFMA's src2 as well as the src2/dst of any transitive MFMA users.
      const TargetRegisterClass *DstExceptRC =
          recomputeRegClassExceptRewritable(MFMADstReg, DstVirtRegRC, VirtRegRC,
                                            RewriteCandidates, RewriteRegs);
      if (!DstExceptRC) {
        LLVM_DEBUG(dbgs() << "Could not recompute the regclass of "
                          << printReg(MFMADstReg, &TRI) << '\n');
        continue;
      }

      // If src2 and dst are different registers, we need to also reassign the
      // input to an available AGPR if it is compatible with all other uses.
      //
      // If we can't reassign it, we'd need to introduce a different copy
      // which is likely worse than the copy we'd be saving.
      //
      // It's likely that the MFMA is used in sequence with other MFMAs; if we
      // cannot migrate the full use/def chain of MFMAs, we would need to
      // introduce intermediate copies somewhere. So we only make the
      // transform if all the interfering MFMAs can also be migrated. Collect
      // the set of rewritable MFMAs and check if we can assign an AGPR at
      // that point.
      //
      // If any of the MFMAs aren't reassignable, we give up and rollback to
      // the original register assignments.

      using RecoloringStack =
          SmallVector<std::pair<const LiveInterval *, MCRegister>, 8>;
      RecoloringStack TentativeReassignments;

      for (Register RewriteReg : RewriteRegs) {
        LiveInterval &LI = LIS.getInterval(RewriteReg);
        TentativeReassignments.push_back({&LI, VRM.getPhys(RewriteReg)});
        LRM.unassign(LI);
      }

      if (!attemptReassignmentsToAGPR(RewriteRegs, PhysReg)) {
        // Roll back the register assignments to the original state.
        for (auto [LI, OldAssign] : TentativeReassignments) {
          if (VRM.hasPhys(LI->reg()))
            LRM.unassign(*LI);
          LRM.assign(*LI, OldAssign);
        }

        continue;
      }

      // Fixup the register classes of the virtual registers now that we've
      // committed to the reassignments.
      for (Register InterferingReg : RewriteRegs) {
        const TargetRegisterClass *EquivalentAGPRRegClass =
            TRI.getEquivalentAGPRClass(MRI.getRegClass(InterferingReg));
        MRI.setRegClass(InterferingReg, EquivalentAGPRRegClass);
      }

      for (MachineInstr *RewriteCandidate : RewriteCandidates) {
        int NewMFMAOp =
            AMDGPU::getMFMASrcCVDstAGPROp(RewriteCandidate->getOpcode());
        RewriteCandidate->setDesc(TII.get(NewMFMAOp));
      }

      // We likely left an identity copy behind after assignment; let
      // VirtRegRewriter deal with it later.
      MadeChange = true;
    }
  }

  return MadeChange;
}

class AMDGPURewriteAGPRCopyMFMALegacy : public MachineFunctionPass {
public:
  static char ID;
  RegisterClassInfo RegClassInfo;

  AMDGPURewriteAGPRCopyMFMALegacy() : MachineFunctionPass(ID) {
    initializeAMDGPURewriteAGPRCopyMFMALegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Rewrite AGPR-Copy-MFMA";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>();

    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<VirtRegMapWrapperLegacy>();
    AU.addPreserved<LiveRegMatrixWrapperLegacy>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPURewriteAGPRCopyMFMALegacy, DEBUG_TYPE,
                      "AMDGPU Rewrite AGPR-Copy-MFMA", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_END(AMDGPURewriteAGPRCopyMFMALegacy, DEBUG_TYPE,
                    "AMDGPU Rewrite AGPR-Copy-MFMA", false, false)

char AMDGPURewriteAGPRCopyMFMALegacy::ID = 0;

char &llvm::AMDGPURewriteAGPRCopyMFMALegacyID =
    AMDGPURewriteAGPRCopyMFMALegacy::ID;

bool AMDGPURewriteAGPRCopyMFMALegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  RegClassInfo.runOnMachineFunction(MF);

  auto &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  auto &LRM = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();

  AMDGPURewriteAGPRCopyMFMAImpl Impl(MF, VRM, LRM, LIS, RegClassInfo);
  return Impl.run(MF);
}

PreservedAnalyses
AMDGPURewriteAGPRCopyMFMAPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  VirtRegMap &VRM = MFAM.getResult<VirtRegMapAnalysis>(MF);
  LiveRegMatrix &LRM = MFAM.getResult<LiveRegMatrixAnalysis>(MF);
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  RegisterClassInfo RegClassInfo;
  RegClassInfo.runOnMachineFunction(MF);

  AMDGPURewriteAGPRCopyMFMAImpl Impl(MF, VRM, LRM, LIS, RegClassInfo);
  if (!Impl.run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
