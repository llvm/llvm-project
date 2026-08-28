//===-- SIPreEmitPeephole.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass performs the peephole optimizations before code emission.
///
/// Additionally, this pass also unpacks packed instructions (V_PK_MUL_F32/F16,
/// V_PK_ADD_F32/F16, V_PK_FMA_F32) adjacent to MFMAs such that they can be
/// co-issued. This helps with overlapping MFMA and certain vector instructions
/// in machine schedules and is expected to improve performance. Only those
/// packed instructions are unpacked that are overlapped by the MFMA latency.
/// Rest should remain untouched.
/// TODO: Add support for F16 packed instructions
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/ReachingDefAnalysis.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Support/BranchProbability.h"
using namespace llvm;

#define DEBUG_TYPE "si-pre-emit-peephole"

STATISTIC(NumModeWritesRemoved,
          "Number of redundant mode register writes removed");

namespace {

/// The state of one independent field of the MODE register, as tracked by
/// removeRedundantModeWrites.
struct ModeFieldState {
  std::optional<int64_t> Value;
  std::optional<int64_t> ValueBeforePendingWrite;
  MachineInstr *PendingWrite = nullptr;

  bool isTracked() const { return PendingWrite || Value; }
};

class SIPreEmitPeephole {
private:
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineLoopInfo *MLI = nullptr;
  const ReachingDefInfo *RDI = nullptr;

  bool optimizeVccBranch(MachineInstr &MI) const;
  void updateMLIBeforeRemovingEdge(MachineBasicBlock *From,
                                   MachineBasicBlock *To) const;
  bool optimizeSetGPR(MachineInstr &First, MachineInstr &MI) const;
  bool getBlockDestinations(MachineBasicBlock &SrcMBB,
                            MachineBasicBlock *&TrueMBB,
                            MachineBasicBlock *&FalseMBB,
                            SmallVectorImpl<MachineOperand> &Cond);
  bool mustRetainExeczBranch(const MachineInstr &Branch,
                             const MachineBasicBlock &From,
                             const MachineBasicBlock &To) const;
  bool removeExeczBranch(MachineInstr &MI, MachineBasicBlock &SrcMBB);
  bool removeRedundantModeWrites(MachineBasicBlock &SrcMBB) const;
  // Creates a list of packed instructions following an MFMA that are suitable
  // for unpacking.
  void collectUnpackingCandidates(MachineInstr &BeginMI,
                                  SetVector<MachineInstr *> &InstrsToUnpack,
                                  uint16_t NumMFMACycles);
  // v_pk_fma_f32 v[0:1], v[0:1], v[2:3], v[2:3] op_sel:[1,1,1]
  // op_sel_hi:[0,0,0]
  // ==>
  // v_fma_f32 v0, v1, v3, v3
  // v_fma_f32 v1, v0, v2, v2
  // Here, we have overwritten v0 before we use it. This function checks if
  // unpacking can lead to such a situation.
  bool canUnpackingClobberRegister(const MachineInstr &MI);
  // Unpack and insert F32 packed instructions, such as V_PK_MUL, V_PK_ADD, and
  // V_PK_FMA. Currently, only V_PK_MUL, V_PK_ADD, V_PK_FMA are supported for
  // this transformation.
  void performF32Unpacking(MachineInstr &I);
  // Select corresponding unpacked instruction
  uint32_t mapToUnpackedOpcode(MachineInstr &I);
  // Creates the unpacked instruction to be inserted. Adds source modifiers to
  // the unpacked instructions based on the source modifiers in the packed
  // instruction.
  MachineInstrBuilder createUnpackedMI(MachineInstr &I, uint32_t UnpackedOpcode,
                                       bool IsHiBits);
  // Process operands/source modifiers from packed instructions and insert the
  // appropriate source modifers and operands into the unpacked instructions.
  void addOperandAndMods(MachineInstrBuilder &NewMI, unsigned SrcMods,
                         bool IsHiBits, const MachineOperand &SrcMO);

  // Hold information about a candidate for delayed post-RA conversion from a
  // two-address instruction to a three-address instruction. The copy
  // instructions become redundant after retargeting the converted instruction
  // to NewDst.
  struct ThreeAddressCandidate {
    MachineInstr *TwoAddrMI;
    Register NewDst;
    SmallVector<MachineInstr *, 2> CopiesToRemove;

    ThreeAddressCandidate(MachineInstr &TwoAddrMI, Register NewDst,
                          ArrayRef<MachineInstr *> Copies)
        : TwoAddrMI(&TwoAddrMI), NewDst(NewDst), CopiesToRemove(Copies) {}
  };

  // Return true if MI is a candidate for late conversion to a three-address
  // instruction to avoid copies.
  bool isLateThreeAddrPeepholeCandidate(const MachineInstr &MI) const;

  // Check if MaybeCopy is a simple copy of Src. If NewDstRC is non-null, also
  // require the copy destination to belong to that register class. The class
  // check is omitted for copies of individual subregisters; their reconstructed
  // super-register is checked against the destination class by the caller.
  // Return the destination register of a supported copy.
  Register checkCopy(MachineInstr &TwoAddress, MachineInstr &MaybeCopy,
                     Register Src, const TargetRegisterClass *NewDstRC);

  // Check whether the two-address instruction should be replaced with a
  // three-address instruction to avoid copies of the output registers. If the
  // replacement should take place, returns the destination register for the
  // replacement and fills Copies with the copy instructions to remove after
  // the replacement.
  Register
  shouldReplaceWithThreeAddress(MachineInstr &MI,
                                SmallVectorImpl<MachineInstr *> &Copies);

  // Convert two-adress instruction in Cand to its three-address equivalent,
  // targeting the register given in Cand.
  bool convertToThreeAddressInstr(ThreeAddressCandidate &Cand);

public:
  bool run(MachineFunction &MF, MachineLoopInfo *MLI,
           const ReachingDefInfo *RDI);
};

class SIPreEmitPeepholeLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIPreEmitPeepholeLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addUsedIfAvailable<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.addRequired<ReachingDefInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto *MLIWrapper = getAnalysisIfAvailable<MachineLoopInfoWrapperPass>();
    MachineLoopInfo *MLI = MLIWrapper ? &MLIWrapper->getLI() : nullptr;
    const ReachingDefInfo *RDI =
        &getAnalysis<ReachingDefInfoWrapperPass>().getRDI();
    return SIPreEmitPeephole().run(MF, MLI, RDI);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS(SIPreEmitPeepholeLegacy, DEBUG_TYPE,
                "SI peephole optimizations", false, false)

char SIPreEmitPeepholeLegacy::ID = 0;

char &llvm::SIPreEmitPeepholeID = SIPreEmitPeepholeLegacy::ID;

void SIPreEmitPeephole::updateMLIBeforeRemovingEdge(
    MachineBasicBlock *From, MachineBasicBlock *To) const {
  if (!MLI)
    return;

  // Only handle back-edges: To must be a loop header with From inside the loop.
  MachineLoop *Loop = MLI->getLoopFor(To);
  if (!Loop || Loop->getHeader() != To || !Loop->contains(From))
    return;

  // Count back-edges
  unsigned BackEdgeCount = 0;
  for (MachineBasicBlock *Pred : To->predecessors()) {
    if (Loop->contains(Pred))
      BackEdgeCount++;
  }

  if (BackEdgeCount > 1)
    return;

  MachineLoop *ParentLoop = Loop->getParentLoop();

  // Re-map blocks directly owned by this loop to the parent.
  for (MachineBasicBlock *BB : Loop->blocks()) {
    if (MLI->getLoopFor(BB) == Loop)
      MLI->changeLoopFor(BB, ParentLoop);
  }

  // Reparent all child loops.
  while (!Loop->isInnermost()) {
    MachineLoop *Child = Loop->removeChildLoop(std::prev(Loop->end()));
    if (ParentLoop)
      ParentLoop->addChildLoop(Child);
    else
      MLI->addTopLevelLoop(Child);
  }

  if (ParentLoop)
    ParentLoop->removeChildLoop(Loop);
  else
    MLI->removeLoop(llvm::find(*MLI, Loop));

  MLI->destroy(Loop);
}

bool SIPreEmitPeephole::optimizeVccBranch(MachineInstr &MI) const {
  // Match:
  // sreg = -1 or 0
  // vcc = S_AND_B64 exec, sreg or S_ANDN2_B64 exec, sreg
  // S_CBRANCH_VCC[N]Z
  // =>
  // S_CBRANCH_EXEC[N]Z
  // We end up with this pattern sometimes after basic block placement.
  // It happens while combining a block which assigns -1 or 0 to a saved mask
  // and another block which consumes that saved mask and then a branch.
  //
  // While searching this also performs the following substitution:
  // vcc = V_CMP
  // vcc = S_AND exec, vcc
  // S_CBRANCH_VCC[N]Z
  // =>
  // vcc = V_CMP
  // S_CBRANCH_VCC[N]Z

  bool Changed = false;
  MachineBasicBlock &MBB = *MI.getParent();
  const GCNSubtarget &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();
  const bool IsWave32 = ST.isWave32();
  const unsigned CondReg = TRI->getVCC();
  const unsigned ExecReg = IsWave32 ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
  const unsigned And = IsWave32 ? AMDGPU::S_AND_B32 : AMDGPU::S_AND_B64;
  const unsigned AndN2 = IsWave32 ? AMDGPU::S_ANDN2_B32 : AMDGPU::S_ANDN2_B64;
  const unsigned Mov = IsWave32 ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;

  MachineBasicBlock::reverse_iterator A = MI.getReverseIterator(),
                                      E = MBB.rend();
  bool ReadsCond = false;
  unsigned Threshold = 5;
  for (++A; A != E; ++A) {
    if (!--Threshold)
      return false;
    if (A->modifiesRegister(ExecReg, TRI))
      return false;
    if (A->modifiesRegister(CondReg, TRI)) {
      if (!A->definesRegister(CondReg, TRI) ||
          (A->getOpcode() != And && A->getOpcode() != AndN2))
        return false;
      break;
    }
    ReadsCond |= A->readsRegister(CondReg, TRI);
  }
  if (A == E)
    return false;

  MachineOperand &Op1 = A->getOperand(1);
  MachineOperand &Op2 = A->getOperand(2);
  if ((!Op1.isReg() || Op1.getReg() != ExecReg) && Op2.isReg() &&
      Op2.getReg() == ExecReg) {
    TII->commuteInstruction(*A);
    Changed = true;
  }
  if (!Op1.isReg() || Op1.getReg() != ExecReg)
    return Changed;
  if (Op2.isImm() && !(Op2.getImm() == -1 || Op2.getImm() == 0))
    return Changed;

  int64_t MaskValue = 0;
  Register SReg;
  if (Op2.isReg()) {
    SReg = Op2.getReg();
    auto M = std::next(A);
    bool ReadsSreg = false;
    bool ModifiesExec = false;
    for (; M != E; ++M) {
      if (M->definesRegister(SReg, TRI))
        break;
      if (M->modifiesRegister(SReg, TRI))
        return Changed;
      ReadsSreg |= M->readsRegister(SReg, TRI);
      ModifiesExec |= M->modifiesRegister(ExecReg, TRI);
    }
    if (M == E)
      return Changed;
    // If SReg is VCC and SReg definition is a VALU comparison.
    // This means S_AND with EXEC is not required, unless
    // the implicit def of SCC is alive.
    // Erase the S_AND and return.
    // Note: isVOPC is used instead of isCompare to catch V_CMP_CLASS
    if (A->getOpcode() == And && SReg == CondReg && !ModifiesExec &&
        TII->isVOPC(*M) && A->allImplicitDefsAreDead()) {
      A->eraseFromParent();
      return true;
    }

    if (!M->isMoveImmediate() || !M->getOperand(1).isImm() ||
        (M->getOperand(1).getImm() != -1 && M->getOperand(1).getImm() != 0))
      return Changed;
    MaskValue = M->getOperand(1).getImm();
    // First if sreg is only used in the AND instruction fold the immediate
    // into the AND.
    if (!ReadsSreg && Op2.isKill()) {
      A->getOperand(2).ChangeToImmediate(MaskValue);
      M->eraseFromParent();
    }
  } else if (Op2.isImm()) {
    MaskValue = Op2.getImm();
  } else {
    llvm_unreachable("Op2 must be register or immediate");
  }

  // Invert mask for s_andn2
  assert(MaskValue == 0 || MaskValue == -1);
  if (A->getOpcode() == AndN2)
    MaskValue = ~MaskValue;

  if (!ReadsCond && A->registerDefIsDead(AMDGPU::SCC, /*TRI=*/nullptr)) {
    if (!MI.killsRegister(CondReg, TRI)) {
      // Replace AND with MOV
      if (MaskValue == 0) {
        BuildMI(*A->getParent(), *A, A->getDebugLoc(), TII->get(Mov), CondReg)
            .addImm(0);
      } else {
        BuildMI(*A->getParent(), *A, A->getDebugLoc(), TII->get(Mov), CondReg)
            .addReg(ExecReg);
      }
    }
    // Remove AND instruction
    A->eraseFromParent();
  }

  bool IsVCCZ = MI.getOpcode() == AMDGPU::S_CBRANCH_VCCZ;
  if (SReg == ExecReg) {
    // EXEC is updated directly
    if (IsVCCZ) {
      MI.eraseFromParent();
      return true;
    }
    MI.setDesc(TII->get(AMDGPU::S_BRANCH));
  } else if (IsVCCZ && MaskValue == 0) {
    // Will always branch
    // Remove all successors shadowed by new unconditional branch
    MachineBasicBlock *Parent = MI.getParent();
    SmallVector<MachineInstr *, 4> ToRemove;
    bool Found = false;
    for (MachineInstr &Term : Parent->terminators()) {
      if (Found) {
        if (Term.isBranch())
          ToRemove.push_back(&Term);
      } else {
        Found = Term.isIdenticalTo(MI);
      }
    }
    assert(Found && "conditional branch is not terminator");
    for (auto *BranchMI : ToRemove) {
      MachineOperand &Dst = BranchMI->getOperand(0);
      assert(Dst.isMBB() && "destination is not basic block");
      updateMLIBeforeRemovingEdge(Parent, Dst.getMBB());
      Parent->removeSuccessor(Dst.getMBB());
      BranchMI->eraseFromParent();
    }

    if (MachineBasicBlock *Succ = Parent->getFallThrough()) {
      updateMLIBeforeRemovingEdge(Parent, Succ);
      Parent->removeSuccessor(Succ);
    }

    // Rewrite to unconditional branch
    MI.setDesc(TII->get(AMDGPU::S_BRANCH));
  } else if (!IsVCCZ && MaskValue == 0) {
    // Will never branch
    MachineOperand &Dst = MI.getOperand(0);
    assert(Dst.isMBB() && "destination is not basic block");
    MachineBasicBlock *Parent = MI.getParent();
    updateMLIBeforeRemovingEdge(Parent, Dst.getMBB());
    Parent->removeSuccessor(Dst.getMBB());
    MI.eraseFromParent();
    return true;
  } else if (MaskValue == -1) {
    // Depends only on EXEC
    MI.setDesc(
        TII->get(IsVCCZ ? AMDGPU::S_CBRANCH_EXECZ : AMDGPU::S_CBRANCH_EXECNZ));
  }

  MI.removeOperand(MI.findRegisterUseOperandIdx(CondReg, TRI, false /*Kill*/));
  MI.addImplicitDefUseOperands(*MBB.getParent());

  return true;
}

bool SIPreEmitPeephole::optimizeSetGPR(MachineInstr &First,
                                       MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();
  const MachineFunction &MF = *MBB.getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
  Register IdxReg = Idx->isReg() ? Idx->getReg() : Register();
  SmallVector<MachineInstr *, 4> ToRemove;
  bool IdxOn = true;

  if (!MI.isIdenticalTo(First))
    return false;

  // Scan back to find an identical S_SET_GPR_IDX_ON
  for (MachineBasicBlock::instr_iterator I = std::next(First.getIterator()),
                                         E = MI.getIterator();
       I != E; ++I) {
    if (I->isBundle() || I->isDebugInstr())
      continue;
    switch (I->getOpcode()) {
    case AMDGPU::S_SET_GPR_IDX_MODE:
      return false;
    case AMDGPU::S_SET_GPR_IDX_OFF:
      IdxOn = false;
      ToRemove.push_back(&*I);
      break;
    default:
      if (I->modifiesRegister(AMDGPU::M0, TRI))
        return false;
      if (IdxReg && I->modifiesRegister(IdxReg, TRI))
        return false;
      if (llvm::any_of(I->operands(), [&MRI, this](const MachineOperand &MO) {
            return MO.isReg() && TRI->isVectorRegister(MRI, MO.getReg());
          })) {
        // The only exception allowed here is another indirect vector move
        // with the same mode.
        if (!IdxOn || !(I->getOpcode() == AMDGPU::V_MOV_B32_indirect_write ||
                        I->getOpcode() == AMDGPU::V_MOV_B32_indirect_read))
          return false;
      }
    }
  }

  MI.eraseFromBundle();
  for (MachineInstr *RI : ToRemove)
    RI->eraseFromBundle();
  return true;
}

bool SIPreEmitPeephole::getBlockDestinations(
    MachineBasicBlock &SrcMBB, MachineBasicBlock *&TrueMBB,
    MachineBasicBlock *&FalseMBB, SmallVectorImpl<MachineOperand> &Cond) {
  if (TII->analyzeBranch(SrcMBB, TrueMBB, FalseMBB, Cond))
    return false;

  if (!FalseMBB)
    FalseMBB = SrcMBB.getNextNode();

  return true;
}

namespace {
class BranchWeightCostModel {
  const SIInstrInfo &TII;
  const TargetSchedModel &SchedModel;
  BranchProbability BranchProb;
  static constexpr uint64_t BranchNotTakenCost = 1;
  uint64_t BranchTakenCost;
  uint64_t ThenCyclesCost = 0;

public:
  BranchWeightCostModel(const SIInstrInfo &TII, const MachineInstr &Branch,
                        const MachineBasicBlock &Succ)
      : TII(TII), SchedModel(TII.getSchedModel()) {
    const MachineBasicBlock &Head = *Branch.getParent();
    const auto *FromIt = find(Head.successors(), &Succ);
    assert(FromIt != Head.succ_end());

    BranchProb = Head.getSuccProbability(FromIt);
    if (BranchProb.isUnknown())
      BranchProb = BranchProbability::getZero();
    BranchTakenCost = SchedModel.computeInstrLatency(&Branch);
  }

  bool isProfitable(const MachineInstr &MI) {
    if (TII.isWaitcnt(MI.getOpcode()))
      return false;

    ThenCyclesCost += SchedModel.computeInstrLatency(&MI);

    // Consider `P = N/D` to be the probability of execz being false (skipping
    // the then-block) The transformation is profitable if always executing the
    // 'then' block is cheaper than executing sometimes 'then' and always
    // executing s_cbranch_execz:
    // * ThenCost <= P*ThenCost + (1-P)*BranchTakenCost + P*BranchNotTakenCost
    // * (1-P) * ThenCost <= (1-P)*BranchTakenCost + P*BranchNotTakenCost
    // * (D-N)/D * ThenCost <= (D-N)/D * BranchTakenCost + N/D *
    // BranchNotTakenCost
    uint64_t Numerator = BranchProb.getNumerator();
    uint64_t Denominator = BranchProb.getDenominator();
    return (Denominator - Numerator) * ThenCyclesCost <=
           ((Denominator - Numerator) * BranchTakenCost +
            Numerator * BranchNotTakenCost);
  }
};

bool SIPreEmitPeephole::mustRetainExeczBranch(
    const MachineInstr &Branch, const MachineBasicBlock &From,
    const MachineBasicBlock &To) const {
  assert(is_contained(Branch.getParent()->successors(), &From));
  BranchWeightCostModel CostModel{*TII, Branch, From};

  const MachineFunction *MF = From.getParent();
  for (MachineFunction::const_iterator MBBI(&From), ToI(&To), End = MF->end();
       MBBI != End && MBBI != ToI; ++MBBI) {
    const MachineBasicBlock &MBB = *MBBI;

    for (const MachineInstr &MI : MBB) {
      // When a uniform loop is inside non-uniform control flow, the branch
      // leaving the loop might never be taken when EXEC = 0.
      // Hence we should retain cbranch out of the loop lest it become infinite.
      if (MI.isConditionalBranch())
        return true;

      if (MI.isUnconditionalBranch() &&
          TII->getBranchDestBlock(MI) != MBB.getNextNode())
        return true;

      if (MI.isMetaInstruction())
        continue;

      if (TII->hasUnwantedEffectsWhenEXECEmpty(MI))
        return true;

      if (!CostModel.isProfitable(MI))
        return true;
    }
  }

  return false;
}
} // namespace

// Returns true if the skip branch instruction is removed.
bool SIPreEmitPeephole::removeExeczBranch(MachineInstr &MI,
                                          MachineBasicBlock &SrcMBB) {

  if (!TII->getSchedModel().hasInstrSchedModel())
    return false;

  MachineBasicBlock *TrueMBB = nullptr;
  MachineBasicBlock *FalseMBB = nullptr;
  SmallVector<MachineOperand, 1> Cond;

  if (!getBlockDestinations(SrcMBB, TrueMBB, FalseMBB, Cond))
    return false;

  // Consider only the forward branches.
  if (SrcMBB.getNumber() >= TrueMBB->getNumber())
    return false;

  // Consider only when it is legal and profitable
  if (mustRetainExeczBranch(MI, *FalseMBB, *TrueMBB))
    return false;

  LLVM_DEBUG(dbgs() << "Removing the execz branch: " << MI);
  MI.eraseFromParent();
  SrcMBB.removeSuccessor(TrueMBB);

  return true;
}

/// Remove writes to the FP round mode and FP denorm mode that can never be
/// observed: either the value written is already live in MODE, or a mode write
/// replaces the whole mode field before anything reads it.
///
/// s_round_mode and s_denorm_mode each assign one field of MODE and preserve
/// the rest of the register, so the two fields are tracked independently and a
/// write to one is transparent to the other.
///
/// This is a purely intra-block analysis: the mode on entry to \p SrcMBB is
/// unknown, and a write that is still live at the end of the block is kept for
/// the benefit of the successors.
bool SIPreEmitPeephole::removeRedundantModeWrites(
    MachineBasicBlock &SrcMBB) const {
  bool Changed = false;
  ModeFieldState DenormMode;
  ModeFieldState RoundMode;

  for (MachineInstr &MI : make_early_inc_range(SrcMBB)) {
    if (MI.isDebugInstr())
      continue;

    unsigned Opc = MI.getOpcode();
    if (Opc == AMDGPU::S_DENORM_MODE || Opc == AMDGPU::S_ROUND_MODE) {
      ModeFieldState &Field =
          Opc == AMDGPU::S_DENORM_MODE ? DenormMode : RoundMode;
      int64_t NewValue = MI.getOperand(0).getImm();

      if (Field.PendingWrite) {
        LLVM_DEBUG(dbgs() << "Removing dead mode write: "
                          << *Field.PendingWrite);
        Field.PendingWrite->eraseFromParent();
        ++NumModeWritesRemoved;
        Changed = true;
        Field.PendingWrite = nullptr;
        Field.Value = Field.ValueBeforePendingWrite;
      }

      if (Field.Value == NewValue) {
        LLVM_DEBUG(dbgs() << "Removing redundant mode write: " << MI);
        MI.eraseFromParent();
        ++NumModeWritesRemoved;
        Changed = true;
        continue;
      }

      Field.ValueBeforePendingWrite = Field.Value;
      Field.PendingWrite = &MI;
      Field.Value = NewValue;
      continue;
    }

    // Nothing tracked yet; skip register checks below.
    if (!DenormMode.isTracked() && !RoundMode.isTracked())
      continue;

    // Inline asm cannot declare a MODE clobber, so assume it writes both.
    if (MI.isInlineAsm() || MI.modifiesRegister(AMDGPU::MODE, TRI)) {
      DenormMode = ModeFieldState();
      RoundMode = ModeFieldState();
      continue;
    }

    if (MI.readsRegister(AMDGPU::MODE, TRI) || MI.hasUnmodeledSideEffects()) {
      DenormMode.PendingWrite = nullptr;
      RoundMode.PendingWrite = nullptr;
    }
  }
  return Changed;
}

bool SIPreEmitPeephole::canUnpackingClobberRegister(const MachineInstr &MI) {
  unsigned OpCode = MI.getOpcode();
  Register DstReg = MI.getOperand(0).getReg();
  // Only the first register in the register pair needs to be checked due to the
  // unpacking order. Packed instructions are unpacked such that the lower 32
  // bits (i.e., the first register in the pair) are written first. This can
  // introduce dependencies if the first register is written in one instruction
  // and then read as part of the higher 32 bits in the subsequent instruction.
  // Such scenarios can arise due to specific combinations of op_sel and
  // op_sel_hi modifiers.
  Register UnpackedDstReg = TRI->getSubReg(DstReg, AMDGPU::sub0);

  const MachineOperand *Src0MO = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
  if (Src0MO && Src0MO->isReg()) {
    Register SrcReg0 = Src0MO->getReg();
    unsigned Src0Mods =
        TII->getNamedOperand(MI, AMDGPU::OpName::src0_modifiers)->getImm();
    Register HiSrc0Reg = (Src0Mods & SISrcMods::OP_SEL_1)
                             ? TRI->getSubReg(SrcReg0, AMDGPU::sub1)
                             : TRI->getSubReg(SrcReg0, AMDGPU::sub0);
    // Check if the register selected by op_sel_hi is the same as the first
    // register in the destination register pair.
    if (TRI->regsOverlap(UnpackedDstReg, HiSrc0Reg))
      return true;
  }

  const MachineOperand *Src1MO = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
  if (Src1MO && Src1MO->isReg()) {
    Register SrcReg1 = Src1MO->getReg();
    unsigned Src1Mods =
        TII->getNamedOperand(MI, AMDGPU::OpName::src1_modifiers)->getImm();
    Register HiSrc1Reg = (Src1Mods & SISrcMods::OP_SEL_1)
                             ? TRI->getSubReg(SrcReg1, AMDGPU::sub1)
                             : TRI->getSubReg(SrcReg1, AMDGPU::sub0);
    if (TRI->regsOverlap(UnpackedDstReg, HiSrc1Reg))
      return true;
  }

  // Applicable for packed instructions with 3 source operands, such as
  // V_PK_FMA.
  if (AMDGPU::hasNamedOperand(OpCode, AMDGPU::OpName::src2)) {
    const MachineOperand *Src2MO =
        TII->getNamedOperand(MI, AMDGPU::OpName::src2);
    if (Src2MO && Src2MO->isReg()) {
      Register SrcReg2 = Src2MO->getReg();
      unsigned Src2Mods =
          TII->getNamedOperand(MI, AMDGPU::OpName::src2_modifiers)->getImm();
      Register HiSrc2Reg = (Src2Mods & SISrcMods::OP_SEL_1)
                               ? TRI->getSubReg(SrcReg2, AMDGPU::sub1)
                               : TRI->getSubReg(SrcReg2, AMDGPU::sub0);
      if (TRI->regsOverlap(UnpackedDstReg, HiSrc2Reg))
        return true;
    }
  }
  return false;
}

uint32_t SIPreEmitPeephole::mapToUnpackedOpcode(MachineInstr &I) {
  unsigned Opcode = I.getOpcode();
  // Use 64 bit encoding to allow use of VOP3 instructions.
  // VOP3 e64 instructions allow source modifiers
  // e32 instructions don't allow source modifiers.
  switch (Opcode) {
  case AMDGPU::V_PK_ADD_F32:
  case AMDGPU::V_PK_ADD_F32_gfx1250:
    return AMDGPU::V_ADD_F32_e64;
  case AMDGPU::V_PK_MUL_F32:
  case AMDGPU::V_PK_MUL_F32_gfx1250:
    return AMDGPU::V_MUL_F32_e64;
  case AMDGPU::V_PK_FMA_F32:
  case AMDGPU::V_PK_FMA_F32_gfx1250:
    return AMDGPU::V_FMA_F32_e64;
  default:
    return std::numeric_limits<uint32_t>::max();
  }
  llvm_unreachable("Fully covered switch");
}

void SIPreEmitPeephole::addOperandAndMods(MachineInstrBuilder &NewMI,
                                          unsigned SrcMods, bool IsHiBits,
                                          const MachineOperand &SrcMO) {
  unsigned NewSrcMods = 0;
  unsigned NegModifier = IsHiBits ? SISrcMods::NEG_HI : SISrcMods::NEG;
  unsigned OpSelModifier = IsHiBits ? SISrcMods::OP_SEL_1 : SISrcMods::OP_SEL_0;
  // Packed instructions (VOP3P) do not support ABS. Hence, no checks are done
  // for ABS modifiers.
  // If NEG or NEG_HI is true, we need to negate the corresponding 32 bit
  // lane.
  // NEG_HI shares the same bit position with ABS. But packed instructions do
  // not support ABS. Therefore, NEG_HI must be translated to NEG source
  // modifier for the higher 32 bits. Unpacked VOP3 instructions support
  // ABS, but do not support NEG_HI. Therefore we need to explicitly add the
  // NEG modifier if present in the packed instruction.
  if (SrcMods & NegModifier)
    NewSrcMods |= SISrcMods::NEG;
  // Src modifiers. Only negative modifiers are added if needed. Unpacked
  // operations do not have op_sel, therefore it must be handled explicitly as
  // done below.
  NewMI.addImm(NewSrcMods);
  if (SrcMO.isImm()) {
    NewMI.addImm(SrcMO.getImm());
    return;
  }
  // If op_sel == 0, select register 0 of reg:sub0_sub1.
  Register UnpackedSrcReg = (SrcMods & OpSelModifier)
                                ? TRI->getSubReg(SrcMO.getReg(), AMDGPU::sub1)
                                : TRI->getSubReg(SrcMO.getReg(), AMDGPU::sub0);

  MachineOperand UnpackedSrcMO =
      MachineOperand::CreateReg(UnpackedSrcReg, /*isDef=*/false);
  if (SrcMO.isKill()) {
    // For each unpacked instruction, mark its source registers as killed if the
    // corresponding source register in the original packed instruction was
    // marked as killed.
    //
    // Exception:
    // If the op_sel and op_sel_hi modifiers require both unpacked instructions
    // to use the same register (e.g., due to overlapping access to low/high
    // bits of the same packed register), then only the *second* (latter)
    // instruction should mark the register as killed. This is because the
    // second instruction handles the higher bits and is effectively the last
    // user of the full register pair.

    bool OpSel = SrcMods & SISrcMods::OP_SEL_0;
    bool OpSelHi = SrcMods & SISrcMods::OP_SEL_1;
    bool KillState = true;
    if ((OpSel == OpSelHi) && !IsHiBits)
      KillState = false;
    UnpackedSrcMO.setIsKill(KillState);
  }
  NewMI.add(UnpackedSrcMO);
}

void SIPreEmitPeephole::collectUnpackingCandidates(
    MachineInstr &BeginMI, SetVector<MachineInstr *> &InstrsToUnpack,
    uint16_t NumMFMACycles) {
  auto *BB = BeginMI.getParent();
  auto E = BB->end();
  int TotalCyclesBetweenCandidates = 0;
  auto SchedModel = TII->getSchedModel();
  Register MFMADef = BeginMI.getOperand(0).getReg();

  for (auto I = std::next(BeginMI.getIterator()); I != E; ++I) {
    MachineInstr &Instr = *I;
    uint32_t UnpackedOpCode = mapToUnpackedOpcode(Instr);
    bool IsUnpackable =
        !(UnpackedOpCode == std::numeric_limits<uint32_t>::max());
    if (Instr.isMetaInstruction())
      continue;
    if ((Instr.isTerminator()) ||
        (TII->isNeverCoissue(Instr) && !IsUnpackable) ||
        (SIInstrInfo::modifiesModeRegister(Instr) &&
         Instr.modifiesRegister(AMDGPU::EXEC, TRI)))
      return;

    const MCSchedClassDesc *InstrSchedClassDesc =
        SchedModel.resolveSchedClass(&Instr);
    uint16_t Latency =
        SchedModel.getWriteProcResBegin(InstrSchedClassDesc)->ReleaseAtCycle;
    TotalCyclesBetweenCandidates += Latency;

    if (TotalCyclesBetweenCandidates >= NumMFMACycles - 1)
      return;
    // Identify register dependencies between those used by the MFMA
    // instruction and the following packed instructions. Also checks for
    // transitive dependencies between the MFMA def and candidate instruction
    // def and uses. Conservatively ensures that we do not incorrectly
    // read/write registers.
    for (const MachineOperand &InstrMO : Instr.operands()) {
      if (!InstrMO.isReg() || !InstrMO.getReg().isValid())
        continue;
      if (TRI->regsOverlap(MFMADef, InstrMO.getReg()))
        return;
    }
    if (!IsUnpackable)
      continue;

    if (canUnpackingClobberRegister(Instr))
      return;
    // If it's a packed instruction, adjust latency: remove the packed
    // latency, add latency of two unpacked instructions (currently estimated
    // as 2 cycles).
    TotalCyclesBetweenCandidates -= Latency;
    // TODO: improve latency handling based on instruction modeling.
    TotalCyclesBetweenCandidates += 2;
    // Subtract 1 to account for MFMA issue latency.
    if (TotalCyclesBetweenCandidates < NumMFMACycles - 1)
      InstrsToUnpack.insert(&Instr);
  }
}

void SIPreEmitPeephole::performF32Unpacking(MachineInstr &I) {
  const MachineOperand &DstOp = I.getOperand(0);

  uint32_t UnpackedOpcode = mapToUnpackedOpcode(I);
  assert(UnpackedOpcode != std::numeric_limits<uint32_t>::max() &&
         "Unsupported Opcode");

  MachineInstrBuilder Op0LOp1L =
      createUnpackedMI(I, UnpackedOpcode, /*IsHiBits=*/false);
  MachineOperand LoDstOp = Op0LOp1L->getOperand(0);

  LoDstOp.setIsUndef(DstOp.isUndef());

  MachineInstrBuilder Op0HOp1H =
      createUnpackedMI(I, UnpackedOpcode, /*IsHiBits=*/true);
  MachineOperand HiDstOp = Op0HOp1H->getOperand(0);

  uint32_t IFlags = I.getFlags();
  Op0LOp1L->setFlags(IFlags);
  Op0HOp1H->setFlags(IFlags);
  LoDstOp.setIsRenamable(DstOp.isRenamable());
  HiDstOp.setIsRenamable(DstOp.isRenamable());

  I.eraseFromParent();
}

MachineInstrBuilder SIPreEmitPeephole::createUnpackedMI(MachineInstr &I,
                                                        uint32_t UnpackedOpcode,
                                                        bool IsHiBits) {
  MachineBasicBlock &MBB = *I.getParent();
  const DebugLoc &DL = I.getDebugLoc();
  const MachineOperand *SrcMO0 = TII->getNamedOperand(I, AMDGPU::OpName::src0);
  const MachineOperand *SrcMO1 = TII->getNamedOperand(I, AMDGPU::OpName::src1);
  Register DstReg = I.getOperand(0).getReg();
  unsigned OpCode = I.getOpcode();
  Register UnpackedDstReg = IsHiBits ? TRI->getSubReg(DstReg, AMDGPU::sub1)
                                     : TRI->getSubReg(DstReg, AMDGPU::sub0);

  int64_t ClampVal = TII->getNamedOperand(I, AMDGPU::OpName::clamp)->getImm();
  unsigned Src0Mods =
      TII->getNamedOperand(I, AMDGPU::OpName::src0_modifiers)->getImm();
  unsigned Src1Mods =
      TII->getNamedOperand(I, AMDGPU::OpName::src1_modifiers)->getImm();

  MachineInstrBuilder NewMI = BuildMI(MBB, I, DL, TII->get(UnpackedOpcode));
  NewMI.addDef(UnpackedDstReg); // vdst
  addOperandAndMods(NewMI, Src0Mods, IsHiBits, *SrcMO0);
  addOperandAndMods(NewMI, Src1Mods, IsHiBits, *SrcMO1);

  if (AMDGPU::hasNamedOperand(OpCode, AMDGPU::OpName::src2)) {
    const MachineOperand *SrcMO2 =
        TII->getNamedOperand(I, AMDGPU::OpName::src2);
    unsigned Src2Mods =
        TII->getNamedOperand(I, AMDGPU::OpName::src2_modifiers)->getImm();
    addOperandAndMods(NewMI, Src2Mods, IsHiBits, *SrcMO2);
  }
  NewMI.addImm(ClampVal); // clamp
  // Packed instructions do not support output modifiers. safe to assign them 0
  // for this use case
  NewMI.addImm(0); // omod
  return NewMI;
}

bool SIPreEmitPeephole::isLateThreeAddrPeepholeCandidate(
    const MachineInstr &MI) const {
  // This is only a subset of the opcodes that are convertible to three-address
  // instructions according to MachineInstr::isConvertibleTo3Addr. This is
  // intended. For some cases, convertToThreeAddress could early-clobber, which
  // would require us to roll back the transformation. However, rollback is not
  // side-effect free for some cases in convertToThreeAddress that fold
  // immediates. Therefore, we limit this peephole to only the instructions for
  // which rollback will not be necessary.
  switch (MI.getOpcode()) {
  case AMDGPU::V_MAC_F16_e32:
  case AMDGPU::V_MAC_F16_e64:
  case AMDGPU::V_MAC_F32_e32:
  case AMDGPU::V_MAC_F32_e64:
  case AMDGPU::V_MAC_LEGACY_F32_e32:
  case AMDGPU::V_MAC_LEGACY_F32_e64:
  case AMDGPU::V_FMAC_F16_e32:
  case AMDGPU::V_FMAC_F16_e64:
  case AMDGPU::V_FMAC_F16_t16_e64:
  case AMDGPU::V_FMAC_F16_fake16_e64:
  case AMDGPU::V_FMAC_F32_e32:
  case AMDGPU::V_FMAC_F32_e64:
  case AMDGPU::V_FMAC_LEGACY_F32_e32:
  case AMDGPU::V_FMAC_LEGACY_F32_e64:
  case AMDGPU::V_FMAC_F64_e32:
  case AMDGPU::V_FMAC_F64_e64:
    return true;
  default:
    return false;
  }
}

Register SIPreEmitPeephole::checkCopy(MachineInstr &TwoAddress,
                                      MachineInstr &MaybeCopy, Register Src,
                                      const TargetRegisterClass *NewDstRC) {
  if (MaybeCopy.isBundle())
    return Register();

  if (!TII->isFoldableCopy(MaybeCopy) ||
      MaybeCopy.getOpcode() == AMDGPU::WWM_COPY)
    return Register();

  if (TII->hasAnyModifiersSet(MaybeCopy))
    return Register();

  MachineOperand &CopySrc =
      MaybeCopy.getOperand(TII->getFoldableCopySrcIdx(MaybeCopy));
  if (!CopySrc.isReg() || CopySrc.getReg() != Src ||
      CopySrc.getSubReg() != AMDGPU::NoSubRegister)
    return Register();

  MachineOperand &Dst = MaybeCopy.getOperand(0);
  if (!Dst.isReg() || Dst.getSubReg() != AMDGPU::NoSubRegister)
    return Register();

  Register DstReg = Dst.getReg();
  if (TRI->getPhysRegBaseClass(DstReg) != TRI->getPhysRegBaseClass(Src))
    return Register();

  if (NewDstRC && !NewDstRC->contains(DstReg))
    return Register();

  // Check that it is safe to define the copy destination register at the
  // current position of the two-address instruction.
  SmallPtrSet<MachineInstr *, 2> Ignore{&TwoAddress, &MaybeCopy};
  if (!RDI->isSafeToDefRegAt(&TwoAddress, DstReg, Ignore))
    return Register();

  // Check that the two-address instruction and the copy have the same exec
  // mask.
  if (!RDI->hasSameReachingDef(&TwoAddress, &MaybeCopy, TRI->getExec()))
    return Register();

  return DstReg;
}

Register SIPreEmitPeephole::shouldReplaceWithThreeAddress(
    MachineInstr &MI, SmallVectorImpl<MachineInstr *> &Copies) {
  MachineOperand *Dst = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);
  if (!Dst || !Dst->isReg())
    return Register();

  const TargetRegisterClass *NewDstRC =
      TII->getRegClass(MI.getDesc(), Dst->getOperandNo());

  auto CheckRegisterCopies =
      [&](Register Reg, const TargetRegisterClass *ReplacementRC) -> Register {
    if (!Reg.isPhysical())
      return Register();
    if (RDI->getLocalLiveOutMIDef(MI.getParent(), Reg) == &MI)
      return Register();

    SmallPtrSet<MachineInstr *, 1> Uses;
    RDI->getReachingLocalUses(&MI, Reg, Uses);
    if (Uses.size() != 1)
      return Register();

    MachineInstr &Use = **Uses.begin();
    Register ReplacementReg = checkCopy(MI, Use, Reg, ReplacementRC);

    if (!ReplacementReg)
      return Register();

    Copies.push_back(&Use);
    return ReplacementReg;
  };

  Register DstReg = Dst->getReg();
  if (Register Replacement = CheckRegisterCopies(DstReg, NewDstRC))
    return Replacement;

  // If we didn't find a copy for the full register, we might be dealing with a
  // 64-bit register and copies for the individual parts. Check both
  // subregisters for matching copies.
  Register Sub0 = TRI->getSubReg(Dst->getReg(), AMDGPU::sub0);
  Register Sub1 = TRI->getSubReg(Dst->getReg(), AMDGPU::sub1);
  if (!Sub0 || !Sub1)
    return Register();

  Register Sub0Replacement = CheckRegisterCopies(Sub0, nullptr);
  Register Sub1Replacement = CheckRegisterCopies(Sub1, nullptr);
  if (!Sub0Replacement || !Sub1Replacement)
    return Register();

  // Try to identify a matching super-register/tuple for the f64 case and abort
  // the transformation if there is none.
  if (!NewDstRC)
    return Register();
  Register Tuple =
      TRI->getMatchingSuperReg(Sub0Replacement, AMDGPU::sub0, NewDstRC);
  if (!Tuple)
    return Register();

  if (Register(TRI->getSubReg(Tuple, AMDGPU::sub1)) != Sub1Replacement)
    return Register();
  return Tuple;
}

bool SIPreEmitPeephole::convertToThreeAddressInstr(
    ThreeAddressCandidate &Cand) {
  MachineInstr *ThreeAddrsInst =
      TII->convertToThreeAddress(*Cand.TwoAddrMI, nullptr, nullptr);
  if (!ThreeAddrsInst)
    return false;
  MachineOperand &NewDst = ThreeAddrsInst->getOperand(0);
  assert(NewDst.isReg());

  // The two asserts are defensive for the case that the implementation of
  // convertToThreeAddress changes in a manner incompatible with this peephole.
  const TargetRegisterClass *DstRC =
      TII->getRegClass(ThreeAddrsInst->getDesc(), NewDst.getOperandNo());
  assert((!DstRC || DstRC->contains(Cand.NewDst)) &&
         "candidate prechecks should guarantee a legal destination class");

  auto RegistersOverlap = [&](MachineOperand &MO) {
    if (!MO.isReg())
      return false;
    return TRI->regsOverlap(Cand.NewDst, MO.getReg());
  };
  assert((!NewDst.isEarlyClobber() ||
          llvm::none_of(ThreeAddrsInst->uses(), RegistersOverlap)) &&
         "three-address instruction must not early-clobber the new destination "
         "register");

  NewDst.setReg(Cand.NewDst);
  NewDst.setIsRenamable(false);
  Cand.TwoAddrMI->eraseFromParent();
  llvm::for_each(Cand.CopiesToRemove,
                 [](MachineInstr *Copy) { Copy->eraseFromParent(); });
  return true;
}

PreservedAnalyses
llvm::SIPreEmitPeepholePass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  auto *MLI = MFAM.getCachedResult<MachineLoopAnalysis>(MF);
  const ReachingDefInfo &RDI = MFAM.getResult<ReachingDefAnalysis>(MF);
  SIPreEmitPeephole Impl;

  if (Impl.run(MF, MLI, &RDI)) {
    auto PA = getMachineFunctionPassPreservedAnalyses();
    PA.preserve<MachineLoopAnalysis>();
    return PA;
  }

  return PreservedAnalyses::all();
}

bool SIPreEmitPeephole::run(MachineFunction &MF, MachineLoopInfo *LoopInfo,
                            const ReachingDefInfo *ReachingDefInfo) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MLI = LoopInfo;
  RDI = ReachingDefInfo;
  bool Changed = false;

  MF.RenumberBlocks();

  for (MachineBasicBlock &MBB : MF) {
    Changed |= removeRedundantModeWrites(MBB);

    MachineBasicBlock::iterator TermI = MBB.getFirstTerminator();
    // Check first terminator for branches to optimize
    if (TermI != MBB.end()) {
      MachineInstr &MI = *TermI;
      switch (MI.getOpcode()) {
      case AMDGPU::S_CBRANCH_VCCZ:
      case AMDGPU::S_CBRANCH_VCCNZ:
        Changed |= optimizeVccBranch(MI);
        break;
      case AMDGPU::S_CBRANCH_EXECZ:
        Changed |= removeExeczBranch(MI, MBB);
        break;
      }
    }

    // Delayed optimization to replace two-address instructions followed by
    // copies with a three-address instruction that directly writes to the
    // copied registers instead. The pattern we seek to optimize is:
    // $vgpr2_vgpr3 = V_FMAC_F64 ..., $vgpr_2_vgpr3
    // $vgpr0 = V_MOV_B32 $vgrp2
    // $vgpr1 = V_MOV_B32 $vgrp3
    //
    // This can be optimized to:
    // $vgpr0_vgpr1 = V_FMA_F64 ..., $vgpr2_vgpr3
    //
    // The TwoAddressInstruction pass handles some cases, but bails in more
    // complex cases.
    SmallVector<ThreeAddressCandidate> Candidates;
    // First, collect the candidates. We can't perform the transformation right
    // away, it would invalidate the iterator.
    for (MachineInstr &MI : MBB.instrs()) {
      if (MI.isBundle())
        continue;

      if (!isLateThreeAddrPeepholeCandidate(MI))
        continue;

      SmallVector<MachineInstr *, 2> Copies;
      if (Register Dst = shouldReplaceWithThreeAddress(MI, Copies))
        Candidates.emplace_back(MI, Dst, Copies);
    }

    for (ThreeAddressCandidate &Cand : Candidates)
      Changed |= convertToThreeAddressInstr(Cand);

    if (!ST.hasVGPRIndexMode())
      continue;

    MachineInstr *SetGPRMI = nullptr;
    const unsigned Threshold = 20;
    unsigned Count = 0;
    // Scan the block for two S_SET_GPR_IDX_ON instructions to see if a
    // second is not needed. Do expensive checks in the optimizeSetGPR()
    // and limit the distance to 20 instructions for compile time purposes.
    // Note: this needs to work on bundles as S_SET_GPR_IDX* instructions
    // may be bundled with the instructions they modify.
    for (auto &MI : make_early_inc_range(MBB.instrs())) {
      if (Count == Threshold)
        SetGPRMI = nullptr;
      else
        ++Count;

      if (MI.getOpcode() != AMDGPU::S_SET_GPR_IDX_ON)
        continue;

      Count = 0;
      if (!SetGPRMI) {
        SetGPRMI = &MI;
        continue;
      }

      if (optimizeSetGPR(*SetGPRMI, MI))
        Changed = true;
      else
        SetGPRMI = &MI;
    }
  }

  // TODO: Fold this into previous block, if possible. Evaluate and handle any
  // side effects.

  // Perform the extra MF scans only for supported archs
  if (!ST.hasGFX940Insts())
    return Changed;
  for (MachineBasicBlock &MBB : MF) {
    // Unpack packed instructions overlapped by MFMAs. This allows the
    // compiler to co-issue unpacked instructions with MFMA
    auto SchedModel = TII->getSchedModel();
    SetVector<MachineInstr *> InstrsToUnpack;
    for (auto &MI : make_early_inc_range(MBB.instrs())) {
      if (!SIInstrInfo::isMFMA(MI))
        continue;
      const MCSchedClassDesc *SchedClassDesc =
          SchedModel.resolveSchedClass(&MI);
      uint16_t NumMFMACycles =
          SchedModel.getWriteProcResBegin(SchedClassDesc)->ReleaseAtCycle;
      collectUnpackingCandidates(MI, InstrsToUnpack, NumMFMACycles);
    }
    for (MachineInstr *MI : InstrsToUnpack) {
      performF32Unpacking(*MI);
    }
  }

  return Changed;
}
