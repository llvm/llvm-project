//===-- SIInsertSkips.cpp - Use predicates for control flow ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass inserts branches on the 0 exec mask over divergent branches
/// branches when it's expected that jumping over the untaken control flow will
/// be cheaper than having every workitem no-op through it.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-insert-skips"

static cl::opt<unsigned> SkipThresholdFlag(
  "amdgpu-skip-threshold-legacy",
  cl::desc("Number of instructions before jumping over divergent control flow"),
  cl::init(12), cl::Hidden);

namespace {

class SIInsertSkips : public MachineFunctionPass {
private:
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  unsigned SkipThreshold = 0;
  MachineDominatorTree *MDT = nullptr;

  MachineBasicBlock *EarlyExitBlock = nullptr;
  bool EarlyExitClearsExec = false;

  bool shouldSkip(const MachineBasicBlock &From,
                  const MachineBasicBlock &To) const;

  void ensureEarlyExitBlock(MachineBasicBlock &MBB, bool ClearExec);

  void earlyTerm(MachineInstr &MI);

  bool skipMaskBranch(MachineInstr &MI, MachineBasicBlock &MBB);

public:
  static char ID;

  unsigned MovOpc;
  Register ExecReg;

  SIInsertSkips() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI insert s_cbranch_execz instructions";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char SIInsertSkips::ID = 0;

INITIALIZE_PASS_BEGIN(SIInsertSkips, DEBUG_TYPE,
                      "SI insert s_cbranch_execz instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(SIInsertSkips, DEBUG_TYPE,
                    "SI insert s_cbranch_execz instructions", false, false)

char &llvm::SIInsertSkipsPassID = SIInsertSkips::ID;

static bool opcodeEmitsNoInsts(const MachineInstr &MI) {
  if (MI.isMetaInstruction())
    return true;

  // Handle target specific opcodes.
  switch (MI.getOpcode()) {
  case AMDGPU::SI_MASK_BRANCH:
    return true;
  default:
    return false;
  }
}

bool SIInsertSkips::shouldSkip(const MachineBasicBlock &From,
                               const MachineBasicBlock &To) const {
  unsigned NumInstr = 0;
  const MachineFunction *MF = From.getParent();

  for (MachineFunction::const_iterator MBBI(&From), ToI(&To), End = MF->end();
       MBBI != End && MBBI != ToI; ++MBBI) {
    const MachineBasicBlock &MBB = *MBBI;

    for (MachineBasicBlock::const_iterator I = MBB.begin(), E = MBB.end();
         NumInstr < SkipThreshold && I != E; ++I) {
      if (opcodeEmitsNoInsts(*I))
        continue;

      // FIXME: Since this is required for correctness, this should be inserted
      // during SILowerControlFlow.

      // When a uniform loop is inside non-uniform control flow, the branch
      // leaving the loop might be an S_CBRANCH_VCCNZ, which is never taken
      // when EXEC = 0. We should skip the loop lest it becomes infinite.
      if (I->getOpcode() == AMDGPU::S_CBRANCH_VCCNZ ||
          I->getOpcode() == AMDGPU::S_CBRANCH_VCCZ)
        return true;

      if (TII->hasUnwantedEffectsWhenEXECEmpty(*I))
        return true;

      // These instructions are potentially expensive even if EXEC = 0.
      if (TII->isSMRD(*I) || TII->isVMEM(*I) || TII->isFLAT(*I) ||
          I->getOpcode() == AMDGPU::S_WAITCNT)
        return true;

      ++NumInstr;
      if (NumInstr >= SkipThreshold)
        return true;
    }
  }

  return false;
}

static void generateEndPgm(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, DebugLoc DL,
                           const SIInstrInfo *TII, bool IsPS) {
  // "null export"
  if (IsPS) {
    BuildMI(MBB, I, DL, TII->get(AMDGPU::EXP_DONE))
        .addImm(AMDGPU::Exp::ET_NULL)
        .addReg(AMDGPU::VGPR0, RegState::Undef)
        .addReg(AMDGPU::VGPR0, RegState::Undef)
        .addReg(AMDGPU::VGPR0, RegState::Undef)
        .addReg(AMDGPU::VGPR0, RegState::Undef)
        .addImm(1)  // vm
        .addImm(0)  // compr
        .addImm(0); // en
  }
  // s_endpgm
  BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ENDPGM)).addImm(0);
}

void SIInsertSkips::ensureEarlyExitBlock(MachineBasicBlock &MBB,
                                         bool ClearExec) {
  MachineFunction *MF = MBB.getParent();
  DebugLoc DL;

  if (!EarlyExitBlock) {
    EarlyExitBlock = MF->CreateMachineBasicBlock();
    MF->insert(MF->end(), EarlyExitBlock);
    generateEndPgm(*EarlyExitBlock, EarlyExitBlock->end(), DL, TII,
                   MF->getFunction().getCallingConv() ==
                       CallingConv::AMDGPU_PS);
    EarlyExitClearsExec = false;
  }

  if (ClearExec && !EarlyExitClearsExec) {
    auto ExitI = EarlyExitBlock->getFirstNonPHI();
    BuildMI(*EarlyExitBlock, ExitI, DL, TII->get(MovOpc), ExecReg).addImm(0);
    EarlyExitClearsExec = true;
  }
}

static void splitBlock(MachineBasicBlock &MBB, MachineInstr &MI,
                       MachineDominatorTree *MDT) {
  MachineBasicBlock *SplitBB = MBB.splitAt(MI, /*UpdateLiveIns*/ true);

  // Update dominator tree
  using DomTreeT = DomTreeBase<MachineBasicBlock>;
  SmallVector<DomTreeT::UpdateType, 16> DTUpdates;
  for (MachineBasicBlock *Succ : SplitBB->successors()) {
    DTUpdates.push_back({DomTreeT::Insert, SplitBB, Succ});
    DTUpdates.push_back({DomTreeT::Delete, &MBB, Succ});
  }
  DTUpdates.push_back({DomTreeT::Insert, &MBB, SplitBB});
  MDT->getBase().applyUpdates(DTUpdates);
}

void SIInsertSkips::earlyTerm(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc DL = MI.getDebugLoc();

  ensureEarlyExitBlock(MBB, true);

  auto BranchMI = BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_CBRANCH_SCC0))
                      .addMBB(EarlyExitBlock);
  auto Next = std::next(MI.getIterator());

  if (Next != MBB.end() && !Next->isTerminator())
    splitBlock(MBB, *BranchMI, MDT);

  MBB.addSuccessor(EarlyExitBlock);
  MDT->getBase().insertEdge(&MBB, EarlyExitBlock);
}

// Returns true if a branch over the block was inserted.
bool SIInsertSkips::skipMaskBranch(MachineInstr &MI,
                                   MachineBasicBlock &SrcMBB) {
  MachineBasicBlock *DestBB = MI.getOperand(0).getMBB();

  if (!shouldSkip(**SrcMBB.succ_begin(), *DestBB))
    return false;

  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator InsPt = std::next(MI.getIterator());

  BuildMI(SrcMBB, InsPt, DL, TII->get(AMDGPU::S_CBRANCH_EXECZ))
    .addMBB(DestBB);

  return true;
}

bool SIInsertSkips::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MDT = &getAnalysis<MachineDominatorTree>();
  SkipThreshold = SkipThresholdFlag;

  MovOpc = ST.isWave32() ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;
  ExecReg = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;

  SmallVector<MachineInstr *, 4> EarlyTermInstrs;
  bool MadeChange = false;

  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock::iterator I, Next;
    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);
      MachineInstr &MI = *I;

      switch (MI.getOpcode()) {
      case AMDGPU::SI_MASK_BRANCH:
        MadeChange |= skipMaskBranch(MI, MBB);
        break;

      case AMDGPU::S_BRANCH:
        // Optimize out branches to the next block.
        // FIXME: Shouldn't this be handled by BranchFolding?
        if (MBB.isLayoutSuccessor(MI.getOperand(0).getMBB())) {
          assert(&MI == &MBB.back());
          MI.eraseFromParent();
          MadeChange = true;
        }
        break;

      case AMDGPU::SI_EARLY_TERMINATE_SCC0:
        EarlyTermInstrs.push_back(&MI);
        break;

      default:
        break;
      }
    }
  }

  for (MachineInstr *Instr : EarlyTermInstrs) {
    // Early termination in GS does nothing
    if (MF.getFunction().getCallingConv() != CallingConv::AMDGPU_GS)
      earlyTerm(*Instr);
    Instr->eraseFromParent();
  }
  EarlyTermInstrs.clear();
  EarlyExitBlock = nullptr;

  return MadeChange;
}
