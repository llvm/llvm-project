//===-- SIInsertSkips.cpp - Use predicates for control flow ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief This pass inserts branches on the 0 exec mask over divergent branches
/// branches when it's expected that jumping over the untaken control flow will
/// be cheaper than having every workitem no-op through it.
//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/MC/MCAsmInfo.h"

using namespace llvm;

#define DEBUG_TYPE "si-insert-skips"

namespace {

static cl::opt<unsigned> SkipThresholdFlag(
  "amdgpu-skip-threshold",
  cl::desc("Number of instructions before jumping over divergent control flow"),
  cl::init(12), cl::Hidden);

class SIInsertSkips : public MachineFunctionPass {
private:
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  unsigned SkipThreshold;

  bool shouldSkip(const MachineBasicBlock &From,
                  const MachineBasicBlock &To) const;

  bool skipIfDead(MachineInstr &MI, MachineBasicBlock &NextBB);

  void kill(MachineInstr &MI);

  MachineBasicBlock *insertSkipBlock(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  bool skipMaskBranch(MachineInstr &MI, MachineBasicBlock &MBB);

public:
  static char ID;

  SIInsertSkips() :
    MachineFunctionPass(ID), TRI(nullptr), TII(nullptr), SkipThreshold(0) { }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI insert s_cbranch_execz instructions";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace

char SIInsertSkips::ID = 0;

INITIALIZE_PASS(SIInsertSkips, DEBUG_TYPE,
                "SI insert s_cbranch_execz instructions", false, false)

char &llvm::SIInsertSkipsPassID = SIInsertSkips::ID;

static bool opcodeEmitsNoInsts(unsigned Opc) {
  switch (Opc) {
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
  case TargetOpcode::BUNDLE:
  case TargetOpcode::CFI_INSTRUCTION:
  case TargetOpcode::EH_LABEL:
  case TargetOpcode::GC_LABEL:
  case TargetOpcode::DBG_VALUE:
    return true;
  default:
    return false;
  }
}

bool SIInsertSkips::shouldSkip(const MachineBasicBlock &From,
                               const MachineBasicBlock &To) const {
  if (From.succ_empty())
    return false;

  unsigned NumInstr = 0;
  const MachineFunction *MF = From.getParent();

  for (MachineFunction::const_iterator MBBI(&From), ToI(&To), End = MF->end();
       MBBI != End && MBBI != ToI; ++MBBI) {
    const MachineBasicBlock &MBB = *MBBI;

    for (MachineBasicBlock::const_iterator I = MBB.begin(), E = MBB.end();
         NumInstr < SkipThreshold && I != E; ++I) {
      if (opcodeEmitsNoInsts(I->getOpcode()))
        continue;

      // FIXME: Since this is required for correctness, this should be inserted
      // during SILowerControlFlow.

      // When a uniform loop is inside non-uniform control flow, the branch
      // leaving the loop might be an S_CBRANCH_VCCNZ, which is never taken
      // when EXEC = 0. We should skip the loop lest it becomes infinite.
      if (I->getOpcode() == AMDGPU::S_CBRANCH_VCCNZ ||
          I->getOpcode() == AMDGPU::S_CBRANCH_VCCZ)
        return true;

      if (I->isInlineAsm()) {
        const MCAsmInfo *MAI = MF->getTarget().getMCAsmInfo();
        const char *AsmStr = I->getOperand(0).getSymbolName();

        // inlineasm length estimate is number of bytes assuming the longest
        // instruction.
        uint64_t MaxAsmSize = TII->getInlineAsmLength(AsmStr, *MAI);
        NumInstr += MaxAsmSize / MAI->getMaxInstLength();
      } else {
        ++NumInstr;
      }

      if (NumInstr >= SkipThreshold)
        return true;
    }
  }

  return false;
}

bool SIInsertSkips::skipIfDead(MachineInstr &MI, MachineBasicBlock &NextBB) {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction *MF = MBB.getParent();

  if (MF->getFunction()->getCallingConv() != CallingConv::AMDGPU_PS ||
      !shouldSkip(MBB, MBB.getParent()->back()))
    return false;

  MachineBasicBlock *SkipBB = insertSkipBlock(MBB, MI.getIterator());

  const DebugLoc &DL = MI.getDebugLoc();

  // If the exec mask is non-zero, skip the next two instructions
  BuildMI(&MBB, DL, TII->get(AMDGPU::S_CBRANCH_EXECNZ))
    .addMBB(&NextBB);

  MachineBasicBlock::iterator Insert = SkipBB->begin();

  // Exec mask is zero: Export to NULL target...
  BuildMI(*SkipBB, Insert, DL, TII->get(AMDGPU::EXP))
    .addImm(0)
    .addImm(0x09) // V_008DFC_SQ_EXP_NULL
    .addImm(0)
    .addImm(1)
    .addImm(1)
    .addReg(AMDGPU::VGPR0, RegState::Undef)
    .addReg(AMDGPU::VGPR0, RegState::Undef)
    .addReg(AMDGPU::VGPR0, RegState::Undef)
    .addReg(AMDGPU::VGPR0, RegState::Undef);

  // ... and terminate wavefront.
  BuildMI(*SkipBB, Insert, DL, TII->get(AMDGPU::S_ENDPGM));

  return true;
}

void SIInsertSkips::kill(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  const MachineOperand &Op = MI.getOperand(0);

#ifndef NDEBUG
  CallingConv::ID CallConv = MBB.getParent()->getFunction()->getCallingConv();
  // Kill is only allowed in pixel / geometry shaders.
  assert(CallConv == CallingConv::AMDGPU_PS ||
         CallConv == CallingConv::AMDGPU_GS);
#endif
  // Clear this thread from the exec mask if the operand is negative.
  if (Op.isImm()) {
    // Constant operand: Set exec mask to 0 or do nothing
    if (Op.getImm() & 0x80000000) {
      BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_MOV_B64), AMDGPU::EXEC)
        .addImm(0);
    }
  } else {
    BuildMI(MBB, &MI, DL, TII->get(AMDGPU::V_CMPX_LE_F32_e32))
      .addImm(0)
      .addOperand(Op);
  }
}

MachineBasicBlock *SIInsertSkips::insertSkipBlock(
  MachineBasicBlock &MBB, MachineBasicBlock::iterator I) const {
  MachineFunction *MF = MBB.getParent();

  MachineBasicBlock *SkipBB = MF->CreateMachineBasicBlock();
  MachineFunction::iterator MBBI(MBB);
  ++MBBI;

  MF->insert(MBBI, SkipBB);
  MBB.addSuccessor(SkipBB);

  return SkipBB;
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
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  SkipThreshold = SkipThresholdFlag;

  bool HaveKill = false;
  bool MadeChange = false;

  // Track depth of exec mask, divergent branches.
  SmallVector<MachineBasicBlock *, 16> ExecBranchStack;

  MachineFunction::iterator NextBB;

  MachineBasicBlock *EmptyMBBAtEnd = nullptr;

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; BI = NextBB) {
    NextBB = std::next(BI);
    MachineBasicBlock &MBB = *BI;

    if (!ExecBranchStack.empty() && ExecBranchStack.back() == &MBB) {
      // Reached convergence point for last divergent branch.
      ExecBranchStack.pop_back();
    }

    if (HaveKill && ExecBranchStack.empty()) {
      HaveKill = false;

      // TODO: Insert skip if exec is 0?
    }

    MachineBasicBlock::iterator I, Next;
    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);

      MachineInstr &MI = *I;

      switch (MI.getOpcode()) {
      case AMDGPU::SI_MASK_BRANCH: {
        ExecBranchStack.push_back(MI.getOperand(0).getMBB());
        MadeChange |= skipMaskBranch(MI, MBB);
        break;
      }
      case AMDGPU::S_BRANCH: {
        // Optimize out branches to the next block.
        // FIXME: Shouldn't this be handled by BranchFolding?
        if (MBB.isLayoutSuccessor(MI.getOperand(0).getMBB()))
          MI.eraseFromParent();
        break;
      }
      case AMDGPU::SI_KILL_TERMINATOR: {
        MadeChange = true;
        kill(MI);

        if (ExecBranchStack.empty()) {
          if (skipIfDead(MI, *NextBB)) {
            NextBB = std::next(BI);
            BE = MF.end();
            Next = MBB.end();
          }
        } else {
          HaveKill = true;
        }

        MI.eraseFromParent();
        break;
      }
      case AMDGPU::SI_RETURN: {
        // FIXME: Should move somewhere else
        assert(!MF.getInfo<SIMachineFunctionInfo>()->returnsVoid());

        // Graphics shaders returning non-void shouldn't contain S_ENDPGM,
        // because external bytecode will be appended at the end.
        if (BI != --MF.end() || I != MBB.getFirstTerminator()) {
          // SI_RETURN is not the last instruction. Add an empty block at
          // the end and jump there.
          if (!EmptyMBBAtEnd) {
            EmptyMBBAtEnd = MF.CreateMachineBasicBlock();
            MF.insert(MF.end(), EmptyMBBAtEnd);
          }

          MBB.addSuccessor(EmptyMBBAtEnd);
          BuildMI(*BI, I, MI.getDebugLoc(), TII->get(AMDGPU::S_BRANCH))
            .addMBB(EmptyMBBAtEnd);
          I->eraseFromParent();
        }
      }
      default:
        break;
      }
    }
  }

  return MadeChange;
}
