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
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>
#include <cstdint>
#include <iterator>

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

  bool shouldSkip(const MachineBasicBlock &From,
                  const MachineBasicBlock &To) const;

  bool dominatesAllReachable(MachineBasicBlock &MBB);
  void createEarlyExitBlock(MachineBasicBlock &MBB);
  void skipIfDead(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                  DebugLoc DL);

  bool kill(MachineInstr &MI);

  bool skipMaskBranch(MachineInstr &MI, MachineBasicBlock &MBB);

public:
  static char ID;

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

/// Check whether \p MBB dominates all blocks that are reachable from it.
bool SIInsertSkips::dominatesAllReachable(MachineBasicBlock &MBB) {
  for (MachineBasicBlock *Other : depth_first(&MBB)) {
    if (!MDT->dominates(&MBB, Other))
      return false;
  }
  return true;
}

static void generatePsEndPgm(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator I, DebugLoc DL,
                             const SIInstrInfo *TII) {
  // Generate "null export; s_endpgm".
  BuildMI(MBB, I, DL, TII->get(AMDGPU::EXP_DONE))
      .addImm(0x09) // V_008DFC_SQ_EXP_NULL
      .addReg(AMDGPU::VGPR0, RegState::Undef)
      .addReg(AMDGPU::VGPR0, RegState::Undef)
      .addReg(AMDGPU::VGPR0, RegState::Undef)
      .addReg(AMDGPU::VGPR0, RegState::Undef)
      .addImm(1)  // vm
      .addImm(0)  // compr
      .addImm(0); // en
  BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ENDPGM)).addImm(0);
}

void SIInsertSkips::createEarlyExitBlock(MachineBasicBlock &MBB) {
  MachineFunction *MF = MBB.getParent();
  DebugLoc DL;

  assert(!EarlyExitBlock);
  EarlyExitBlock = MF->CreateMachineBasicBlock();
  MF->insert(MF->end(), EarlyExitBlock);

  generatePsEndPgm(*EarlyExitBlock, EarlyExitBlock->end(), DL, TII);
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

/// Insert an "if exec=0 { null export; s_endpgm }" sequence before the given
/// iterator. Only applies to pixel shaders.
void SIInsertSkips::skipIfDead(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I, DebugLoc DL) {
  MachineFunction *MF = MBB.getParent();
  (void)MF;
  assert(MF->getFunction().getCallingConv() == CallingConv::AMDGPU_PS);

  // It is possible for an SI_KILL_*_TERMINATOR to sit at the bottom of a
  // basic block that has no further successors (e.g., there was an
  // `unreachable` there in IR). This can happen with original source of the
  // form:
  //
  //   if (uniform_condition) {
  //     write_to_memory();
  //     discard;
  //   }
  //
  // In this case, we write the "null_export; s_endpgm" skip code in the
  // already-existing basic block.
  auto NextBBI = std::next(MBB.getIterator());
  bool NoSuccessor = I == MBB.end() &&
                     llvm::find(MBB.successors(), &*NextBBI) == MBB.succ_end();

  if (NoSuccessor) {
    generatePsEndPgm(MBB, I, DL, TII);
  } else {
    if (!EarlyExitBlock) {
      createEarlyExitBlock(MBB);
      // Update next block pointer to reflect any new blocks
      NextBBI = std::next(MBB.getIterator());
    }

    MachineInstr *BranchMI =
        BuildMI(MBB, I, DL, TII->get(AMDGPU::S_CBRANCH_EXECZ))
            .addMBB(EarlyExitBlock);

    // Split the block if the branch will not come at the end.
    auto Next = std::next(BranchMI->getIterator());
    if (Next != MBB.end() && !Next->isTerminator())
      splitBlock(MBB, *BranchMI, MDT);

    MBB.addSuccessor(EarlyExitBlock);
    MDT->getBase().insertEdge(&MBB, EarlyExitBlock);
  }
}

/// Translate a SI_KILL_*_TERMINATOR into exec-manipulating instructions.
/// Return true unless the terminator is a no-op.
bool SIInsertSkips::kill(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  switch (MI.getOpcode()) {
  case AMDGPU::SI_KILL_F32_COND_IMM_TERMINATOR: {
    unsigned Opcode = 0;

    // The opcodes are inverted because the inline immediate has to be
    // the first operand, e.g. from "x < imm" to "imm > x"
    switch (MI.getOperand(2).getImm()) {
    case ISD::SETOEQ:
    case ISD::SETEQ:
      Opcode = AMDGPU::V_CMPX_EQ_F32_e64;
      break;
    case ISD::SETOGT:
    case ISD::SETGT:
      Opcode = AMDGPU::V_CMPX_LT_F32_e64;
      break;
    case ISD::SETOGE:
    case ISD::SETGE:
      Opcode = AMDGPU::V_CMPX_LE_F32_e64;
      break;
    case ISD::SETOLT:
    case ISD::SETLT:
      Opcode = AMDGPU::V_CMPX_GT_F32_e64;
      break;
    case ISD::SETOLE:
    case ISD::SETLE:
      Opcode = AMDGPU::V_CMPX_GE_F32_e64;
      break;
    case ISD::SETONE:
    case ISD::SETNE:
      Opcode = AMDGPU::V_CMPX_LG_F32_e64;
      break;
    case ISD::SETO:
      Opcode = AMDGPU::V_CMPX_O_F32_e64;
      break;
    case ISD::SETUO:
      Opcode = AMDGPU::V_CMPX_U_F32_e64;
      break;
    case ISD::SETUEQ:
      Opcode = AMDGPU::V_CMPX_NLG_F32_e64;
      break;
    case ISD::SETUGT:
      Opcode = AMDGPU::V_CMPX_NGE_F32_e64;
      break;
    case ISD::SETUGE:
      Opcode = AMDGPU::V_CMPX_NGT_F32_e64;
      break;
    case ISD::SETULT:
      Opcode = AMDGPU::V_CMPX_NLE_F32_e64;
      break;
    case ISD::SETULE:
      Opcode = AMDGPU::V_CMPX_NLT_F32_e64;
      break;
    case ISD::SETUNE:
      Opcode = AMDGPU::V_CMPX_NEQ_F32_e64;
      break;
    default:
      llvm_unreachable("invalid ISD:SET cond code");
    }

    const GCNSubtarget &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();
    if (ST.hasNoSdstCMPX())
      Opcode = AMDGPU::getVCMPXNoSDstOp(Opcode);

    assert(MI.getOperand(0).isReg());

    if (TRI->isVGPR(MBB.getParent()->getRegInfo(),
                    MI.getOperand(0).getReg())) {
      Opcode = AMDGPU::getVOPe32(Opcode);
      BuildMI(MBB, &MI, DL, TII->get(Opcode))
          .add(MI.getOperand(1))
          .add(MI.getOperand(0));
    } else {
      auto I = BuildMI(MBB, &MI, DL, TII->get(Opcode));
      if (!ST.hasNoSdstCMPX())
        I.addReg(AMDGPU::VCC, RegState::Define);

      I.addImm(0)  // src0 modifiers
        .add(MI.getOperand(1))
        .addImm(0)  // src1 modifiers
        .add(MI.getOperand(0));

      I.addImm(0);  // omod
    }
    return true;
  }
  case AMDGPU::SI_KILL_I1_TERMINATOR: {
    const MachineFunction *MF = MI.getParent()->getParent();
    const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
    unsigned Exec = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
    const MachineOperand &Op = MI.getOperand(0);
    int64_t KillVal = MI.getOperand(1).getImm();
    assert(KillVal == 0 || KillVal == -1);

    // Kill all threads if Op0 is an immediate and equal to the Kill value.
    if (Op.isImm()) {
      int64_t Imm = Op.getImm();
      assert(Imm == 0 || Imm == -1);

      if (Imm == KillVal) {
        BuildMI(MBB, &MI, DL, TII->get(ST.isWave32() ? AMDGPU::S_MOV_B32
                                                     : AMDGPU::S_MOV_B64), Exec)
          .addImm(0);
        return true;
      }
      return false;
    }

    unsigned Opcode = KillVal ? AMDGPU::S_ANDN2_B64 : AMDGPU::S_AND_B64;
    if (ST.isWave32())
      Opcode = KillVal ? AMDGPU::S_ANDN2_B32 : AMDGPU::S_AND_B32;
    BuildMI(MBB, &MI, DL, TII->get(Opcode), Exec)
        .addReg(Exec)
        .add(Op);
    return true;
  }
  default:
    llvm_unreachable("invalid opcode, expected SI_KILL_*_TERMINATOR");
  }
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

  SmallVector<MachineInstr *, 4> KillInstrs;
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

      case AMDGPU::SI_KILL_F32_COND_IMM_TERMINATOR:
      case AMDGPU::SI_KILL_I1_TERMINATOR: {
        MadeChange = true;
        bool CanKill = kill(MI);

        // Check if we can add an early "if exec=0 { end shader }".
        //
        // Note that we _always_ do this if it is correct, even if the kill
        // happens fairly late in the shader, because the null export should
        // generally still be cheaper than normal export(s).
        //
        // TODO: The dominatesAllReachable check is conservative: if the
        //       dominance is only missing due to _uniform_ branches, we could
        //       in fact insert the early-exit as well.
        if (CanKill &&
            MF.getFunction().getCallingConv() == CallingConv::AMDGPU_PS &&
            dominatesAllReachable(MBB)) {
          // Mark the instruction for kill-if-dead insertion. We delay this
          // change because it modifies the CFG.
          KillInstrs.push_back(&MI);
        } else {
          MI.eraseFromParent();
        }
        break;
      }

      case AMDGPU::SI_KILL_CLEANUP:
        if (MF.getFunction().getCallingConv() == CallingConv::AMDGPU_PS &&
            dominatesAllReachable(MBB)) {
          KillInstrs.push_back(&MI);
        } else {
          MI.eraseFromParent();
        }
        break;

      default:
        break;
      }
    }
  }

  for (MachineInstr *Kill : KillInstrs) {
    skipIfDead(*Kill->getParent(), std::next(Kill->getIterator()),
               Kill->getDebugLoc());
    Kill->eraseFromParent();
  }
  KillInstrs.clear();
  EarlyExitBlock = nullptr;

  return MadeChange;
}
