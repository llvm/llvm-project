//===-- SILateBranchLowering.cpp - Final preparation of branches ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass mainly lowers early terminate pseudo instructions.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-late-branch-lowering"

namespace {

class SILateBranchLowering : public MachineFunctionPass {
private:
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  MachineDominatorTree *MDT = nullptr;

  void earlyTerm(MachineInstr &MI, MachineBasicBlock *EarlyExitBlock);

public:
  static char ID;

  unsigned MovOpc;
  Register ExecReg;

  SILateBranchLowering() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI Final Branch Preparation";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char SILateBranchLowering::ID = 0;

INITIALIZE_PASS_BEGIN(SILateBranchLowering, DEBUG_TYPE,
                      "SI insert s_cbranch_execz instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(SILateBranchLowering, DEBUG_TYPE,
                    "SI insert s_cbranch_execz instructions", false, false)

char &llvm::SILateBranchLoweringPassID = SILateBranchLowering::ID;

static void generateEndPgm(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, DebugLoc DL,
                           const SIInstrInfo *TII, MachineFunction &MF) {
  const Function &F = MF.getFunction();
  bool IsPS = F.getCallingConv() == CallingConv::AMDGPU_PS;

  // Check if hardware has been configured to expect color or depth exports.
  bool HasExports =
      AMDGPU::getHasColorExport(F) || AMDGPU::getHasDepthExport(F);

  // Prior to GFX10, hardware always expects at least one export for PS.
  bool MustExport = !AMDGPU::isGFX10Plus(TII->getSubtarget());

  if (IsPS && (HasExports || MustExport)) {
    // Generate "null export" if hardware is expecting PS to export.
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

void SILateBranchLowering::earlyTerm(MachineInstr &MI,
                                     MachineBasicBlock *EarlyExitBlock) {
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc DL = MI.getDebugLoc();

  auto BranchMI = BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_CBRANCH_SCC0))
                      .addMBB(EarlyExitBlock);
  auto Next = std::next(MI.getIterator());

  if (Next != MBB.end() && !Next->isTerminator())
    splitBlock(MBB, *BranchMI, MDT);

  MBB.addSuccessor(EarlyExitBlock);
  MDT->getBase().insertEdge(&MBB, EarlyExitBlock);
}

bool SILateBranchLowering::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MDT = &getAnalysis<MachineDominatorTree>();

  MovOpc = ST.isWave32() ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;
  ExecReg = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;

  SmallVector<MachineInstr *, 4> EarlyTermInstrs;
  SmallVector<MachineInstr *, 1> EpilogInstrs;
  bool MadeChange = false;

  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock::iterator I, Next;
    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);
      MachineInstr &MI = *I;

      switch (MI.getOpcode()) {
      case AMDGPU::S_BRANCH:
        // Optimize out branches to the next block.
        // This only occurs in -O0 when BranchFolding is not executed.
        if (MBB.isLayoutSuccessor(MI.getOperand(0).getMBB())) {
          assert(&MI == &MBB.back());
          MI.eraseFromParent();
          MadeChange = true;
        }
        break;

      case AMDGPU::SI_EARLY_TERMINATE_SCC0:
        EarlyTermInstrs.push_back(&MI);
        break;

      case AMDGPU::SI_RETURN_TO_EPILOG:
        EpilogInstrs.push_back(&MI);
        break;

      default:
        break;
      }
    }
  }

  // Lower any early exit branches first
  if (!EarlyTermInstrs.empty()) {
    MachineBasicBlock *EarlyExitBlock = MF.CreateMachineBasicBlock();
    DebugLoc DL;

    MF.insert(MF.end(), EarlyExitBlock);
    BuildMI(*EarlyExitBlock, EarlyExitBlock->end(), DL, TII->get(MovOpc),
            ExecReg)
        .addImm(0);
    generateEndPgm(*EarlyExitBlock, EarlyExitBlock->end(), DL, TII, MF);

    for (MachineInstr *Instr : EarlyTermInstrs) {
      // Early termination in GS does nothing
      if (MF.getFunction().getCallingConv() != CallingConv::AMDGPU_GS)
        earlyTerm(*Instr, EarlyExitBlock);
      Instr->eraseFromParent();
    }

    EarlyTermInstrs.clear();
    MadeChange = true;
  }

  // Now check return to epilog instructions occur at function end
  if (!EpilogInstrs.empty()) {
    MachineBasicBlock *EmptyMBBAtEnd = nullptr;
    assert(!MF.getInfo<SIMachineFunctionInfo>()->returnsVoid());

    // If there are multiple returns to epilog then all will
    // become jumps to new empty end block.
    if (EpilogInstrs.size() > 1) {
      EmptyMBBAtEnd = MF.CreateMachineBasicBlock();
      MF.insert(MF.end(), EmptyMBBAtEnd);
    }

    for (auto MI : EpilogInstrs) {
      auto MBB = MI->getParent();
      if (MBB == &MF.back() && MI == &MBB->back())
        continue;

      // SI_RETURN_TO_EPILOG is not the last instruction.
      // Jump to empty block at function end.
      if (!EmptyMBBAtEnd) {
        EmptyMBBAtEnd = MF.CreateMachineBasicBlock();
        MF.insert(MF.end(), EmptyMBBAtEnd);
      }

      MBB->addSuccessor(EmptyMBBAtEnd);
      MDT->getBase().insertEdge(MBB, EmptyMBBAtEnd);
      BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(AMDGPU::S_BRANCH))
          .addMBB(EmptyMBBAtEnd);
      MI->eraseFromParent();
      MadeChange = true;
    }

    EpilogInstrs.clear();
  }

  return MadeChange;
}
