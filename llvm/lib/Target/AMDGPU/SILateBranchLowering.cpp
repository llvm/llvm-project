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
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-late-branch-lowering"

namespace {

class SILateBranchLowering {
private:
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  MachineDominatorTree *MDT = nullptr;

  void expandChainCall(MachineInstr &MI, const GCNSubtarget &ST,
                       bool DynamicVGPR);
  void earlyTerm(MachineInstr &MI, MachineBasicBlock *EarlyExitBlock);

public:
  SILateBranchLowering(MachineDominatorTree *MDT) : MDT(MDT) {}

  bool run(MachineFunction &MF);

  unsigned MovOpc;
  Register ExecReg;
};

class SILateBranchLoweringLegacy : public MachineFunctionPass {
public:
  static char ID;
  SILateBranchLoweringLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto *MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
    return SILateBranchLowering(MDT).run(MF);
  }

  StringRef getPassName() const override {
    return "SI Final Branch Preparation";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char SILateBranchLoweringLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(SILateBranchLoweringLegacy, DEBUG_TYPE,
                      "SI insert s_cbranch_execz instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(SILateBranchLoweringLegacy, DEBUG_TYPE,
                    "SI insert s_cbranch_execz instructions", false, false)

char &llvm::SILateBranchLoweringPassID = SILateBranchLoweringLegacy::ID;

static void generateEndPgm(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, DebugLoc DL,
                           const SIInstrInfo *TII, MachineFunction &MF) {
  const Function &F = MF.getFunction();
  bool IsPS = F.getCallingConv() == CallingConv::AMDGPU_PS;

  // Check if hardware has been configured to expect color or depth exports.
  bool HasColorExports = AMDGPU::getHasColorExport(F);
  bool HasDepthExports = AMDGPU::getHasDepthExport(F);
  bool HasExports = HasColorExports || HasDepthExports;

  // Prior to GFX10, hardware always expects at least one export for PS.
  bool MustExport = !AMDGPU::isGFX10Plus(TII->getSubtarget());

  if (IsPS && (HasExports || MustExport)) {
    // Generate "null export" if hardware is expecting PS to export.
    const GCNSubtarget &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();
    int Target =
        ST.hasNullExportTarget()
            ? AMDGPU::Exp::ET_NULL
            : (HasColorExports ? AMDGPU::Exp::ET_MRT0 : AMDGPU::Exp::ET_MRTZ);
    BuildMI(MBB, I, DL, TII->get(AMDGPU::EXP_DONE))
        .addImm(Target)
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
  MDT->applyUpdates(DTUpdates);
}

static void copyOpWithoutRegFlags(MachineInstrBuilder &MIB,
                                  MachineOperand &Op) {
  if (Op.isReg())
    MIB.addReg(Op.getReg());
  else
    MIB.add(Op);
}

void SILateBranchLowering::expandChainCall(MachineInstr &MI,
                                           const GCNSubtarget &ST,
                                           bool DynamicVGPR) {
  // This is a tail call that needs to be expanded into at least
  // 2 instructions, one for setting EXEC and one for the actual tail call.
  int ExecIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::exec);
  assert(ExecIdx != -1 && "Missing EXEC operand");
  const DebugLoc &DL = MI.getDebugLoc();
  if (DynamicVGPR) {
    // We have 3 extra operands and we need to:
    // * Try to change the VGPR allocation
    // * Select the callee based on the result of the reallocation attempt
    // * Select the EXEC mask based on the result of the reallocation attempt
    // If any of the register operands of the chain pseudo is used in more than
    // one of these instructions, we need to make sure that the kill flags
    // aren't copied along.
    auto AllocMI =
        BuildMI(*MI.getParent(), MI, DL, TII->get(AMDGPU::S_ALLOC_VGPR));
    copyOpWithoutRegFlags(AllocMI,
                          *TII->getNamedOperand(MI, AMDGPU::OpName::numvgprs));

    auto SelectCallee =
        BuildMI(*MI.getParent(), MI, DL, TII->get(AMDGPU::S_CSELECT_B64))
            .addDef(TII->getNamedOperand(MI, AMDGPU::OpName::src0)->getReg());
    copyOpWithoutRegFlags(SelectCallee,
                          *TII->getNamedOperand(MI, AMDGPU::OpName::src0));
    copyOpWithoutRegFlags(SelectCallee,
                          *TII->getNamedOperand(MI, AMDGPU::OpName::fbcallee));

    auto SelectExec = BuildMI(*MI.getParent(), MI, DL,
                              TII->get(ST.isWave32() ? AMDGPU::S_CSELECT_B32
                                                     : AMDGPU::S_CSELECT_B64))
                          .addDef(ExecReg);

    copyOpWithoutRegFlags(SelectExec,
                          *TII->getNamedOperand(MI, AMDGPU::OpName::exec));
    copyOpWithoutRegFlags(SelectExec,
                          *TII->getNamedOperand(MI, AMDGPU::OpName::fbexec));
  } else {
    auto SetExec = BuildMI(*MI.getParent(), MI, DL, TII->get(MovOpc), ExecReg);
    copyOpWithoutRegFlags(SetExec,
                          *TII->getNamedOperand(MI, AMDGPU::OpName::exec));
  }

  for (int OpIdx = MI.getNumExplicitOperands() - 1; OpIdx >= ExecIdx; --OpIdx)
    MI.removeOperand(OpIdx);

  MI.setDesc(TII->get(AMDGPU::SI_TCRETURN));
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
  MDT->insertEdge(&MBB, EarlyExitBlock);
}

PreservedAnalyses
llvm::SILateBranchLoweringPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &MFAM) {
  auto *MDT = &MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  if (!SILateBranchLowering(MDT).run(MF))
    return PreservedAnalyses::all();

  return getMachineFunctionPassPreservedAnalyses()
      .preserve<MachineDominatorTreeAnalysis>();
}

bool SILateBranchLowering::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();

  MovOpc = ST.isWave32() ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;
  ExecReg = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;

  SmallVector<MachineInstr *, 4> EarlyTermInstrs;
  SmallVector<MachineInstr *, 1> EpilogInstrs;
  bool MadeChange = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
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

      case AMDGPU::SI_CS_CHAIN_TC_W32:
      case AMDGPU::SI_CS_CHAIN_TC_W64:
        expandChainCall(MI, ST, /*DynamicVGPR=*/false);
        MadeChange = true;
        break;
      case AMDGPU::SI_CS_CHAIN_TC_W32_DVGPR:
      case AMDGPU::SI_CS_CHAIN_TC_W64_DVGPR:
        expandChainCall(MI, ST, /*DynamicVGPR=*/true);
        MadeChange = true;
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

    for (auto *MI : EpilogInstrs) {
      auto *MBB = MI->getParent();
      if (MBB == &MF.back() && MI == &MBB->back())
        continue;

      // SI_RETURN_TO_EPILOG is not the last instruction.
      // Jump to empty block at function end.
      if (!EmptyMBBAtEnd) {
        EmptyMBBAtEnd = MF.CreateMachineBasicBlock();
        MF.insert(MF.end(), EmptyMBBAtEnd);
      }

      MBB->addSuccessor(EmptyMBBAtEnd);
      MDT->insertEdge(MBB, EmptyMBBAtEnd);
      BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(AMDGPU::S_BRANCH))
          .addMBB(EmptyMBBAtEnd);
      MI->eraseFromParent();
      MadeChange = true;
    }

    EpilogInstrs.clear();
  }

  return MadeChange;
}
