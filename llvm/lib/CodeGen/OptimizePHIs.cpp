//===- OptimizePHIs.cpp - Optimize machine instruction PHIs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes machine instruction PHIs to take advantage of
// opportunities created during DAG legalization.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/OptimizePHIs.h"
#include "CodeGenOptions.h"
#include "llvm/ADT/BoolOrDefault.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include <cassert>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "opt-phis"

STATISTIC(NumPHICycles, "Number of PHI cycles replaced");
STATISTIC(NumDeadPHICycles, "Number of dead PHI cycles");

namespace {

class OptimizePHIs {
  MachineFunction *MF = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  MachineDominatorTree *DT = nullptr;
  std::optional<MachineDominatorTree> OwnedDT;

public:
  explicit OptimizePHIs(MachineDominatorTree *DT = nullptr) : DT(DT) {}
  bool run(MachineFunction &Fn);

private:
  using InstrSet = SmallPtrSet<MachineInstr *, 16>;

  bool IsSingleValuePHICycle(MachineInstr *MI, Register &SingleValReg,
                             InstrSet &PHIsInCycle);
  bool IsDeadPHICycle(MachineInstr *MI, InstrSet &PHIsInCycle);
  bool OptimizeBB(MachineBasicBlock &MBB);

  struct ConstantInputs {
    MachineInstr *Def;
    SmallVector<MachineBasicBlock *, 4> DefBlocks;
    SmallVector<unsigned, 4> OperandIdxs;
  };

  MachineDominatorTree &getDomTree();
  MachineInstr *getDefThroughCopies(Register Reg);
  MachineInstr *getMoveImmDef(Register Reg, const MachineInstr &PHI);
  Register cloneMoveImm(const MachineInstr &Def, MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator InsertPt,
                        const DebugLoc &DL);
  bool shouldHoist(const MachineInstr &Def);
  MachineBasicBlock *findHoistBlock(ArrayRef<MachineBasicBlock *> DefBlocks);
  bool hoistDuplicateConstantsInPHI(MachineInstr &PHI);
  bool foldConstantPHI(MachineInstr &PHI);
};

class OptimizePHIsLegacy : public MachineFunctionPass {
public:
  static char ID;
  OptimizePHIsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    auto *MDTWrapper = getAnalysisIfAvailable<MachineDominatorTreeWrapperPass>();
    OptimizePHIs OP(MDTWrapper ? &MDTWrapper->getDomTree() : nullptr);
    return OP.run(MF);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<MachineRegisterClassInfoWrapperPass>();
    AU.addUsedIfAvailable<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // end anonymous namespace

char OptimizePHIsLegacy::ID = 0;

char &llvm::OptimizePHIsLegacyID = OptimizePHIsLegacy::ID;

INITIALIZE_PASS(OptimizePHIsLegacy, DEBUG_TYPE,
                "Optimize machine instruction PHIs", false, false)

PreservedAnalyses OptimizePHIsPass::run(MachineFunction &MF,
                                        MachineFunctionAnalysisManager &MFAM) {
  OptimizePHIs OP(MFAM.getCachedResult<MachineDominatorTreeAnalysis>(MF));
  if (!OP.run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<MachineRegisterClassAnalysis>();
  return PA;
}

bool OptimizePHIs::run(MachineFunction &Fn) {
  MF = &Fn;
  MRI = &Fn.getRegInfo();
  TII = Fn.getSubtarget().getInstrInfo();
  OwnedDT.reset();

  // Find dead PHI cycles and PHI cycles that can be replaced by a single
  // value.  InstCombine does these optimizations, but DAG legalization may
  // introduce new opportunities, e.g., when i64 values are split up for
  // 32-bit targets.
  bool Changed = false;
  for (MachineBasicBlock &MBB : Fn)
    Changed |= OptimizeBB(MBB);

  return Changed;
}

/// IsSingleValuePHICycle - Check if MI is a PHI where all the source operands
/// are copies of SingleValReg, possibly via copies through other PHIs. If
/// SingleValReg is zero on entry, it is set to the register with the single
/// non-copy value. PHIsInCycle is a set used to keep track of the PHIs that
/// have been scanned. PHIs may be grouped by cycle, several cycles or chains.
bool OptimizePHIs::IsSingleValuePHICycle(MachineInstr *MI,
                                         Register &SingleValReg,
                                         InstrSet &PHIsInCycle) {
  assert(MI->isPHI() && "IsSingleValuePHICycle expects a PHI instruction");
  Register DstReg = MI->getOperand(0).getReg();

  // See if we already saw this register.
  if (!PHIsInCycle.insert(MI).second)
    return true;

  // Don't scan crazily complex things.
  if (PHIsInCycle.size() == 16)
    return false;

  // Scan the PHI operands.
  for (unsigned i = 1; i != MI->getNumOperands(); i += 2) {
    Register SrcReg = MI->getOperand(i).getReg();
    if (SrcReg == DstReg)
      continue;
    MachineInstr *SrcMI = MRI->getVRegDef(SrcReg);

    // Skip over register-to-register moves.
    if (SrcMI && SrcMI->isCopy() && !SrcMI->getOperand(0).getSubReg() &&
        !SrcMI->getOperand(1).getSubReg() &&
        SrcMI->getOperand(1).getReg().isVirtual()) {
      SrcReg = SrcMI->getOperand(1).getReg();
      SrcMI = MRI->getVRegDef(SrcReg);
    }
    if (!SrcMI)
      return false;

    if (SrcMI->isPHI()) {
      if (!IsSingleValuePHICycle(SrcMI, SingleValReg, PHIsInCycle))
        return false;
    } else {
      // Fail if there is more than one non-phi/non-move register.
      if (SingleValReg && SingleValReg != SrcReg)
        return false;
      SingleValReg = SrcReg;
    }
  }
  return true;
}

/// IsDeadPHICycle - Check if the register defined by a PHI is only used by
/// other PHIs in a cycle.
bool OptimizePHIs::IsDeadPHICycle(MachineInstr *MI, InstrSet &PHIsInCycle) {
  assert(MI->isPHI() && "IsDeadPHICycle expects a PHI instruction");
  Register DstReg = MI->getOperand(0).getReg();
  assert(DstReg.isVirtual() && "PHI destination is not a virtual register");

  // See if we already saw this register.
  if (!PHIsInCycle.insert(MI).second)
    return true;

  // Don't scan crazily complex things.
  if (PHIsInCycle.size() == 16)
    return false;

  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(DstReg)) {
    if (!UseMI.isPHI() || !IsDeadPHICycle(&UseMI, PHIsInCycle))
      return false;
  }

  return true;
}

MachineDominatorTree &OptimizePHIs::getDomTree() {
  if (!DT)
    DT = &OwnedDT.emplace(*MF);
  return *DT;
}

MachineInstr *OptimizePHIs::getDefThroughCopies(Register Reg) {
  MachineInstr *Def = MRI->getVRegDef(Reg);
  while (Def && Def->isCopy()) {
    const MachineOperand &Src = Def->getOperand(1);
    if (Def->getOperand(0).getSubReg() || Src.getSubReg() ||
        !Src.getReg().isVirtual())
      return nullptr;
    Def = MRI->getVRegDef(Src.getReg());
  }
  return Def;
}

MachineInstr *OptimizePHIs::getMoveImmDef(Register Reg,
                                          const MachineInstr &PHI) {
  MachineInstr *Def = getDefThroughCopies(Reg);
  if (!Def || !Def->isMoveImmediate() || Def->getNumImplicitOperands())
    return nullptr;
  const MachineOperand &Dst = Def->getOperand(0);
  if (!Dst.getReg().isVirtual() ||
      MRI->getRegClass(Dst.getReg()) !=
          MRI->getRegClass(PHI.getOperand(0).getReg()))
    return nullptr;
  const MachineOperand &Imm = Def->getOperand(1);
  if (!Imm.isImm() && !Imm.isCImm() && !Imm.isFPImm())
    return nullptr;
  return Def;
}

Register OptimizePHIs::cloneMoveImm(const MachineInstr &Def,
                                    MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator InsertPt,
                                    const DebugLoc &DL) {
  Register NewReg =
      MRI->createVirtualRegister(MRI->getRegClass(Def.getOperand(0).getReg()));
  MachineInstr *NewMI = MF->CloneMachineInstr(&Def);
  NewMI->getOperand(0).setReg(NewReg);
  NewMI->setDebugLoc(DL);
  MBB.insert(InsertPt, NewMI);
  return NewReg;
}

MachineBasicBlock *
OptimizePHIs::findHoistBlock(ArrayRef<MachineBasicBlock *> DefBlocks) {
  MachineDominatorTree &MDT = getDomTree();
  for (MachineBasicBlock *MBB : DefBlocks) {
    if (MBB->pred_size() != 1)
      continue;
    MachineBasicBlock *Pred = *MBB->pred_begin();
    if (all_of(DefBlocks, [&](MachineBasicBlock *DefMBB) {
          return MDT.dominates(Pred, DefMBB);
        }))
      return Pred;
  }
  return nullptr;
}

bool OptimizePHIs::shouldHoist(const MachineInstr &Def) {
  const auto &Opts =
      MF->getFunction().getContext().getLibraryOptions().get<CodeGenOptions>();
  return valueOr(Opts.hoist_phi_consts, TII->shouldHoistPHIConstant(Def));
}

// If we have a phi of the form
//
//    phi (v1:b1, v2:b2, ..., vk:bk)
//
// where def v1, v2, ..., vk are the same constant c.
//
// If v1 is defined in b1, v2 in b2, ..., vk in bk then we define a new
// def v=c in leastCommonDom(b1, b2, ..., bk) and replace uses of
// v1, v2, ..., vk in phi by v.
bool OptimizePHIs::hoistDuplicateConstantsInPHI(MachineInstr &PHI) {
  SmallVector<ConstantInputs, 4> Constants;
  for (unsigned I = 1, E = PHI.getNumOperands(); I != E; I += 2) {
    MachineInstr *Def = getMoveImmDef(PHI.getOperand(I).getReg(), PHI);
    if (!Def || !shouldHoist(*Def))
      continue;

    auto *It = find_if(Constants, [&](const ConstantInputs &C) {
      return C.Def->isIdenticalTo(*Def, MachineInstr::IgnoreDefs);
    });
    if (It == Constants.end())
      It = &Constants.emplace_back(ConstantInputs{Def, {}, {}});
    It->DefBlocks.push_back(Def->getParent());
    It->OperandIdxs.push_back(I);
  }

  bool Changed = false;
  for (ConstantInputs &C : Constants) {
    if (C.OperandIdxs.size() < 2)
      continue;
    llvm::sort(C.DefBlocks, [](const MachineBasicBlock *A,
                               const MachineBasicBlock *B) {
      return A->getNumber() < B->getNumber();
    });
    C.DefBlocks.erase(llvm::unique(C.DefBlocks), C.DefBlocks.end());
    if (C.DefBlocks.size() == 1)
      continue;

    MachineBasicBlock *Target = findHoistBlock(C.DefBlocks);
    if (!Target)
      continue;

    Register NewReg = cloneMoveImm(*C.Def, *Target,
                                   Target->getFirstTerminator(),
                                   PHI.getDebugLoc());
    for (unsigned I : C.OperandIdxs)
      PHI.getOperand(I).setReg(NewReg);
    Changed = true;
  }
  return Changed;
}

// Given a PHI whose inputs are all identical integer constants, replace it
// with a single definition of the constant in the PHI's block.
bool OptimizePHIs::foldConstantPHI(MachineInstr &PHI) {
  if (PHI.getNumOperands() < 3)
    return false;

  MachineInstr *Def = getMoveImmDef(PHI.getOperand(1).getReg(), PHI);
  if (!Def || !Def->getOperand(1).isImm() || !TII->shouldFoldConstantPHI(*Def))
    return false;

  for (unsigned I = 3, E = PHI.getNumOperands(); I < E; I += 2) {
    MachineInstr *OpDef = getMoveImmDef(PHI.getOperand(I).getReg(), PHI);
    if (!OpDef || !OpDef->isIdenticalTo(*Def, MachineInstr::IgnoreDefs))
      return false;
  }

  MachineBasicBlock &MBB = *PHI.getParent();
  Register NewReg =
      cloneMoveImm(*Def, MBB, MBB.getFirstNonPHI(), PHI.getDebugLoc());
  MRI->replaceRegWith(PHI.getOperand(0).getReg(), NewReg);
  PHI.eraseFromParent();
  return true;
}

/// OptimizeBB - Remove dead PHI cycles and PHI cycles that can be replaced by
/// a single value.
bool OptimizePHIs::OptimizeBB(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineBasicBlock::iterator
         MII = MBB.begin(), E = MBB.end(); MII != E; ) {
    MachineInstr *MI = &*MII++;
    if (!MI->isPHI())
      break;

    Changed |= hoistDuplicateConstantsInPHI(*MI);
    if (foldConstantPHI(*MI)) {
      Changed = true;
      continue;
    }

    // Check for single-value PHI cycles.
    Register SingleValReg;
    InstrSet PHIsInCycle;
    if (IsSingleValuePHICycle(MI, SingleValReg, PHIsInCycle) && SingleValReg) {
      Register OldReg = MI->getOperand(0).getReg();
      if (!MRI->constrainRegClass(SingleValReg, MRI->getRegClass(OldReg)))
        continue;

      MRI->replaceRegWith(OldReg, SingleValReg);
      MI->eraseFromParent();

      ++NumPHICycles;
      Changed = true;
      continue;
    }

    // Check for dead PHI cycles.
    PHIsInCycle.clear();
    if (IsDeadPHICycle(MI, PHIsInCycle)) {
      for (MachineInstr *PhiMI : PHIsInCycle) {
        if (MII == PhiMI)
          ++MII;
        PhiMI->eraseFromParent();
      }
      ++NumDeadPHICycles;
      Changed = true;
    }
  }
  return Changed;
}
