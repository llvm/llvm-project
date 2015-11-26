//===-- WebAssemblyRegStackify.cpp - Register Stackification --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a register stacking pass.
///
/// This pass reorders instructions to put register uses and defs in an order
/// such that they form single-use expression trees. Registers fitting this form
/// are then marked as "stackified", meaning references to them are replaced by
/// "push" and "pop" from the stack.
///
/// This is primarily a code size optimiation, since temporary values on the
/// expression don't need to be named.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h" // for WebAssembly::ARGUMENT_*
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-reg-stackify"

namespace {
class WebAssemblyRegStackify final : public MachineFunctionPass {
  const char *getPassName() const override {
    return "WebAssembly Register Stackify";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addPreserved<MachineBlockFrequencyInfo>();
    AU.addPreservedID(MachineDominatorsID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyRegStackify() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyRegStackify::ID = 0;
FunctionPass *llvm::createWebAssemblyRegStackify() {
  return new WebAssemblyRegStackify();
}

// Decorate the given instruction with implicit operands that enforce the
// expression stack ordering constraints.
static void ImposeStackOrdering(MachineInstr *MI) {
  // Read and write the opaque EXPR_STACK register.
  MI->addOperand(MachineOperand::CreateReg(WebAssembly::EXPR_STACK,
                                           /*isDef=*/true,
                                           /*isImp=*/true));
  MI->addOperand(MachineOperand::CreateReg(WebAssembly::EXPR_STACK,
                                           /*isDef=*/false,
                                           /*isImp=*/true));
}

// Test whether it's safe to move Def to just before Insert. Note that this
// doesn't account for physical register dependencies, because WebAssembly
// doesn't have any (other than special ones like EXPR_STACK).
// TODO: Compute memory dependencies in a way that doesn't require always
// walking the block.
// TODO: Compute memory dependencies in a way that uses AliasAnalysis to be
// more precise.
static bool IsSafeToMove(const MachineInstr *Def, const MachineInstr *Insert,
                         AliasAnalysis &AA) {
  bool SawStore = false, SawSideEffects = false;
  MachineBasicBlock::const_iterator D(Def), I(Insert);
  for (--I; I != D; --I)
    SawSideEffects |= I->isSafeToMove(&AA, SawStore);

  return !(SawStore && Def->mayLoad() && !Def->isInvariantLoad(&AA)) &&
         !(SawSideEffects && !Def->isSafeToMove(&AA, SawStore));
}

bool WebAssemblyRegStackify::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "********** Register Stackifying **********\n"
                  "********** Function: "
               << MF.getName() << '\n');

  bool Changed = false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  WebAssemblyFunctionInfo &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  AliasAnalysis &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();

  // Walk the instructions from the bottom up. Currently we don't look past
  // block boundaries, and the blocks aren't ordered so the block visitation
  // order isn't significant, but we may want to change this in the future.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : reverse(MBB)) {
      MachineInstr *Insert = &MI;

      // Don't nest anything inside a phi.
      if (Insert->getOpcode() == TargetOpcode::PHI)
        break;

      // Don't nest anything inside an inline asm, because we don't have
      // constraints for $push inputs.
      if (Insert->getOpcode() == TargetOpcode::INLINEASM)
        break;

      // Iterate through the inputs in reverse order, since we'll be pulling
      // operands off the stack in FIFO order.
      bool AnyStackified = false;
      for (MachineOperand &Op : reverse(Insert->uses())) {
        // We're only interested in explicit virtual register operands.
        if (!Op.isReg() || Op.isImplicit() || !Op.isUse())
          continue;

        unsigned Reg = Op.getReg();
        if (!TargetRegisterInfo::isVirtualRegister(Reg))
          continue;

        // Only consider registers with a single definition.
        // TODO: Eventually we may relax this, to stackify phi transfers.
        MachineInstr *Def = MRI.getVRegDef(Reg);
        if (!Def)
          continue;

        // There's no use in nesting implicit defs inside anything.
        if (Def->getOpcode() == TargetOpcode::IMPLICIT_DEF)
          continue;

        // Don't nest an INLINE_ASM def into anything, because we don't have
        // constraints for $pop outputs.
        if (Def->getOpcode() == TargetOpcode::INLINEASM)
          continue;

        // Don't nest PHIs inside of anything.
        if (Def->getOpcode() == TargetOpcode::PHI)
          continue;

        // Argument instructions represent live-in registers and not real
        // instructions.
        if (Def->getOpcode() == WebAssembly::ARGUMENT_I32 ||
            Def->getOpcode() == WebAssembly::ARGUMENT_I64 ||
            Def->getOpcode() == WebAssembly::ARGUMENT_F32 ||
            Def->getOpcode() == WebAssembly::ARGUMENT_F64)
          continue;

        // Single-use expression trees require defs that have one use, or that
        // they be trivially clonable.
        // TODO: Eventually we'll relax this, to take advantage of set_local
        // returning its result.
        if (!MRI.hasOneUse(Reg))
          continue;

        // For now, be conservative and don't look across block boundaries,
        // unless we have something trivially clonable.
        // TODO: Be more aggressive.
        if (Def->getParent() != &MBB && !Def->isMoveImmediate())
          continue;

        // Don't move instructions that have side effects or memory dependencies
        // or other complications.
        if (!IsSafeToMove(Def, Insert, AA))
          continue;

        Changed = true;
        AnyStackified = true;
        // Move the def down and nest it in the current instruction.
        MBB.insert(MachineBasicBlock::instr_iterator(Insert),
                   Def->removeFromParent());
        MFI.stackifyVReg(Reg);
        ImposeStackOrdering(Def);
        Insert = Def;
      }
      if (AnyStackified)
        ImposeStackOrdering(&MI);
    }
  }

  // If we used EXPR_STACK anywhere, add it to the live-in sets everywhere
  // so that it never looks like a use-before-def.
  if (Changed) {
    MF.getRegInfo().addLiveIn(WebAssembly::EXPR_STACK);
    for (MachineBasicBlock &MBB : MF)
      MBB.addLiveIn(WebAssembly::EXPR_STACK);
  }

#ifndef NDEBUG
  // Verify that pushes and pops are performed in FIFO order.
  SmallVector<unsigned, 0> Stack;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : reverse(MI.explicit_operands())) {
        if (!MO.isReg()) continue;
        unsigned VReg = MO.getReg();

        if (MFI.isVRegStackified(VReg)) {
          if (MO.isDef())
            Stack.push_back(VReg);
          else
            assert(Stack.pop_back_val() == VReg);
        }
      }
    }
    // TODO: Generalize this code to support keeping values on the stack across
    // basic block boundaries.
    assert(Stack.empty());
  }
#endif

  return Changed;
}
