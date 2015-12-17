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
/// This is primarily a code size optimization, since temporary values on the
/// expression don't need to be named.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h" // for WebAssembly::ARGUMENT_*
#include "WebAssemblyMachineFunctionInfo.h"
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
// expression stack ordering constraints needed for an instruction which is
// consumed by an instruction using the expression stack.
static void ImposeStackInputOrdering(MachineInstr *MI) {
  // Write the opaque EXPR_STACK register.
  if (!MI->definesRegister(WebAssembly::EXPR_STACK))
    MI->addOperand(MachineOperand::CreateReg(WebAssembly::EXPR_STACK,
                                             /*isDef=*/true,
                                             /*isImp=*/true));
}

// Decorate the given instruction with implicit operands that enforce the
// expression stack ordering constraints for an instruction which is on
// the expression stack.
static void ImposeStackOrdering(MachineInstr *MI, MachineRegisterInfo &MRI) {
  ImposeStackInputOrdering(MI);

  // Also read the opaque EXPR_STACK register.
  if (!MI->readsRegister(WebAssembly::EXPR_STACK))
    MI->addOperand(MachineOperand::CreateReg(WebAssembly::EXPR_STACK,
                                             /*isDef=*/false,
                                             /*isImp=*/true));

  // Also, mark any inputs to this instruction as being consumed by an
  // instruction on the expression stack.
  // TODO: Find a lighter way to describe the appropriate constraints.
  for (MachineOperand &MO : MI->uses()) {
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    MachineInstr *Def = MRI.getVRegDef(Reg);
    if (Def->getOpcode() == TargetOpcode::PHI)
      continue;
    ImposeStackInputOrdering(Def);
  }
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
  assert(Def->getParent() == Insert->getParent());
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

  assert(MRI.isSSA() && "RegStackify depends on SSA form");

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
      // operands off the stack in LIFO order.
      bool AnyStackified = false;
      for (MachineOperand &Op : reverse(Insert->uses())) {
        // We're only interested in explicit virtual register operands.
        if (!Op.isReg() || Op.isImplicit() || !Op.isUse())
          continue;

        unsigned Reg = Op.getReg();
        if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
          // An instruction with a physical register. Conservatively mark it as
          // an expression stack input so that it isn't reordered with anything
          // in an expression stack which might use it (physical registers
          // aren't in SSA form so it's not trivial to determine this).
          // TODO: Be less conservative.
          ImposeStackInputOrdering(Insert);
          continue;
        }

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

        // Single-use expression trees require defs that have one use.
        // TODO: Eventually we'll relax this, to take advantage of set_local
        // returning its result.
        if (!MRI.hasOneUse(Reg))
          continue;

        // For now, be conservative and don't look across block boundaries.
        // TODO: Be more aggressive.
        if (Def->getParent() != &MBB)
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
        ImposeStackOrdering(Def, MRI);
        Insert = Def;
      }
      if (AnyStackified)
        ImposeStackOrdering(&MI, MRI);
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
        if (!MO.isReg())
          continue;
        unsigned VReg = MO.getReg();

        // Don't stackify physregs like SP or FP.
        if (!TargetRegisterInfo::isVirtualRegister(VReg))
          continue;

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
