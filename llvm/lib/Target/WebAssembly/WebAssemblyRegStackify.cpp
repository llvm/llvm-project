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
#include "WebAssemblySubtarget.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
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
    AU.addRequired<LiveIntervals>();
    AU.addPreserved<MachineBlockFrequencyInfo>();
    AU.addPreserved<SlotIndexes>();
    AU.addPreserved<LiveIntervals>();
    AU.addPreservedID(MachineDominatorsID);
    AU.addPreservedID(LiveVariablesID);
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
// expression stack ordering constraints for an instruction which is on
// the expression stack.
static void ImposeStackOrdering(MachineInstr *MI) {
  // Write the opaque EXPR_STACK register.
  if (!MI->definesRegister(WebAssembly::EXPR_STACK))
    MI->addOperand(MachineOperand::CreateReg(WebAssembly::EXPR_STACK,
                                             /*isDef=*/true,
                                             /*isImp=*/true));

  // Also read the opaque EXPR_STACK register.
  if (!MI->readsRegister(WebAssembly::EXPR_STACK))
    MI->addOperand(MachineOperand::CreateReg(WebAssembly::EXPR_STACK,
                                             /*isDef=*/false,
                                             /*isImp=*/true));
}

// Test whether it's safe to move Def to just before Insert.
// TODO: Compute memory dependencies in a way that doesn't require always
// walking the block.
// TODO: Compute memory dependencies in a way that uses AliasAnalysis to be
// more precise.
static bool IsSafeToMove(const MachineInstr *Def, const MachineInstr *Insert,
                         AliasAnalysis &AA, LiveIntervals &LIS,
                         MachineRegisterInfo &MRI) {
  assert(Def->getParent() == Insert->getParent());
  bool SawStore = false, SawSideEffects = false;
  MachineBasicBlock::const_iterator D(Def), I(Insert);

  // Check for register dependencies.
  for (const MachineOperand &MO : Def->operands()) {
    if (!MO.isReg() || MO.isUndef())
      continue;
    unsigned Reg = MO.getReg();

    // If the register is dead here and at Insert, ignore it.
    if (MO.isDead() && Insert->definesRegister(Reg) &&
        !Insert->readsRegister(Reg))
      continue;

    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      // If the physical register is never modified, ignore it.
      if (!MRI.isPhysRegModified(Reg))
        continue;
      // Otherwise, it's a physical register with unknown liveness.
      return false;
    }

    // Ask LiveIntervals whether moving this virtual register use or def to
    // Insert will change value numbers are seen.
    const LiveInterval &LI = LIS.getInterval(Reg);
    VNInfo *DefVNI =
        MO.isDef() ? LI.getVNInfoAt(LIS.getInstructionIndex(Def).getRegSlot())
                   : LI.getVNInfoBefore(LIS.getInstructionIndex(Def));
    assert(DefVNI && "Instruction input missing value number");
    VNInfo *InsVNI = LI.getVNInfoBefore(LIS.getInstructionIndex(Insert));
    if (InsVNI && DefVNI != InsVNI)
      return false;
  }

  // Check for memory dependencies and side effects.
  for (--I; I != D; --I)
    SawSideEffects |= !I->isSafeToMove(&AA, SawStore);
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
  const auto *TII = MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  const auto *TRI = MF.getSubtarget<WebAssemblySubtarget>().getRegisterInfo();
  AliasAnalysis &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  LiveIntervals &LIS = getAnalysis<LiveIntervals>();

  // Walk the instructions from the bottom up. Currently we don't look past
  // block boundaries, and the blocks aren't ordered so the block visitation
  // order isn't significant, but we may want to change this in the future.
  for (MachineBasicBlock &MBB : MF) {
    // Don't use a range-based for loop, because we modify the list as we're
    // iterating over it and the end iterator may change.
    for (auto MII = MBB.rbegin(); MII != MBB.rend(); ++MII) {
      MachineInstr *Insert = &*MII;
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

        // Only consider registers with a single definition.
        // TODO: Eventually we may relax this, to stackify phi transfers.
        MachineInstr *Def = MRI.getUniqueVRegDef(Reg);
        if (!Def)
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

        if (MRI.hasOneUse(Reg) && Def->getParent() == &MBB &&
            IsSafeToMove(Def, Insert, AA, LIS, MRI)) {
          // A single-use def in the same block with no intervening memory or
          // register dependencies; move the def down and nest it with the
          // current instruction.
          // TODO: Stackify multiple-use values, taking advantage of set_local
          // returning its result.
          Changed = true;
          AnyStackified = true;
          MBB.splice(Insert, &MBB, Def);
          LIS.handleMove(Def);
          MFI.stackifyVReg(Reg);
          ImposeStackOrdering(Def);
          Insert = Def;
        } else if (Def->isAsCheapAsAMove() &&
                   TII->isTriviallyReMaterializable(Def, &AA)) {
          // A trivially cloneable instruction; clone it and nest the new copy
          // with the current instruction.
          Changed = true;
          AnyStackified = true;
          unsigned OldReg = Def->getOperand(0).getReg();
          unsigned NewReg = MRI.createVirtualRegister(MRI.getRegClass(OldReg));
          TII->reMaterialize(MBB, Insert, NewReg, 0, Def, *TRI);
          Op.setReg(NewReg);
          MachineInstr *Clone =
              &*std::prev(MachineBasicBlock::instr_iterator(Insert));
          LIS.InsertMachineInstrInMaps(Clone);
          LIS.createAndComputeVirtRegInterval(NewReg);
          MFI.stackifyVReg(NewReg);
          ImposeStackOrdering(Clone);
          Insert = Clone;

          // If that was the last use of the original, delete the original.
          // Otherwise shrink the LiveInterval.
          if (MRI.use_empty(OldReg)) {
            SlotIndex Idx = LIS.getInstructionIndex(Def).getRegSlot();
            LIS.removePhysRegDefAt(WebAssembly::ARGUMENTS, Idx);
            LIS.removeVRegDefAt(LIS.getInterval(OldReg), Idx);
            LIS.removeInterval(OldReg);
            LIS.RemoveMachineInstrFromMaps(Def);
            Def->eraseFromParent();
          } else {
            LIS.shrinkToUses(&LIS.getInterval(OldReg));
          }
        }
      }
      if (AnyStackified)
        ImposeStackOrdering(&*MII);
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
  // Verify that pushes and pops are performed in LIFO order.
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
