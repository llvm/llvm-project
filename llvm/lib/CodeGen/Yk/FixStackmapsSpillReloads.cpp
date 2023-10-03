//===-- FixStackmapsSpillReloads.cpp - Fix spills before stackmaps --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass fixes stackmaps in regards to spill reloads inserted by the
// register allocator. For example, if we have the LLVM IR
//
//   call foo($10, $11)
//   call llvm.experimental.stackmaps(1, 0, $8, $9)
//
// After register allocation we might get something like
//
//   movrr $rbx, $rsi
//   movmr $rbp, -8, $rdi
//   ...
//   call foo($rsi, $rdi)
//   movrr $rsi, $rbx
//   movrm $rdi, $rbp, -8
//   STACKMAP $rsi, $rdi
//
// In order to pass arguments to foo, the register allocator had to spill the
// values in $rdi and $rsi into another register or onto the stack before the
// call. Then immediately after the call it inserted instructions to reload
// the spilled values back into the original registers. Since during
// deoptimisation we return to immediately after the call, the stackmap is now
// tracking the wrong values, e.g. in this case $rdi and $rsi instead of the
// spill locations.
//
// This pass interates over all basic blocks, finds spill reloads inserted
// inbetween a call and stackmap, replaces the stackmap operands with the
// spill reloads, and then moves the stackmap instruction up just below the
// call.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "fix-stackmaps-spill-reloads"

namespace {

class FixStackmapsSpillReloads : public MachineFunctionPass {
public:
  static char ID;

  FixStackmapsSpillReloads() : MachineFunctionPass(ID) {
    initializeFixStackmapsSpillReloadsPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "Stackmaps Fix Post RegAlloc Pass";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // namespace

char FixStackmapsSpillReloads::ID = 0;
char &llvm::FixStackmapsSpillReloadsID = FixStackmapsSpillReloads::ID;

INITIALIZE_PASS_BEGIN(FixStackmapsSpillReloads, DEBUG_TYPE, "Fixup Stackmap Spills",
                      false, false)
INITIALIZE_PASS_END(FixStackmapsSpillReloads, DEBUG_TYPE, "Fixup Stackmap Spills",
                    false, false)


bool FixStackmapsSpillReloads::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  for (MachineBasicBlock &MBB : MF) {
    bool Collect = false;
    std::set<MachineInstr *> Erased;
    MachineInstr *LastCall = nullptr;
    std::map<Register, MachineInstr *> Spills;
    for (MachineInstr &MI : MBB) {
      if (MI.isCall() && !MI.isInlineAsm()) {
        // YKFIXME: Do we need to check for intrinsics here or have they been
        // removed during lowering?
        if (MI.getOpcode() != TargetOpcode::STACKMAP &&
            MI.getOpcode() != TargetOpcode::PATCHPOINT) {
          // If we see a normal function call we know it will be followed by a
          // STACKMAP instruction. Set `Collect` to `true` to collect all spill
          // reload instructions between this call and the STACKMAP instruction.
          // Also remember this call, so we can insert the new STACKMAP
          // instruction right below it.
          Collect = true;
          LastCall = &MI;
          Spills.clear();
          continue;
        }
      }

      if (MI.getOpcode() == TargetOpcode::STACKMAP) {
        if (LastCall == nullptr) {
          // There wasn't a call preceeding this stackmap, so this must be
          // attached to a branch instruction.
          continue;
        }
        Collect = false;
        // Assemble a new stackmap instruction by copying over the operands of
        // the old instruction to the new one, while replacing spilled operands
        // as we go.
        MachineInstr *NewMI =
          MF.CreateMachineInstr(TII->get(TargetOpcode::STACKMAP), MI.getDebugLoc(), true);
        MachineInstrBuilder MIB(MF, NewMI);
        // Copy ID and shadow
        auto *MOI = MI.operands_begin();
        MIB.add(*MOI); // ID
        MOI++;
        MIB.add(*MOI); // Shadow
        MOI++;
        while (MOI != MI.operands_end()) {
          if (MOI->isReg()) {
            Register Reg = MOI->getReg();
            // Check if the register operand in the stackmap is a restored
            // spill.
            // Since implicit operands are ignored by stackmaps (they are not
            // added into the record) we must not replace them with spills so
            // we don't add extra locations that aren't needed. Doing so leads
            // to bugs during deoptimisation.
            if (Spills.count(Reg) > 0 && !MOI->isImplicit()) {
              // Get spill reload instruction
              MachineInstr *SMI = Spills[Reg];
              int FI;
              if (TII->isCopyInstr(*SMI)) {
                // If the reload is a simple copy, e.g. $rax = $rbx,
                // just replace the stackmap operand with the source of the
                // copy instruction.
                MIB.add(SMI->getOperand(1));
              } else if (TII->isLoadFromStackSlotPostFE(*SMI, FI)) {
                // If the reload is a load from the stack, replace the operand
                // with multiple operands describing a stack location.
                std::optional<unsigned> Size = SMI->getRestoreSize(TII);
                if(!Size.has_value()) {
                  // This reload isn't a spill (e.g. this could be loading an
                  // argument passed via the stack), so we don't need to
                  // replace it. Since registers in lower frames aren't reset
                  // during deopt, this is only of consequence to the top stack
                  // frame. And even there this will simply temporarily put a
                  // value into the register MOI, only to then immediately
                  // reload the same value into MOI once the reload instruction
                  // `SMI` is executed after deopt returns to normal execution.
                  MIB.add(*MOI);
                } else {
                  MIB.addImm(StackMaps::IndirectMemRefOp);
                  MIB.addImm(Size.value()); // Size
                  MIB.add(SMI->getOperand(1)); // Register
                  MIB.add(SMI->getOperand(4)); // Offset
                }
              } else {
                assert(false && "Unknown instruction found");
              }
            } else {
              MIB.add(*MOI);
            }
            MOI++;
            continue;
          }
          // Copy all other operands over as is.
          MIB.add(*MOI);
          switch (MOI->getImm()) {
            default:
              llvm_unreachable("Unrecognized operand type.");
            case StackMaps::DirectMemRefOp: {
              MOI++;
              MIB.add(*MOI); // Register
              MOI++;
              MIB.add(*MOI); // Offset
              break;
            }
            case StackMaps::IndirectMemRefOp: {
              MOI++;
              MIB.add(*MOI); // Size
              MOI++;
              MIB.add(*MOI); // Register
              MOI++;
              MIB.add(*MOI); // Offset
              break;
            }
            case StackMaps::ConstantOp: {break;}
            case StackMaps::NextLive: {break;}
          }
          MOI++;
        }
        MachineOperand *Last = NewMI->operands_end() - 1;
        if (!Last->isReg() || !Last->isImplicit()) {
          // Unless the last operand is an implicit register, the last operand
          // needs to be `NextLive` due to the way stackmaps are parsed.
          MIB.addImm(StackMaps::NextLive);
        }
        // Insert the new stackmap instruction just after the last call.
        MI.getParent()->insertAfter(LastCall, NewMI);
        // Remember the old stackmap instruction for deletion later.
        Erased.insert(&MI);
        LastCall = nullptr;
        Changed = true;
      }

      // Collect spill reloads that appear between a call and its corresponding
      // STACKMAP instruction.
      if (Collect) {
        int FI;
        if (TII->isCopyInstr(MI) || TII->isLoadFromStackSlotPostFE(MI, FI)) {
          Spills[MI.getOperand(0).getReg()] = &MI;
        }
      }
    }
    // Remove old stackmap instructions.
    for (MachineInstr *E : Erased) {
      E->eraseFromParent();
    }
  }

  return Changed;
}
