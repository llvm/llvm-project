//===-- AGXLowerControlFlow.cpp - Lower structured control flow -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Expands the structured control flow pseudos into execution mask
// instructions on the nesting counter r0l and jumps (like SILowerControlFlow):
//
//   AGX_IF c, %flow     if_icmp r0l, ne, c, 0, 1;   jmp_exec_none %flow
//   AGX_ELSE %end       else_fcmp r0l, eq, 0.0, 0.0, 1; jmp_exec_none %end
//   AGX_LOOP brk, %hdr  while_icmp r0l, eq, brk, 0, 1;  jmp_exec_any %hdr
//   AGX_LOOP_BEGIN      if_fcmp r0l, eq, 0.0, 0.0, 1 (push_exec 1)
//   AGX_END_CF          pop_exec r0l, 1
//   AGX_BR %bb          jmp_exec_any %bb; jmp_exec_none %bb
//
// The jumps only skip code no thread executes; the execution mask
// instructions alone implement the semantics. The nesting counter is zeroed
// at kernel entry.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "agx-lower-control-flow"

namespace {

class AGXLowerControlFlow : public MachineFunctionPass {
public:
  static char ID;
  AGXLowerControlFlow() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AGX Lower Control Flow";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // namespace

char AGXLowerControlFlow::ID = 0;

INITIALIZE_PASS(AGXLowerControlFlow, DEBUG_TYPE, "AGX Lower Control Flow",
                false, false)

FunctionPass *llvm::createAGXLowerControlFlowPass() {
  return new AGXLowerControlFlow();
}

// Integer conditions.
static constexpr unsigned ICondUEQ = 0;
static constexpr unsigned ICondNE = 8;
// Float condition eq, and the 8-bit float immediate 0.0.
static constexpr unsigned FCondEQ = 0;
static constexpr unsigned FPImmZero = 0;

/// Build an execution mask instruction: (nest, cc, a, b, n).
static MachineInstrBuilder buildExecOp(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       const DebugLoc &DL,
                                       const AGXInstrInfo &TII, unsigned Opc,
                                       unsigned CC) {
  return BuildMI(MBB, I, DL, TII.get(Opc)).addImm(0).addImm(CC);
}

bool AGXLowerControlFlow::runOnMachineFunction(MachineFunction &MF) {
  const AGXInstrInfo &TII = *MF.getSubtarget<AGXSubtarget>().getInstrInfo();
  bool HasControlFlow = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      MachineBasicBlock::iterator I = MI.getIterator();
      const DebugLoc &DL = MI.getDebugLoc();
      switch (MI.getOpcode()) {
      case AGX::AGX_IF: {
        buildExecOp(MBB, I, DL, TII, AGX::IF_ICMP, ICondNE)
            .addImm(0)
            .add(MI.getOperand(0))
            .addImm(0)
            .addImm(0)
            .addImm(1);
        BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_NONE))
            .addMBB(MI.getOperand(1).getMBB());
        break;
      }
      case AGX::AGX_ELSE:
        buildExecOp(MBB, I, DL, TII, AGX::ELSE_FCMP, FCondEQ)
            .addImm(0)
            .addImm(FPImmZero)
            .addImm(0)
            .addImm(FPImmZero)
            .addImm(1);
        BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_NONE))
            .addMBB(MI.getOperand(0).getMBB());
        break;
      case AGX::AGX_LOOP:
        // Threads stay in the loop while brk is zero.
        buildExecOp(MBB, I, DL, TII, AGX::WHILE_ICMP, ICondUEQ)
            .addImm(0)
            .add(MI.getOperand(0))
            .addImm(0)
            .addImm(0)
            .addImm(1);
        BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_ANY))
            .addMBB(MI.getOperand(1).getMBB());
        break;
      case AGX::AGX_LOOP_BEGIN:
        buildExecOp(MBB, I, DL, TII, AGX::IF_FCMP, FCondEQ)
            .addImm(0)
            .addImm(FPImmZero)
            .addImm(0)
            .addImm(FPImmZero)
            .addImm(1);
        break;
      case AGX::AGX_END_CF:
        BuildMI(MBB, I, DL, TII.get(AGX::POP_EXEC)).addImm(0).addImm(1);
        break;
      case AGX::AGX_BR: {
        MachineBasicBlock *Target = MI.getOperand(0).getMBB();
        if (!MBB.isLayoutSuccessor(Target)) {
          BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_ANY)).addMBB(Target);
          BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_NONE)).addMBB(Target);
        }
        MI.eraseFromParent();
        continue;
      }
      default:
        continue;
      }
      HasControlFlow = true;
      MI.eraseFromParent();
    }
  }

  if (HasControlFlow) {
    MachineBasicBlock &Entry = MF.front();
    BuildMI(Entry, Entry.begin(), DebugLoc(), TII.get(AGX::SET_NEST))
        .addImm(0);
  }
  return true;
}
