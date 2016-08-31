//===-- XRayInstrumentation.cpp - Adds XRay instrumentation to functions. -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a MachineFunctionPass that inserts the appropriate
// XRay instrumentation instructions. We look for XRay-specific attributes
// on the function to determine whether we should insert the replacement
// operations.
//
//===---------------------------------------------------------------------===//

#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

namespace {
struct XRayInstrumentation : public MachineFunctionPass {
  static char ID;

  XRayInstrumentation() : MachineFunctionPass(ID) {
    initializeXRayInstrumentationPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};
}

bool XRayInstrumentation::runOnMachineFunction(MachineFunction &MF) {
  auto &F = *MF.getFunction();
  auto InstrAttr = F.getFnAttribute("function-instrument");
  bool AlwaysInstrument = !InstrAttr.hasAttribute(Attribute::None) &&
                          InstrAttr.isStringAttribute() &&
                          InstrAttr.getValueAsString() == "xray-always";
  Attribute Attr = F.getFnAttribute("xray-instruction-threshold");
  unsigned XRayThreshold = 0;
  if (!AlwaysInstrument) {
    if (Attr.hasAttribute(Attribute::None) || !Attr.isStringAttribute())
      return false; // XRay threshold attribute not found.
    if (Attr.getValueAsString().getAsInteger(10, XRayThreshold))
      return false; // Invalid value for threshold.
    if (F.size() < XRayThreshold)
      return false; // Function is too small.
  }

  // FIXME: Do the loop triviality analysis here or in an earlier pass.

  // First, insert an PATCHABLE_FUNCTION_ENTER as the first instruction of the
  // MachineFunction.
  auto &FirstMBB = *MF.begin();
  auto &FirstMI = *FirstMBB.begin();
  auto *TII = MF.getSubtarget().getInstrInfo();
  BuildMI(FirstMBB, FirstMI, FirstMI.getDebugLoc(),
          TII->get(TargetOpcode::PATCHABLE_FUNCTION_ENTER));

  // Then we look for *all* terminators and returns, then replace those with
  // PATCHABLE_RET instructions.
  SmallVector<MachineInstr *, 4> Terminators;
  for (auto &MBB : MF) {
    for (auto &T : MBB.terminators()) {
      // FIXME: Handle tail calls here too?
      if (T.isReturn() && T.getOpcode() == TII->getReturnOpcode()) {
        // Replace return instructions with:
        //   PATCHABLE_RET <Opcode>, <Operand>...
        auto MIB = BuildMI(MBB, T, T.getDebugLoc(),
                           TII->get(TargetOpcode::PATCHABLE_RET))
                       .addImm(T.getOpcode());
        for (auto &MO : T.operands())
          MIB.addOperand(MO);
        Terminators.push_back(&T);
      }
    }
  }

  for (auto &I : Terminators)
    I->eraseFromParent();

  return true;
}

char XRayInstrumentation::ID = 0;
char &llvm::XRayInstrumentationID = XRayInstrumentation::ID;
INITIALIZE_PASS(XRayInstrumentation, "xray-instrumentation", "Insert XRay ops",
                false, false)
