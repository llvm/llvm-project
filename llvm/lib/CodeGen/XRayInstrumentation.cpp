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

private:
  // Replace the original RET instruction with the exit sled code ("patchable
  //   ret" pseudo-instruction), so that at runtime XRay can replace the sled
  //   with a code jumping to XRay trampoline, which calls the tracing handler
  //   and, in the end, issues the RET instruction.
  // This is the approach to go on CPUs which have a single RET instruction,
  //   like x86/x86_64.
  void replaceRetWithPatchableRet(MachineFunction &MF,
    const TargetInstrInfo *TII);
  // Prepend the original return instruction with the exit sled code ("patchable
  //   function exit" pseudo-instruction), preserving the original return
  //   instruction just after the exit sled code.
  // This is the approach to go on CPUs which have multiple options for the
  //   return instruction, like ARM. For such CPUs we can't just jump into the
  //   XRay trampoline and issue a single return instruction there. We rather
  //   have to call the trampoline and return from it to the original return
  //   instruction of the function being instrumented.
  void prependRetWithPatchableExit(MachineFunction &MF,
    const TargetInstrInfo *TII);
};
} // anonymous namespace

void XRayInstrumentation::replaceRetWithPatchableRet(MachineFunction &MF,
  const TargetInstrInfo *TII)
{
  // We look for *all* terminators and returns, then replace those with
  // PATCHABLE_RET instructions.
  SmallVector<MachineInstr *, 4> Terminators;
  for (auto &MBB : MF) {
    for (auto &T : MBB.terminators()) {
      unsigned Opc = 0;
      if (T.isReturn() && T.getOpcode() == TII->getReturnOpcode()) {
        // Replace return instructions with:
        //   PATCHABLE_RET <Opcode>, <Operand>...
        Opc = TargetOpcode::PATCHABLE_RET;
      }
      if (TII->isTailCall(T)) {
        // Treat the tail call as a return instruction, which has a
        // different-looking sled than the normal return case.
        Opc = TargetOpcode::PATCHABLE_TAIL_CALL;
      }
      if (Opc != 0) {
        auto MIB = BuildMI(MBB, T, T.getDebugLoc(), TII->get(Opc))
                       .addImm(T.getOpcode());
        for (auto &MO : T.operands())
          MIB.addOperand(MO);
        Terminators.push_back(&T);
      }
    }
  }

  for (auto &I : Terminators)
    I->eraseFromParent();
}

void XRayInstrumentation::prependRetWithPatchableExit(MachineFunction &MF,
  const TargetInstrInfo *TII)
{
  for (auto &MBB : MF) {
    for (auto &T : MBB.terminators()) {
      if (T.isReturn()) {
        // Prepend the return instruction with PATCHABLE_FUNCTION_EXIT
        BuildMI(MBB, T, T.getDebugLoc(),
                TII->get(TargetOpcode::PATCHABLE_FUNCTION_EXIT));
      }
    }
  }
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

  auto &FirstMBB = *MF.begin();
  auto &FirstMI = *FirstMBB.begin();

  if (!MF.getSubtarget().isXRaySupported()) {
    FirstMI.emitError("An attempt to perform XRay instrumentation for an"
      " unsupported target.");
    return false;
  }

  // FIXME: Do the loop triviality analysis here or in an earlier pass.

  // First, insert an PATCHABLE_FUNCTION_ENTER as the first instruction of the
  // MachineFunction.
  auto *TII = MF.getSubtarget().getInstrInfo();
  BuildMI(FirstMBB, FirstMI, FirstMI.getDebugLoc(),
          TII->get(TargetOpcode::PATCHABLE_FUNCTION_ENTER));

  switch (MF.getTarget().getTargetTriple().getArch()) {
  case Triple::ArchType::arm:
  case Triple::ArchType::thumb:
    // For the architectures which don't have a single return instruction
    prependRetWithPatchableExit(MF, TII);
    break;
  default:
    // For the architectures that have a single return instruction (such as
    //   RETQ on x86_64).
    replaceRetWithPatchableRet(MF, TII);
    break;
  }
  return true;
}

char XRayInstrumentation::ID = 0;
char &llvm::XRayInstrumentationID = XRayInstrumentation::ID;
INITIALIZE_PASS(XRayInstrumentation, "xray-instrumentation", "Insert XRay ops",
                false, false)
