//===-- LX32.td - LX32 Target Machine Description -------------------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This is the top-level TableGen file for the LX32 target.  It performs three
// tasks:
//
//   1. Includes the LLVM target infrastructure (Target.td) so that base
//      classes like Target, InstrInfo, RegisterClass, etc. are available.
//
//   2. Includes the four LX32-specific sub-files in dependency order:
//        LX32RegisterInfo.td  — registers and register classes
//        LX32InstrFormats.td  — instruction encoding base classes
//        LX32InstrInfo.td     — instruction definitions, patterns, pseudos
//        LX32CallingConv.td   — calling convention and callee-saved sets
//      The order matters: each file may reference definitions from the files
//      listed before it.
//
//   3. Defines the top-level target objects (LX32InstrInfo and LX32) that
//      llvm-tblgen uses to drive code generation.
//
// File should remain a thin coordinator.  No instruction definitions,
// register definitions, or patterns belong here.
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Target-specific includes — order is significant
//===----------------------------------------------------------------------===//

include "LX32RegisterInfo.td"
include "LX32InstrFormats.td"
include "LX32InstrInfo.td"
include "LX32CallingConv.td"

// Minimal processor model so TableGen emits a non-empty CPU table and
// `-mcpu=generic` is recognized by MCSubtargetInfo.
def : ProcessorModel<"generic", NoSchedModel, []>;

//===----------------------------------------------------------------------===//
// Instruction set descriptor
//
// LX32InstrInfo wraps the instruction definitions for use by the Target
// record below.  guessInstructionProperties = 0 disables TableGen's
// heuristic inference of mayLoad / mayStore / hasSideEffects — every
// instruction must declare these attributes explicitly, which avoids
// silent mis-classification and keeps the backend predictable.
//===----------------------------------------------------------------------===//

def LX32InstrInfo : InstrInfo {
  let guessInstructionProperties = 0;
}

def LX32AsmParser : AsmParser {
  let ShouldEmitMatchRegisterName = 1;
}

def LX32AsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

//===----------------------------------------------------------------------===//
// Pointer-operand remapping
//
// RemapAllTargetPseudoPointerOperands rewrites the pointer-typed operands of
// LLVM's built-in pseudo-instructions (LOAD_STACK_GUARD, PREALLOCATED_SETUP,
// etc.) to use the LX32 GPR register class.  Without this, those pseudos
// carry LLVM's default "ptr" operand type, which would be 64-bit on a
// 64-bit host and would mismatch the 32-bit lx32 register class.
//===----------------------------------------------------------------------===//

defm : RemapAllTargetPseudoPointerOperands<GPR>;

//===----------------------------------------------------------------------===//
// Target record
//
// LX32 ties the instruction set to the target and enables register renaming.
// AllowRegisterRenaming = 1 permits the register allocator to rename
// registers to reduce false dependencies — safe for an in-order machine
// without register-file constraints.
//===----------------------------------------------------------------------===//

def LX32 : Target {
  let InstructionSet      = LX32InstrInfo;
  let AssemblyParsers     = [LX32AsmParser];
  let AssemblyWriters     = [LX32AsmWriter];
  let AllowRegisterRenaming = 1;
}
