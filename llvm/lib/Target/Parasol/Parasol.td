//=== Parasol.td - Describe the Parasol Target Machine ---------*- tablegen -*-===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// Top tablegen file that pulls in code from the LLVM framework along with
// code specific to this target.
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Calling Conv, Instruction Descriptions, Register information, etc
//===----------------------------------------------------------------------===//

include "ParasolSchedule.td"
include "ParasolRegisterInfo.td"
include "ParasolCallingConv.td"
include "ParasolInstrInfo.td"
include "ParasolRegisterBanks.td"

def ParasolInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Parasol processors supported
//===----------------------------------------------------------------------===//

def : ProcessorModel<"generic",  NoSchedModel, []>;

//===----------------------------------------------------------------------===//
// Parasol Target declaration
//===----------------------------------------------------------------------===//

def Parasol : Target {
  let InstructionSet = ParasolInstrInfo;
}
