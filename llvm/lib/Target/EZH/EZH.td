//===- EZH.td - Describe the EZH Target Machine --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//

include "EZHSchedule.td"
include "EZHRegisterInfo.td"
include "EZHCallingConv.td"
include "EZHInstrInfo.td"

defm : RemapAllTargetPseudoPointerOperands<GPR>;

def EZHInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// EZH Subtarget features.
//===----------------------------------------------------------------------===//

def FeatureBitSliceInterrupts : SubtargetFeature<"bitslice-interrupts", "HasBitSliceInterrupts", "true",
                                                 "Enable EZH bitslice interrupt workaround (injection of e_gotol_bs before branches/calls)">;

//===----------------------------------------------------------------------===//
// EZH processors supported.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"generic", EZHSchedModel, [FeatureBitSliceInterrupts]>;
def : ProcessorModel<"v11", EZHSchedModel, [FeatureBitSliceInterrupts]>;

def EZHInstPrinter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def EZH : Target {
  // Pull in Instruction Info:
  let InstructionSet = EZHInstrInfo;
  let AssemblyWriters = [EZHInstPrinter];
}
