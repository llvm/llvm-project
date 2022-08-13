//===-- M88k.td - Describe the M88000 Target Machine -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// M88k subtarget features.
//===----------------------------------------------------------------------===//

def FeatureGraphics : SubtargetFeature<"graphics", "HasGraphics", "true",
                                       "has graphics instructions", []>;

def Proc88100       : SubtargetFeature<"mc88100", "M88kProc", "MC88100",
                                       "Motorola 88100 processor", []>;
def Proc88110       : SubtargetFeature<"mc88110", "M88kProc", "MC88110",
                                       "Motorola 88110 processor",
                                       [FeatureGraphics]>;

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//

include "M88kRegisterInfo.td"
include "GISel/M88kRegisterBanks.td"
include "M88kCallingConv.td"
include "M88kInstrFormats.td"
include "M88kInstrInfo.td"
include "M88kCombine.td"
include "M88kSchedule.td"

//===----------------------------------------------------------------------===//
// M88k processors supported.
//===----------------------------------------------------------------------===//

def : ProcessorModel<"mc88100", M88100SchedModel, [Proc88100]>;
def : ProcessorModel<"mc88110", M88110SchedModel, [Proc88110]>;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def M88kInstrInfo : InstrInfo;
def M88kAsmParser : AsmParser;
def M88kAsmParserVariant : AsmParserVariant {
  let RegisterPrefix = "%";
  let TokenizingCharacters = "[]<>*!";
}

def M88kAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  int PassSubtarget = 1;
  int Variant = 0;
}

def M88k : Target {
  let InstructionSet = M88kInstrInfo;
  let AssemblyParsers  = [M88kAsmParser];
  let AssemblyParserVariants = [M88kAsmParserVariant];
  let AssemblyWriters = [M88kAsmWriter];
  let AllowRegisterRenaming = 1;
}
