//=- AArch64.td - Describe the AArch64 Target Machine --------*- tablegen -*-=//
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
// Target-independent interfaces which we are implementing.
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Subtarget features.
//===----------------------------------------------------------------------===//
include "AArch64Features.td"
include "AArch64FMV.td"

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "AArch64RegisterInfo.td"
include "AArch64RegisterBanks.td"
include "AArch64CallingConvention.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "AArch64Schedule.td"
include "AArch64InstrInfo.td"
include "AArch64SchedPredicates.td"
include "AArch64SchedPredExynos.td"
include "AArch64SchedPredNeoverse.td"
include "AArch64Combine.td"

defm : RemapAllTargetPseudoPointerOperands<GPR64sp>;

def AArch64InstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Named operands for MRS/MSR/TLBI/...
//===----------------------------------------------------------------------===//

include "AArch64SystemOperands.td"

//===----------------------------------------------------------------------===//
// LFI rewriter metadata consumed by AArch64MCLFIRewriter. Must follow
// AArch64SystemOperands.td due to use of SearchableTable.td.
//===----------------------------------------------------------------------===//

include "AArch64LFI.td"

//===----------------------------------------------------------------------===//
// AArch64 Processors supported.
//

include "AArch64SchedModels.td"

include "AArch64SchedA320.td"
include "AArch64SchedA53.td"
include "AArch64SchedA55.td"
include "AArch64SchedA510.td"
include "AArch64SchedA57.td"
include "AArch64SchedC1Nano.td"
include "AArch64SchedC1Ultra.td"
include "AArch64SchedC1Premium.td"
include "AArch64SchedCyclone.td"
include "AArch64SchedFalkor.td"
include "AArch64SchedKryo.td"
include "AArch64SchedExynosM3.td"
include "AArch64SchedExynosM4.td"
include "AArch64SchedExynosM5.td"
include "AArch64SchedThunderX.td"
include "AArch64SchedThunderX2T99.td"
include "AArch64SchedA64FX.td"
include "AArch64SchedThunderX3T110.td"
include "AArch64SchedTSV110.td"
include "AArch64SchedHIP12.td"
include "AArch64SchedAmpere1.td"
include "AArch64SchedAmpere1B.td"
include "AArch64SchedNeoverseN1.td"
include "AArch64SchedNeoverseN2.td"
include "AArch64SchedNeoverseN3.td"
include "AArch64SchedNeoverseV1.td"
include "AArch64SchedNeoverseV2.td"
include "AArch64SchedNeoverseV3.td"
include "AArch64SchedNeoverseV3AE.td"
include "AArch64SchedOlympus.td"
include "AArch64SchedOryon.td"

include "AArch64Processors.td"

//===----------------------------------------------------------------------===//
// Assembly parser
//===----------------------------------------------------------------------===//

def GenericAsmParserVariant : AsmParserVariant {
  int Variant = 0;
  string Name = "generic";
  string BreakCharacters = ".";
  string TokenizingCharacters = "[]*!/";
}

def AppleAsmParserVariant : AsmParserVariant {
  int Variant = 1;
  string Name = "apple-neon";
  string BreakCharacters = ".";
  string TokenizingCharacters = "[]*!/";
}

//===----------------------------------------------------------------------===//
// Assembly printer
//===----------------------------------------------------------------------===//
// AArch64 Uses the MC printer for asm output, so make sure the TableGen
// AsmWriter bits get associated with the correct class.
def GenericAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  int PassSubtarget = 1;
  int Variant = 0;
  bit isMCAsmWriter = 1;
}

def AppleAsmWriter : AsmWriter {
  let AsmWriterClassName = "AppleInstPrinter";
  int PassSubtarget = 1;
  int Variant = 1;
  int isMCAsmWriter = 1;
}

//===----------------------------------------------------------------------===//
// Target Declaration
//===----------------------------------------------------------------------===//

def AArch64 : Target {
  let InstructionSet = AArch64InstrInfo;
  let AssemblyParserVariants = [GenericAsmParserVariant, AppleAsmParserVariant];
  let AssemblyWriters = [GenericAsmWriter, AppleAsmWriter];
  let AllowRegisterRenaming = 1;
}

//===----------------------------------------------------------------------===//
// Pfm Counters
//===----------------------------------------------------------------------===//

include "AArch64PfmCounters.td"
