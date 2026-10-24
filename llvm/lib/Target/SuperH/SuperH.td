//===- SuperH.td - Describe the SuperH Target Machine ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is the top level entry point for the SuperH target.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"
include "llvm/TableGen/SearchableTable.td"

//===----------------------------------------------------------------------===//
// SuperH Subtarget features
//===----------------------------------------------------------------------===//

def FeatureFPU      : SubtargetFeature<"fp32", "HasFPU", "true",
                                       "Enable use of 32-bit floating point instructions.">;
def FeatureFPU64    : SubtargetFeature<"fp64", "HasFP64", "true",
                                       "Enable use of 64-bit floating point instructions.",
                                       [FeatureFPU]>;

def FeatureDSP      : SubtargetFeature<"dsp", "HasDSP", "true",
                                       "Enable SuperH DSP Extensions">;

def FeatureGPRBanks : SubtargetFeature<"gprbanks", "HasGPRBanks", "true",
                                       "Enable Banked General Purpose Registers">;

// NOTE:  SH-4A implicitly has FSCA and FSRRA, however some earlier processors
//        of the SH4 variant have a subset of SH-4A's features. As such those
//        undocumented features are exposed as individual flags.
def FeatureFSCA     : SubtargetFeature<"fsca", "HasFSCA", "true",
                                       "Enable use of fsca instruction.",
                                       [FeatureFPU]>;
def FeatureFSRRA    : SubtargetFeature<"fsrra", "HasFSRRA", "true",
                                       "Enable use of fsrra instruction.",
                                       [FeatureFPU]>;

//===----------------------------------------------------------------------===//
// SuperH CPU Family features
//===----------------------------------------------------------------------===//

def FeatureSH2      : SubtargetFeature<"sh2", "IsSH2", "true",
                          "SH-2 ISA Support",
                          []>;
def FeatureSH2A     : SubtargetFeature<"sh2a", "IsSH2A", "true",
                          "SH-2A ISA Support",
                          [FeatureSH2]>;
def FeatureSH3      : SubtargetFeature<"sh3", "IsSH3", "true",
                          "SH-3 ISA Support",
                          [FeatureSH2]>;
def FeatureSH3E     : SubtargetFeature<"sh3e", "IsSH3E", "true",
                          "SH-3E ISA Support",
                          [FeatureSH3, FeatureFPU]>;
def FeatureSH4      : SubtargetFeature<"sh4", "IsSH4", "true",
                          "SH-4 ISA Support",
                          [FeatureSH3, FeatureFPU, FeatureFPU64,
                           FeatureGPRBanks]>;
def FeatureSH4A     : SubtargetFeature<"sh4a", "IsSH4A", "true",
                          "SH-4A ISA Support",
                          [FeatureSH4, FeatureFSCA, FeatureFSRRA]>;

//===--------------------------------------------------------------------------===//
// Predicates
//===--------------------------------------------------------------------------===//

def IsSH2      : Predicate<"Subtarget->isSH2()">,
                  AssemblerPredicate<(all_of FeatureSH2)>;
def IsSH2A     : Predicate<"Subtarget->isSH2A()">,
                  AssemblerPredicate<(all_of FeatureSH2A)>;
def IsSH3      : Predicate<"Subtarget->isSH3()">,
                  AssemblerPredicate<(all_of FeatureSH3)>;
def IsSH3E     : Predicate<"Subtarget->isSH3E()">,
                  AssemblerPredicate<(all_of FeatureSH3E)>;
def IsSH4      : Predicate<"Subtarget->isSH4()">,
                  AssemblerPredicate<(all_of FeatureSH4)>;
def IsSH4A     : Predicate<"Subtarget->isSH4A()">,
                  AssemblerPredicate<(all_of FeatureSH4A)>;

def HasFSCA     : Predicate<"Subtarget->hasFSCA()">,
                  AssemblerPredicate<(all_of FeatureFPU)>;
def HasFSRRA    : Predicate<"Subtarget->hasFSRRA()">,
                  AssemblerPredicate<(all_of FeatureFPU)>;
def HasFPU      : Predicate<"Subtarget->hasFPU()">,
                  AssemblerPredicate<(all_of FeatureFPU)>;
def HasF64      : Predicate<"Subtarget->hasFP64()">,
                  AssemblerPredicate<(all_of FeatureFPU)>;

def HasDSP      : Predicate<"Subtarget->hasDSP()">,
                  AssemblerPredicate<(all_of FeatureDSP)>;

//===----------------------------------------------------------------------===//
// SuperH Processors
//===----------------------------------------------------------------------===//

include "SuperHSchedule.td"

class Proc<string n, list<SubtargetFeature> f, list<SubtargetFeature> tunef = []>
  : Processor<n, NoItineraries, f, tunef>;

class ProcModel<string Name, list<SubtargetFeature> Features>
    : ProcessorModel<Name, GenericSuperHModel, Features>;

def : ProcModel<"generic", [FeatureSH2]>;
def : ProcModel<"sh1", []>;
def : ProcModel<"sh2", [FeatureSH2]>;
def : ProcModel<"sh2e", [FeatureSH2, FeatureFPU]>;
def : ProcModel<"sh2a", [FeatureSH2A]>;
def : ProcModel<"sh3", [FeatureSH3]>;
def : ProcModel<"sh3e", [FeatureSH3E]>;
def : ProcModel<"sh4", [FeatureSH4]>;
def : ProcModel<"sh4a", [FeatureSH4A]>;

// NOTE:  SEGA got some special variants of the SuperH family
//        produced with some extra features enabled, defining them here.
def : Proc<"sh7091",  [FeatureSH4, FeatureFSCA, FeatureFSRRA]>;
def : Proc<"sh7095",  [FeatureSH2]>;

// All of the publicly available CPU cores.
def : Proc<"sh7020",  []>;
def : Proc<"sh7021",  []>;
def : Proc<"sh7131",  [FeatureSH2]>;
def : Proc<"sh7132",  [FeatureSH2]>;
def : Proc<"sh7136",  [FeatureSH2]>;
def : Proc<"sh7137",  [FeatureSH2]>;
def : Proc<"sh7146",  [FeatureSH2]>;
def : Proc<"sh7149",  [FeatureSH2]>;
def : Proc<"sh7124",  [FeatureSH2, FeatureDSP]>;
def : Proc<"sh7125",  [FeatureSH2, FeatureDSP]>;
def : Proc<"sh7201",  [FeatureSH2A, FeatureFPU]>;
def : Proc<"sh7203",  [FeatureSH2A, FeatureFPU, FeatureFPU64]>;
def : Proc<"sh7205",  [FeatureSH2A]>;
def : Proc<"sh7206",  [FeatureSH2]>;
def : Proc<"sh7211",  [FeatureSH2A]>;
def : Proc<"sh7214",  [FeatureSH2A]>;
def : Proc<"sh7216",  [FeatureSH2A, FeatureFPU]>;
def : Proc<"sh7231",  [FeatureSH2A, FeatureFPU]>;
def : Proc<"sh7237",  [FeatureSH2A]>;
def : Proc<"sh7239",  [FeatureSH2A, FeatureFPU]>;
def : Proc<"sh7243",  [FeatureSH2A, FeatureGPRBanks]>;
def : Proc<"sh7720",  [FeatureSH3, FeatureDSP]>;
def : Proc<"sh7721",  [FeatureSH3, FeatureDSP]>;
def : Proc<"sh7750",  [FeatureSH4]>;
def : Proc<"sh7751",  [FeatureSH4]>;
def : Proc<"sh7780",  [FeatureSH4A]>;
def : Proc<"sh7785",  [FeatureSH4A]>;
def : Proc<"sh7731",  [FeatureSH4A, FeatureDSP]>;
def : Proc<"sh7734",  [FeatureSH4A]>;
def : Proc<"sh7722",  [FeatureSH4A, FeatureDSP]>;
def : Proc<"sh7723",  [FeatureSH4A]>;
def : Proc<"sh7724",  [FeatureSH4A]>;

// TODO:  Add more families from this listing
//        https://www.renesas.com/en/document/gde/superh-risc-engine-family-product-part-number-guide?r=1055111

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "SuperHRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "SuperHInstrInfo.td"

defm : RemapAllTargetPseudoPointerOperands<sh_ptr_rc>;

def SuperHInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Calling Conventions
//===----------------------------------------------------------------------===//

include "SuperHCallingConv.td"

//===---------------------------------------------------------------------===//
// Assembly Printers
//===---------------------------------------------------------------------===//

def SuperHAsmWriter : AsmWriter {
 string AsmWriterClassName = "InstPrinter";
 bit isMCAsmWriter = 1;
}

//===---------------------------------------------------------------------===//
// Assembly Parsers
//===---------------------------------------------------------------------===//

def SuperHAsmParser : AsmParser {
  let ShouldEmitMatchRegisterName = 1;
  let ShouldEmitMatchRegisterAltName = 0;
}

def SuperHAsmParserVariant : AsmParserVariant {
  let Variant = 0;
  let Name = "GCC";
  let TokenizingCharacters = "()+-@#";
  let CommentDelimiter = "!";
}

//===----------------------------------------------------------------------===//
// Target Declaration
//===----------------------------------------------------------------------===//

def SuperH : Target {
  let InstructionSet = SuperHInstrInfo;
  let AssemblyParsers = [SuperHAsmParser];
  let AssemblyParserVariants = [SuperHAsmParserVariant];
  let AssemblyWriters = [SuperHAsmWriter];
}