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

//===----------------------------------------------------------------------===//
// SuperH Subtarget features
//===----------------------------------------------------------------------===//

// NOTE:  Some SH4 CPUs support FSCA and FSRRA despite their
//        ISA manuals not specifying so.
def FeatureFSCA     : SubtargetFeature<"fsca", "HasFSCA", "true",
                                       "Enable use of fsca instruction.">;
def FeatureFSRRA    : SubtargetFeature<"fsrra", "HasFSRRA", "true",
                                       "Enable use of fsrra instruction.">;
def FeatureFP32     : SubtargetFeature<"fp32", "HasFP32", "true",
                                       "Enable use of 32-bit floating point instructions.">;
def FeatureFP64     : SubtargetFeature<"fp64", "HasFP64", "true",
                                       "Enable use of 64-bit floating point instructions.">;
def FeatureDSP      : SubtargetFeature<"dsp", "HasDSP", "true",
                                       "Enable SuperH DSP Extensions">;

//===----------------------------------------------------------------------===//
// SuperH CPU Family features
//===----------------------------------------------------------------------===//

def FeatureSH1      : SubtargetFeature<"sh1", "SHArchVersion", "SH1",
                          "SH-1 ISA Support">;
def FeatureSH2      : SubtargetFeature<"sh2", "SHArchVersion", "SH2",
                          "SH-2 ISA Support",
                          [FeatureSH1]>;
def FeatureSH2E     : SubtargetFeature<"sh2e", "SHArchVersion", "SH2E",
                          "SH-2E ISA Support",
                          [FeatureSH1, FeatureSH2, FeatureFP32]>;
def FeatureSH2A     : SubtargetFeature<"sh2a", "SHArchVersion", "SH2A",
                          "SH-2A ISA Support",
                          [FeatureSH1, FeatureSH2, FeatureFP32, FeatureFP64]>;
def FeatureSH3      : SubtargetFeature<"sh3", "SHArchVersion", "SH3",
                          "SH-3 ISA Support",
                          [FeatureSH1, FeatureSH2]>;
def FeatureSH3E     : SubtargetFeature<"sh3e", "SHArchVersion", "SH3E",
                          "SH-3E ISA Support",
                          [FeatureSH1, FeatureSH2, FeatureFP32]>;
def FeatureSH4      : SubtargetFeature<"sh4", "SHArchVersion", "SH4",
                          "SH-4 ISA Support",
                          [FeatureSH1, FeatureSH2, FeatureSH3, FeatureFP32, 
                           FeatureFP64]>;
def FeatureSH4A     : SubtargetFeature<"sh4a", "SHArchVersion", "SH4A",
                          "SH-4A ISA Support",
                          [FeatureSH1, FeatureSH2, FeatureSH3, FeatureSH4, 
                           FeatureFP32, FeatureFP64, FeatureFSCA, FeatureFSRRA]>;

//===----------------------------------------------------------------------===//
// SuperH Processors
//===----------------------------------------------------------------------===//

include "SuperHSchedule.td"

class Proc<string Name, list<SubtargetFeature> Features>
    : ProcessorModel<Name, GenericSuperHModel, Features>;

def : Proc<"generic", [FeatureSH4]>;
def : Proc<"sh1", [FeatureSH1]>;
def : Proc<"sh2", [FeatureSH2]>;
def : Proc<"sh3", [FeatureSH3]>;
def : Proc<"sh4", [FeatureSH4]>;
def : Proc<"sh4a", [FeatureSH4A]>;

// NOTE:  SEGA got some special variants of the SuperH family
//        produced with some extra features enabled, defining them here.
def : Proc<"dreamcast", [FeatureSH4, FeatureFSCA, FeatureFSRRA]>;
def : Proc<"naomi",     [FeatureSH4, FeatureFSCA, FeatureFSRRA]>;
def : Proc<"saturn",    [FeatureSH2]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "SuperHRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "SuperHInstrInfo.td"
def SuperHInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Calling Conventions
//===----------------------------------------------------------------------===//


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
  let ShouldEmitMatchRegisterName = 0;
  let ShouldEmitMatchRegisterAltName = 0;
}

def SuperHAsmParserVariant : AsmParserVariant {
  int Variant = 0;
}

//===----------------------------------------------------------------------===//
// Target Declaration
//===----------------------------------------------------------------------===//

def SuperH : Target {
  let InstructionSet = SuperHInstrInfo;
  let AssemblyParsers = [SuperHAsmParser];
  let AssemblyWriters = [SuperHAsmWriter];
}