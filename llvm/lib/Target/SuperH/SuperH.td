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

def FeatureFPU      : SubtargetFeature<"fpu", "HasFPU", "false",
                          "Enable FPU Support">;
def FeatureVFPU     : SubtargetFeature<"vfpu", "HasVFPU", "true",
                          "Enable vector FPU instructions">;
def FeatureSH1      : SubtargetFeature<"sh1", "SHArchVersion", "SH1",
                          "SH-1 ISA Support">;
def FeatureSH2      : SubtargetFeature<"sh2", "SHArchVersion", "SH2",
                          "SH-2 ISA Support",
                          [FeatureSH1]>;
def FeatureSH3      : SubtargetFeature<"sh3", "SHArchVersion", "SH3",
                          "SH-3 ISA Support",
                          [FeatureSH1, FeatureSH2]>;
def FeatureSH4      : SubtargetFeature<"sh4", "SHArchVersion", "SH3",
                          "SH-4 ISA Support",
                          [FeatureSH1, FeatureSH2, FeatureSH3, FeatureFPU]>;
def FeatureSH4A     : SubtargetFeature<"sh4a", "SHArchVersion", "SH4A",
                          "SH-4A ISA Support",
                          [FeatureSH1, FeatureSH2, FeatureSH3, FeatureSH4, FeatureFPU]>;

//===----------------------------------------------------------------------===//
// SuperH Processors
//===----------------------------------------------------------------------===//

include "SuperHSchedule.td"

class Proc<string Name, list<SubtargetFeature> Features>
    : ProcessorModel<Name, GenericSHModel, Features>;

def : Proc<"generic", [FeatureSH4]>;
def : Proc<"sh1", [FeatureSH1]>;
def : Proc<"sh2", [FeatureSH2]>;
def : Proc<"sh3", [FeatureSH3]>;
def : Proc<"sh4", [FeatureSH4]>;
def : Proc<"sh4a", [FeatureSH4A]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "SuperHRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

// include "SuperHInstrInfo.td"

def SuperHInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Target Declaration
//===----------------------------------------------------------------------===//

def SuperH : Target {
  let InstructionSet = SuperHInstrInfo;
}