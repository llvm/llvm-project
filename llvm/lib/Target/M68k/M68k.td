//===-- M68k.td - Motorola 680x0 target definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is a target description file for the Motorola 680x0 family, referred
/// to here as the "M68k" architecture.
///
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// M68k Subtarget features
//===----------------------------------------------------------------------===//

def FeatureISA00
  : SubtargetFeature<"isa-68000", "SubtargetKind", "M00",
                     "Is M68000 ISA supported">;

def FeatureISA10
  : SubtargetFeature<"isa-68010", "SubtargetKind", "M10",
                     "Is M68010 ISA supported",
                     [ FeatureISA00 ]>;

def FeatureISA20
  : SubtargetFeature<"isa-68020", "SubtargetKind", "M20",
                     "Is M68020 ISA supported",
                     [ FeatureISA10 ]>;

def FeatureISA30
  : SubtargetFeature<"isa-68030", "SubtargetKind", "M30",
                     "Is M68030 ISA supported",
                     [ FeatureISA20 ]>;

def FeatureISA40
  : SubtargetFeature<"isa-68040", "SubtargetKind", "M40",
                     "Is M68040 ISA supported",
                     [ FeatureISA30 ]>;

def FeatureISA60
  : SubtargetFeature<"isa-68060", "SubtargetKind", "M60",
                     "Is M68060 ISA supported",
                     [ FeatureISA40 ]>;

//===----------------------------------------------------------------------===//
// M68k processors supported.
//===----------------------------------------------------------------------===//

include "M68kSchedule.td"

class Proc<string Name, list<SubtargetFeature> Features>
    : ProcessorModel<Name, GenericM68kModel, Features>;

def : Proc<"generic", [ FeatureISA00 ]>;
def : Proc<"M68000",  [ FeatureISA00 ]>;
def : Proc<"M68010",  [ FeatureISA10 ]>;
def : Proc<"M68020",  [ FeatureISA20 ]>;
def : Proc<"M68030",  [ FeatureISA30 ]>;
def : Proc<"M68040",  [ FeatureISA40 ]>;
def : Proc<"M68060",  [ FeatureISA60 ]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "M68kRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "M68kInstrInfo.td"

def M68kInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Calling Conventions
//===----------------------------------------------------------------------===//

include "M68kCallingConv.td"

//===----------------------------------------------------------------------===//
// Target
//===----------------------------------------------------------------------===//

def M68k : Target {
  let InstructionSet = M68kInstrInfo;
}
