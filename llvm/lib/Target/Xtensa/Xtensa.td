//===- Xtensa.td - Describe the Xtensa Target Machine -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Subtarget Features. 
//===----------------------------------------------------------------------===//
def FeatureDensity : SubtargetFeature<"density", "HasDensity", "true",
                                      "Enable Density instructions">;
def HasDensity : Predicate<"Subtarget->hasDensity()">,
                           AssemblerPredicate<"FeatureDensity">;

def FeatureSingleFloat : SubtargetFeature<"fp", "HasSingleFloat", "true",
                                          "Enable Xtensa Single FP instructions">;
def HasSingleFloat : Predicate<"Subtarget->hasSingleFloat()">,
                               AssemblerPredicate<"FeatureSingleFloat">;

def FeatureLoop : SubtargetFeature<"loop", "HasLoop", "true",
                                   "Enable Xtensa Loop extension">;
def HasLoop : Predicate<"Subtarget->hasLoop()">,
                        AssemblerPredicate<"FeatureLoop">;

def FeatureMAC16 : SubtargetFeature<"mac16", "HasMAC16", "true",
                    "Enable Xtensa MAC16 instructions">;
def HasMAC16 : Predicate<"Subtarget->hasMAC16()">,
                           AssemblerPredicate<"FeatureMAC16">;

def FeatureWindowed : SubtargetFeature<"windowed", "HasWindowed", "true",
                    "Enable Xtensa Windowed Register option">;
def HasWindowed : Predicate<"Subtarget->hasWindowed()">,
                           AssemblerPredicate<"FeatureWindowed">;

def FeatureBoolean : SubtargetFeature<"bool", "HasBoolean", "true",
                    "Enable Xtensa Boolean extension">;
def HasBoolean : Predicate<"Subtarget->hasBoolean()">,
                           AssemblerPredicate<"FeatureBoolean">;

def FeatureSEXT : SubtargetFeature<"sext", "HasSEXT", "true",
                    "Enable Xtensa Sign Extend option">;
def HasSEXT : Predicate<"Subtarget->hasSEXT()">,
                           AssemblerPredicate<"FeatureSEXT">;

def FeatureNSA : SubtargetFeature<"nsa", "HasNSA", "true",
                    "Enable Xtensa NSA option">;
def HasNSA : Predicate<"Subtarget->hasNSA()">,
                           AssemblerPredicate<"FeatureNSA">;

def FeatureMul32 : SubtargetFeature<"mul32", "HasMul32", "true",
                    "Enable Xtensa Mul32 option">;
def HasMul32 : Predicate<"Subtarget->hasMul32()">,
                           AssemblerPredicate<"FeatureMul32">;

def FeatureMul32High : SubtargetFeature<"mul32high", "HasMul32High", "true",
                    "Enable Xtensa Mul32High option">;
def HasMul32High : Predicate<"Subtarget->hasMul32High()">,
                           AssemblerPredicate<"FeatureMul32High">;

def FeatureDiv32 : SubtargetFeature<"div32", "HasDiv32", "true",
                    "Enable Xtensa Div32 option">;
def HasDiv32 : Predicate<"Subtarget->hasDiv32()">,
                           AssemblerPredicate<"FeatureDiv32">;

def FeatureS32C1I : SubtargetFeature<"s32c1i", "HasS32C1I", "true",
                    "Enable Xtensa S32C1I option">;
def HasS32C1I : Predicate<"Subtarget->hasS32C1I()">,
                           AssemblerPredicate<"FeatureS32C1I">;

def FeatureTHREADPTR : SubtargetFeature<"threadptr", "HasTHREADPTR", "true",
                    "Enable Xtensa THREADPTR option">;
def HasTHREADPTR : Predicate<"Subtarget->hasTHREADPTR()">,
                           AssemblerPredicate<"FeatureTHREADPTR">;

//===----------------------------------------------------------------------===//
// Xtensa supported processors.
//===----------------------------------------------------------------------===//
class Proc<string Name, list<SubtargetFeature> Features>
    : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "XtensaRegisterInfo.td"
 
//===----------------------------------------------------------------------===//
// Calling Convention Description
//===----------------------------------------------------------------------===//

include "XtensaCallingConv.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "XtensaInstrInfo.td"

def XtensaInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Target Declaration
//===----------------------------------------------------------------------===//

def XtensaAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = 1;
}

def XtensaInstPrinter : AsmWriter 
{
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
} 

def Xtensa : Target 
{
  let InstructionSet = XtensaInstrInfo;
  let AssemblyWriters = [XtensaInstPrinter];
  let AssemblyParsers = [XtensaAsmParser];
}

