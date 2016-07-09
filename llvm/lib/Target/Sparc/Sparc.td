//===-- Sparc.td - Describe the Sparc Target Machine -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
// SPARC Subtarget features.
//

def FeatureV9
    : SubtargetFeature<"v9", "IsV9", "true", "Enable SPARC-V9 instructions">;
def FeatureV8Deprecated
    : SubtargetFeature<"deprecated-v8", "V8DeprecatedInsts", "true",
                       "Enable deprecated V8 instructions in V9 mode">;
def FeatureVIS
    : SubtargetFeature<"vis", "IsVIS", "true",
                       "Enable UltraSPARC Visual Instruction Set extensions">;
def FeatureVIS2
    : SubtargetFeature<"vis2", "IsVIS2", "true",
                       "Enable Visual Instruction Set extensions II">;
def FeatureVIS3
    : SubtargetFeature<"vis3", "IsVIS3", "true",
                       "Enable Visual Instruction Set extensions III">;
def FeatureLeon
    : SubtargetFeature<"leon", "IsLeon", "true", "Enable LEON extensions">;

def FeatureHardQuad
    : SubtargetFeature<"hard-quad-float", "HasHardQuad", "true",
                       "Enable quad-word floating point instructions">;

def UsePopc : SubtargetFeature<"popc", "UsePopc", "true",
                               "Use the popc (population count) instruction">;

def FeatureSoftFloat
    : SubtargetFeature<"soft-float", "UseSoftFloat", "true",
                       "Use software emulation for floating point">;

//==== Features added predmoninantly for LEON subtarget support
include "LeonFeatures.td"

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//

include "SparcRegisterInfo.td"
include "SparcCallingConv.td"
include "SparcSchedule.td"
include "SparcInstrInfo.td"

def SparcInstrInfo : InstrInfo;

def SparcAsmParser : AsmParser { bit ShouldEmitMatchRegisterName = 0; }

//===----------------------------------------------------------------------===//
// SPARC processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
    : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;
def : Proc<"v7", []>;
def : Proc<"v8", []>;
def : Proc<"supersparc", []>;
def : Proc<"sparclite", []>;
def : Proc<"f934", []>;
def : Proc<"hypersparc", []>;
def : Proc<"sparclite86x", []>;
def : Proc<"sparclet", []>;
def : Proc<"tsc701", []>;
def : Proc<"myriad2", []>;
def : Proc<"myriad2.1", []>;
def : Proc<"myriad2.2", []>;
def : Proc<"v9", [ FeatureV9 ]>;
def : Proc<"ultrasparc", [ FeatureV9, FeatureV8Deprecated, FeatureVIS ]>;
def : Proc<"ultrasparc3",
           [ FeatureV9, FeatureV8Deprecated, FeatureVIS, FeatureVIS2 ]>;
def : Proc<"niagara",
           [ FeatureV9, FeatureV8Deprecated, FeatureVIS, FeatureVIS2 ]>;
def : Proc<"niagara2", [
  FeatureV9, FeatureV8Deprecated, UsePopc, FeatureVIS, FeatureVIS2
]>;
def : Proc<"niagara3", [
  FeatureV9, FeatureV8Deprecated, UsePopc, FeatureVIS, FeatureVIS2
]>;
def : Proc<"niagara4", [
  FeatureV9, FeatureV8Deprecated, UsePopc, FeatureVIS, FeatureVIS2, FeatureVIS3
]>;

// LEON 2 FT generic
def : Processor<"leon2", LEON2Itineraries, [ FeatureLeon ]>;

// LEON 2 FT (AT697E)
// AT697E: Provides full coverage of AT697E - covers all the erratum fixes for
// LEON2 AT697E
def : Processor<"at697e", LEON2Itineraries, [
  FeatureLeon, ReplaceSDIV, FixCALL, IgnoreZeroFlag, InsertNOPDoublePrecision
]>;

// LEON 2 FT (AT697F)
// AT697F: Provides full coverage of AT697F - covers all the erratum fixes for
// LEON2 AT697F
def : Processor<"at697f", LEON2Itineraries,
                [ FeatureLeon, InsertNOPDoublePrecision ]>;

// LEON 3 FT generic
def : Processor<"leon3", LEON3Itineraries, [ FeatureLeon, UMACSMACSupport ]>;

// LEON 3 FT (UT699). Provides features for the UT699 processor
// - covers all the erratum fixes for LEON3, but does not support the CASA
// instruction.
def : Processor<"ut699", LEON3Itineraries, [
  FeatureLeon, FixFSMULD, ReplaceFMULS, PreventRoundChange,
  FixAllFDIVSQRT, InsertNOPLoad, FlushCacheLineSWAP, InsertNOPsLoadStore
]>;

// LEON3 FT (GR712RC). Provides features for the GR712RC processor.
// - covers all the erratum fixed for LEON3 and support for the CASA
// instruction.
def : Processor<"gr712rc", LEON3Itineraries,
                [ FeatureLeon, LeonCASA ]>;

// LEON 4 FT generic
def : Processor<"leon4", LEON4Itineraries,
                [ FeatureLeon, LeonCASA ]>;

// GR740: Provides full coverage of GR740 - covers all the erratum fixes for
// LEON3 + support to CASA + LEON 4 instruction timings
def : Processor<"gr740", LEON4Itineraries,
                [ FeatureLeon, LeonCASA ]> {}

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def SparcAsmWriter : AsmWriter {
  string AsmWriterClassName = "InstPrinter";
  int PassSubtarget = 1;
  int Variant = 0;
}

def Sparc : Target {
  // Pull in Instruction Info:
  let InstructionSet = SparcInstrInfo;
  let AssemblyParsers = [ SparcAsmParser ];
  let AssemblyWriters = [ SparcAsmWriter ];
}
