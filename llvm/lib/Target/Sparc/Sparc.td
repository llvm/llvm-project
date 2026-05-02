//===-- Sparc.td - Describe the Sparc Target Machine -------*- tablegen -*-===//
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
include "llvm/TableGen/SearchableTable.td"

//===----------------------------------------------------------------------===//
// SPARC Subtarget features.
//

// Some words about SPARC ISA versions and feature levels:
// - V7: The first public version, used in some early Sun 4 models. Lacks integer
//       multiply and divide in hardware, and doesn't have a formal memory model,
//       though in practice the few multiprocessor machines that do exist
//       are sequentially consistent.
// - V8: Adds hardware multiply and divide, and formally defines TSO and PSO
//       memory models. Most 32-bit chips currently used are of this version.
// - V9: Adds 64-bit capability and RMO memory model, aside from new instructions.
//       Note that unlike most other architectures, instructions that operate
//       on 64-bit registers (e.g casx) are still available even in 32-bit ILP32
//       mode - the G/O registers are always 64-bit regardless of runtime bitness.
//       To let compilers take advantage of this feature, Sun defined an extension
//       to the 32-bit ABI called V8+. It's still possible, though, to compile for
//       V9 ISA level while not emitting any of the 64-bit instructions, that is
//       why -mv8plus is defined separately from the usual -mcpu flags.
//       (OTOH, the combination of `-mcpu=v8 -mv8plus` is effectively a no-op)
//
// As for memory models, the ordering from weakest to strongest is as follows:
// (weaker) RMO -> PSO -> TSO -> SC (stronger)
// Those are defined in such a way that code compiled for a weaker model will
// run fine on processors that only implements a stronger model.
// For simplcity, LLVM targets the weakest model available in a particular
// ISA version: RMO for v9, and PSO for v7/v8.
//
// As for triples, Linux/BSD and Solaris follows different conventions:
// Linux/BSD name | Solaris name | Meaning
// ---------------|--------------|-------------
// sparc-*-*      | sparc-*-*    | 32-bit SPARC
// sparc64-*-*    | sparcv9-*-*  | 64-bit SPARC
// Rather confusingly, Solaris calls the 64-bit triple "sparcv9".
// LLVM accepts both sparc64- and sparcv9- forms for 64-bit triples.

def FeatureSoftMulDiv
  : SubtargetFeature<"soft-mul-div", "UseSoftMulDiv", "true",
                     "Use software emulation for integer multiply and divide">;

def FeatureNoFSMULD
  : SubtargetFeature<"no-fsmuld", "HasNoFSMULD", "true",
                     "Disable the fsmuld instruction.">;
def FeatureNoFMULS
  : SubtargetFeature<"no-fmuls", "HasNoFMULS", "true",
                     "Disable the fmuls instruction.">;

def FeatureV9
  : SubtargetFeature<"v9", "IsV9", "true",
                     "Enable SPARC-V9 instructions">;
def Feature64Bit : SubtargetFeature<"64bit", "Is64Bit", "true",
                                    "Enable 64-bit mode", [FeatureV9]>;

def FeatureV8Plus
  : SubtargetFeature<"v8plus", "IsV8Plus", "true",
                     "Enable V8+ mode, allowing use of 64-bit V9 instructions in 32-bit code">;
def FeatureV8Deprecated
  : SubtargetFeature<"deprecated-v8", "UseV8DeprecatedInsts", "true",
                     "Enable deprecated V8 instructions in V9 mode">;
def FeatureVIS
  : SubtargetFeature<"vis", "IsVIS", "true",
                     "Enable UltraSPARC Visual Instruction Set extensions",
                     [FeatureV9]>;
def FeatureVIS2
  : SubtargetFeature<"vis2", "IsVIS2", "true",
                     "Enable Visual Instruction Set extensions II",
                     [FeatureV9]>;
def FeatureVIS3
  : SubtargetFeature<"vis3", "IsVIS3", "true",
                     "Enable Visual Instruction Set extensions III",
                     [FeatureV9]>;
def FeatureUA2005
  : SubtargetFeature<"ua2005", "IsUA2005", "true",
                     "Enable UltraSPARC Architecture 2005 extensions",
                     [FeatureV9, FeatureVIS, FeatureVIS2]>;
def FeatureUA2007
  : SubtargetFeature<"ua2007", "IsUA2007", "true",
                     "Enable UltraSPARC Architecture 2007 extensions",
                     [FeatureV9, FeatureVIS, FeatureVIS2]>;
def FeatureOSA2011
  : SubtargetFeature<"osa2011", "IsOSA2011", "true",
                     "Enable Oracle SPARC Architecture 2011 extensions",
                     [FeatureV9, FeatureVIS, FeatureVIS2, FeatureVIS3]>;
def FeatureCrypto
  : SubtargetFeature<"crypto", "IsCrypto", "true",
                     "Enable cryptographic extensions",
                     [FeatureOSA2011]>;
def FeatureLeon
  : SubtargetFeature<"leon", "IsLeon", "true",
                     "Enable LEON extensions">;
def FeaturePWRPSR
  : SubtargetFeature<"leonpwrpsr", "HasPWRPSR", "true",
                     "Enable the PWRPSR instruction">;

def FeatureHardQuad
  : SubtargetFeature<"hard-quad-float", "HasHardQuad", "true",
                     "Enable quad-word floating point instructions">;

def UsePopc : SubtargetFeature<"popc", "UsePopc", "true",
                               "Use the popc (population count) instruction">;

def FeatureSoftFloat : SubtargetFeature<"soft-float", "UseSoftFloat", "true",
                              "Use software emulation for floating point">;

//===----------------------------------------------------------------------===//
// SPARC Subtarget tuning features.
//

def TuneSlowRDPC : SubtargetFeature<"slow-rdpc", "HasSlowRDPC", "true",
                                    "rd %pc, %XX is slow", [FeatureV9]>;

def TuneNoPredictor : SubtargetFeature<"no-predictor", "HasNoPredictor", "true",
                                    "Processor has no branch predictor, branches stall execution", []>;

//==== Features added predmoninantly for LEON subtarget support
include "LeonFeatures.td"

//==== Register allocation tweaks needed by some low-level software
foreach i = 1 ... 7  in
    def FeatureReserveG#i : SubtargetFeature<"reserve-g"#i, "ReserveRegister["#i#" + SP::G0]", "true",
                                             "Reserve G"#i#", making it unavailable as a GPR">;
foreach i = 0 ... 5 in
    def FeatureReserveO#i : SubtargetFeature<"reserve-o"#i, "ReserveRegister["#i#" + SP::O0]", "true",
                                             "Reserve O"#i#", making it unavailable as a GPR">;
foreach i = 0 ... 7 in
    def FeatureReserveL#i : SubtargetFeature<"reserve-l"#i, "ReserveRegister["#i#" + SP::L0]", "true",
                                             "Reserve L"#i#", making it unavailable as a GPR">;
foreach i = 0 ... 5 in
    def FeatureReserveI#i : SubtargetFeature<"reserve-i"#i, "ReserveRegister["#i#" + SP::I0]", "true",
                                             "Reserve I"#i#", making it unavailable as a GPR">;

//===----------------------------------------------------------------------===//
// Register File, Calling Conv, Instruction Descriptions
//===----------------------------------------------------------------------===//

include "SparcASITags.td"
include "SparcPrefetchTags.td"
include "SparcRegisterInfo.td"
include "SparcCallingConv.td"
include "SparcSchedule.td"
include "SparcInstrInfo.td"

defm : RemapAllTargetPseudoPointerOperands<sparc_ptr_rc>;

def SparcInstrInfo : InstrInfo;

def SparcAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = true;
  let AllowDuplicateRegisterNames = true;
}

def SparcAsmParserVariant : AsmParserVariant {
  let RegisterPrefix = "%";
}

//===----------------------------------------------------------------------===//
// SPARC processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features,
           list<SubtargetFeature> TuneFeatures = []>
 : Processor<Name, NoItineraries, Features, TuneFeatures>;

def : Proc<"generic",         []>;
def : Proc<"v7",              [FeatureSoftMulDiv, FeatureNoFSMULD]>;
def : Proc<"v8",              []>;
def : Proc<"supersparc",      []>;
def : Proc<"sparclite",       []>;
def : Proc<"f934",            []>;
def : Proc<"hypersparc",      []>;
def : Proc<"sparclite86x",    []>;
def : Proc<"sparclet",        []>;
def : Proc<"tsc701",          []>;
def : Proc<"myriad2",         [FeatureLeon, LeonCASA]>;
def : Proc<"myriad2.1",       [FeatureLeon, LeonCASA]>;
def : Proc<"myriad2.2",       [FeatureLeon, LeonCASA]>;
def : Proc<"myriad2.3",       [FeatureLeon, LeonCASA]>;
def : Proc<"ma2100",          [FeatureLeon, LeonCASA]>;
def : Proc<"ma2150",          [FeatureLeon, LeonCASA]>;
def : Proc<"ma2155",          [FeatureLeon, LeonCASA]>;
def : Proc<"ma2450",          [FeatureLeon, LeonCASA]>;
def : Proc<"ma2455",          [FeatureLeon, LeonCASA]>;
def : Proc<"ma2x5x",          [FeatureLeon, LeonCASA]>;
def : Proc<"ma2080",          [FeatureLeon, LeonCASA]>;
def : Proc<"ma2085",          [FeatureLeon, LeonCASA]>;
def : Proc<"ma2480",          [FeatureLeon, LeonCASA]>;
def : Proc<"ma2485",          [FeatureLeon, LeonCASA]>;
def : Proc<"ma2x8x",          [FeatureLeon, LeonCASA]>;
def : Proc<"v9",              [FeatureV9]>;
def : Proc<"ultrasparc",      [FeatureV9, FeatureV8Deprecated, FeatureVIS],
                              [TuneSlowRDPC]>;
def : Proc<"ultrasparc3",     [FeatureV9, FeatureV8Deprecated, FeatureVIS,
                               FeatureVIS2],
                              [TuneSlowRDPC]>;
def : Proc<"niagara",         [FeatureV9, FeatureV8Deprecated, FeatureVIS,
                               FeatureVIS2, FeatureUA2005],
                              [TuneNoPredictor]>;
def : Proc<"niagara2",        [FeatureV9, FeatureV8Deprecated, UsePopc,
                               FeatureVIS, FeatureVIS2, FeatureUA2005],
                              [TuneNoPredictor]>;
def : Proc<"niagara3",        [FeatureV9, FeatureV8Deprecated, UsePopc,
                               FeatureVIS, FeatureVIS2, FeatureVIS3,
                               FeatureUA2005, FeatureUA2007],
                              [TuneNoPredictor]>;
def : Proc<"niagara4",        [FeatureV9, FeatureV8Deprecated, UsePopc,
                               FeatureVIS, FeatureVIS2, FeatureVIS3,
                               FeatureUA2005, FeatureUA2007, FeatureOSA2011,
                               FeatureCrypto]>;

// LEON 2 FT generic
def : Processor<"leon2", LEON2Itineraries,
                [FeatureLeon]>;

// LEON 2 FT (AT697E)
// TO DO: Place-holder: Processor specific features will be added *very* soon here.
def : Processor<"at697e", LEON2Itineraries,
                [FeatureLeon, InsertNOPLoad]>;

// LEON 2 FT (AT697F)
// TO DO: Place-holder: Processor specific features will be added *very* soon here.
def : Processor<"at697f", LEON2Itineraries,
                [FeatureLeon, InsertNOPLoad]>;


// LEON 3 FT generic
def : Processor<"leon3", LEON3Itineraries,
                [FeatureLeon, UMACSMACSupport]>;

// LEON 3 FT (UT699). Provides features for the UT699 processor
// - covers all the erratum fixes for LEON3, but does not support the CASA instruction.
def : Processor<"ut699", LEON3Itineraries,
                [FeatureLeon, InsertNOPLoad, FeatureNoFSMULD, FeatureNoFMULS, FixAllFDIVSQRT]>;

// LEON3 FT (GR712RC). Provides features for the GR712RC processor.
// - covers all the erratum fixed for LEON3 and support for the CASA instruction.
def : Processor<"gr712rc", LEON3Itineraries,
                [FeatureLeon, LeonCASA]>;

// LEON 4 FT generic
def : Processor<"leon4", LEON4Itineraries,
                [FeatureLeon, UMACSMACSupport, LeonCASA]>;

// LEON 4 FT (GR740)
// TO DO: Place-holder: Processor specific features will be added *very* soon here.
def : Processor<"gr740", LEON4Itineraries,
                [FeatureLeon, UMACSMACSupport, LeonCASA, LeonCycleCounter,
                 FeaturePWRPSR]>;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def SparcAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  int PassSubtarget = 1;
  int Variant = 0;
}

def Sparc : Target {
  // Pull in Instruction Info:
  let InstructionSet = SparcInstrInfo;
  let AssemblyParsers  = [SparcAsmParser];
  let AssemblyParserVariants = [SparcAsmParserVariant];
  let AssemblyWriters = [SparcAsmWriter];
  let AllowRegisterRenaming = 1;
}
