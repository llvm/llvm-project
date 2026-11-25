//===-- SparcFeatures.td - SPARC Feature Definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
