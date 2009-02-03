//===- PPC.td - Describe the PowerPC Target Machine --------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is the top level entry point for the PowerPC target.
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing.
//
include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// PowerPC Subtarget features.
//
 
//===----------------------------------------------------------------------===//
// CPU Directives                                                             //
//===----------------------------------------------------------------------===//

def Directive601 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_601", "">;
def Directive602 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_602", "">;
def Directive603 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_603", "">;
def Directive604 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_603", "">;
def Directive620 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_603", "">;
def Directive7400: SubtargetFeature<"", "DarwinDirective", "PPC::DIR_7400", "">;
def Directive750 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_750", "">;
def Directive970 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_970", "">;
def Directive32  : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_32", "">;
def Directive64  : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_64", "">;

def Feature64Bit     : SubtargetFeature<"64bit","Has64BitSupport", "true",
                                        "Enable 64-bit instructions">;
def Feature64BitRegs : SubtargetFeature<"64bitregs","Use64BitRegs", "true",
                              "Enable 64-bit registers usage for ppc32 [beta]">;
def FeatureAltivec   : SubtargetFeature<"altivec","HasAltivec", "true",
                                        "Enable Altivec instructions">;
def FeatureGPUL      : SubtargetFeature<"gpul","IsGigaProcessor", "true",
                                        "Enable GPUL instructions">;
def FeatureFSqrt     : SubtargetFeature<"fsqrt","HasFSQRT", "true",
                                        "Enable the fsqrt instruction">; 
def FeatureSTFIWX    : SubtargetFeature<"stfiwx","HasSTFIWX", "true",
                                        "Enable the stfiwx instruction">; 

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "PPCRegisterInfo.td"
include "PPCSchedule.td"
include "PPCInstrInfo.td"

//===----------------------------------------------------------------------===//
// PowerPC processors supported.
//

def : Processor<"generic", G3Itineraries, [Directive32]>;
def : Processor<"601", G3Itineraries, [Directive601]>;
def : Processor<"602", G3Itineraries, [Directive602]>;
def : Processor<"603", G3Itineraries, [Directive603]>;
def : Processor<"603e", G3Itineraries, [Directive603]>;
def : Processor<"603ev", G3Itineraries, [Directive603]>;
def : Processor<"604", G3Itineraries, [Directive604]>;
def : Processor<"604e", G3Itineraries, [Directive604]>;
def : Processor<"620", G3Itineraries, [Directive620]>;
def : Processor<"g3", G3Itineraries, [Directive7400]>;
def : Processor<"7400", G4Itineraries, [Directive7400, FeatureAltivec]>;
def : Processor<"g4", G4Itineraries, [Directive7400, FeatureAltivec]>;
def : Processor<"7450", G4PlusItineraries, [Directive7400, FeatureAltivec]>;
def : Processor<"g4+", G4PlusItineraries, [Directive750, FeatureAltivec]>;
def : Processor<"750", G4Itineraries, [Directive750, FeatureAltivec]>;
def : Processor<"970", G5Itineraries,
                  [Directive970, FeatureAltivec,
                   FeatureGPUL, FeatureFSqrt, FeatureSTFIWX,
                   Feature64Bit /*, Feature64BitRegs */]>;
def : Processor<"g5", G5Itineraries,
                  [Directive970, FeatureAltivec,
                   FeatureGPUL, FeatureFSqrt, FeatureSTFIWX,
                   Feature64Bit /*, Feature64BitRegs */]>;
def : Processor<"ppc", G3Itineraries, [Directive32]>;
def : Processor<"ppc64", G5Itineraries,
                  [Directive64, FeatureAltivec,
                   FeatureGPUL, FeatureFSqrt, FeatureSTFIWX,
                   Feature64Bit /*, Feature64BitRegs */]>;


//===----------------------------------------------------------------------===//
// Calling Conventions
//===----------------------------------------------------------------------===//

include "PPCCallingConv.td"

def PPCInstrInfo : InstrInfo {
  // Define how we want to layout our TargetSpecific information field... This
  // should be kept up-to-date with the fields in the PPCInstrInfo.h file.
  let TSFlagsFields = ["PPC970_First",
                       "PPC970_Single",
                       "PPC970_Cracked",
                       "PPC970_Unit"];
  let TSFlagsShifts = [0, 1, 2, 3];

  let isLittleEndianEncoding = 1;
}


def PPC : Target {
  // Information about the instructions.
  let InstructionSet = PPCInstrInfo;
}
