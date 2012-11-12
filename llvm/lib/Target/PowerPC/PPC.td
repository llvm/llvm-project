//===-- PPC.td - Describe the PowerPC Target Machine -------*- tablegen -*-===//
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

def Directive440 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_440", "">;
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
def DirectiveA2  : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_A2", "">;
def DirectiveE500mc : SubtargetFeature<"", "DarwinDirective",
                                       "PPC::DIR_E500mc", "">;
def DirectiveE5500  : SubtargetFeature<"", "DarwinDirective", 
                                       "PPC::DIR_E5500", "">;
def DirectivePwr6: SubtargetFeature<"", "DarwinDirective", "PPC::DIR_PWR6", "">;
def DirectivePwr7: SubtargetFeature<"", "DarwinDirective", "PPC::DIR_PWR7", "">;

def Feature64Bit     : SubtargetFeature<"64bit","Has64BitSupport", "true",
                                        "Enable 64-bit instructions">;
def Feature64BitRegs : SubtargetFeature<"64bitregs","Use64BitRegs", "true",
                              "Enable 64-bit registers usage for ppc32 [beta]">;
def FeatureAltivec   : SubtargetFeature<"altivec","HasAltivec", "true",
                                        "Enable Altivec instructions">;
def FeatureMFOCRF    : SubtargetFeature<"mfocrf","HasMFOCRF", "true",
                                        "Enable the MFOCRF instruction">;
def FeatureFSqrt     : SubtargetFeature<"fsqrt","HasFSQRT", "true",
                                        "Enable the fsqrt instruction">;
def FeatureSTFIWX    : SubtargetFeature<"stfiwx","HasSTFIWX", "true",
                                        "Enable the stfiwx instruction">;
def FeatureISEL      : SubtargetFeature<"isel","HasISEL", "true",
                                        "Enable the isel instruction">;
def FeatureBookE     : SubtargetFeature<"booke", "IsBookE", "true",
                                        "Enable Book E instructions">;

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
def : Processor<"440", PPC440Itineraries, [Directive440, FeatureISEL,
                                           FeatureBookE]>;
def : Processor<"450", PPC440Itineraries, [Directive440, FeatureISEL,
                                           FeatureBookE]>;
def : Processor<"601", G3Itineraries, [Directive601]>;
def : Processor<"602", G3Itineraries, [Directive602]>;
def : Processor<"603", G3Itineraries, [Directive603]>;
def : Processor<"603e", G3Itineraries, [Directive603]>;
def : Processor<"603ev", G3Itineraries, [Directive603]>;
def : Processor<"604", G3Itineraries, [Directive604]>;
def : Processor<"604e", G3Itineraries, [Directive604]>;
def : Processor<"620", G3Itineraries, [Directive620]>;
def : Processor<"750", G4Itineraries, [Directive750]>;
def : Processor<"g3", G3Itineraries, [Directive750]>;
def : Processor<"7400", G4Itineraries, [Directive7400, FeatureAltivec]>;
def : Processor<"g4", G4Itineraries, [Directive7400, FeatureAltivec]>;
def : Processor<"7450", G4PlusItineraries, [Directive7400, FeatureAltivec]>;
def : Processor<"g4+", G4PlusItineraries, [Directive7400, FeatureAltivec]>;
def : Processor<"970", G5Itineraries,
                  [Directive970, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt, FeatureSTFIWX,
                   Feature64Bit /*, Feature64BitRegs */]>;
def : Processor<"g5", G5Itineraries,
                  [Directive970, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt, FeatureSTFIWX,
                   Feature64Bit /*, Feature64BitRegs */]>;
def : ProcessorModel<"e500mc", PPCE500mcModel,
                  [DirectiveE500mc, FeatureMFOCRF,
                   FeatureSTFIWX, FeatureBookE, FeatureISEL]>;
def : ProcessorModel<"e5500", PPCE5500Model,
                  [DirectiveE5500, FeatureMFOCRF, Feature64Bit,
                   FeatureSTFIWX, FeatureBookE, FeatureISEL]>;
def : Processor<"a2", PPCA2Itineraries, [DirectiveA2, FeatureBookE,
                                         FeatureMFOCRF, FeatureFSqrt,
                                         FeatureSTFIWX, FeatureISEL,
                                         Feature64Bit
                                     /*, Feature64BitRegs */]>;
def : Processor<"pwr6", G5Itineraries,
                  [DirectivePwr6, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt, FeatureSTFIWX,
                   Feature64Bit /*, Feature64BitRegs */]>;
def : Processor<"pwr7", G5Itineraries,
                  [DirectivePwr7, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt, FeatureSTFIWX,
                   FeatureISEL, Feature64Bit /*, Feature64BitRegs */]>;
def : Processor<"ppc", G3Itineraries, [Directive32]>;
def : Processor<"ppc64", G5Itineraries,
                  [Directive64, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt, FeatureSTFIWX,
                   Feature64Bit /*, Feature64BitRegs */]>;


//===----------------------------------------------------------------------===//
// Calling Conventions
//===----------------------------------------------------------------------===//

include "PPCCallingConv.td"

def PPCInstrInfo : InstrInfo {
  let isLittleEndianEncoding = 1;
}

def PPCAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

def PPC : Target {
  // Information about the instructions.
  let InstructionSet = PPCInstrInfo;
  
  let AssemblyWriters = [PPCAsmWriter];
}
