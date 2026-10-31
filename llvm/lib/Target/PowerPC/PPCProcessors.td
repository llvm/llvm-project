//===-- PPCProcessors.td - PowerPC Processor Definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PowerPC processors supported.
//

def : Processor<"generic", G3Itineraries, [Directive32, FeatureHardFloat,
                                           FeatureMFTB]>;
def : ProcessorModel<"440", PPC440Model, [Directive440, FeatureISEL,
                                          FeatureFRES, FeatureFRSQRTE,
                                          FeatureICBT, FeatureBookE,
                                          FeatureMSYNC, FeatureMFTB]>;
def : ProcessorModel<"450", PPC440Model, [Directive440, FeatureISEL,
                                          FeatureFRES, FeatureFRSQRTE,
                                          FeatureICBT, FeatureBookE,
                                          FeatureMSYNC, FeatureMFTB]>;
def : Processor<"601", G3Itineraries, [Directive601, FeatureFPU]>;
def : Processor<"602", G3Itineraries, [Directive602, FeatureFPU,
                                       FeatureMFTB]>;
def : Processor<"603", G3Itineraries, [Directive603,
                                       FeatureFRES, FeatureFRSQRTE,
                                       FeatureMFTB]>;
def : Processor<"603e", G3Itineraries, [Directive603,
                                        FeatureFRES, FeatureFRSQRTE,
                                        FeatureMFTB]>;
def : Processor<"603ev", G3Itineraries, [Directive603,
                                         FeatureFRES, FeatureFRSQRTE,
                                         FeatureMFTB]>;
def : Processor<"604", G3Itineraries, [Directive604,
                                       FeatureFRES, FeatureFRSQRTE,
                                       FeatureMFTB]>;
def : Processor<"604e", G3Itineraries, [Directive604,
                                        FeatureFRES, FeatureFRSQRTE,
                                        FeatureMFTB]>;
def : Processor<"620", G3Itineraries, [Directive620,
                                       FeatureFRES, FeatureFRSQRTE,
                                       FeatureMFTB]>;
def : Processor<"750", G4Itineraries, [Directive750,
                                       FeatureFRES, FeatureFRSQRTE,
                                       FeatureMFTB]>;
def : Processor<"g3", G3Itineraries, [Directive750,
                                      FeatureFRES, FeatureFRSQRTE,
                                      FeatureMFTB]>;
def : Processor<"7400", G4Itineraries, [Directive7400, FeatureAltivec,
                                        FeatureFRES, FeatureFRSQRTE,
                                        FeatureMFTB]>;
def : Processor<"g4", G4Itineraries, [Directive7400, FeatureAltivec,
                                      FeatureFRES, FeatureFRSQRTE,
                                      FeatureMFTB]>;
def : Processor<"7450", G4PlusItineraries, [Directive7400, FeatureAltivec,
                                            FeatureFRES, FeatureFRSQRTE,
                                            FeatureMFTB]>;
def : Processor<"g4+", G4PlusItineraries, [Directive7400, FeatureAltivec,
                                           FeatureFRES, FeatureFRSQRTE,
                                           FeatureMFTB]>;

def : ProcessorModel<"970", G5Model,
                  [Directive970, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt,
                   FeatureFRES, FeatureFRSQRTE, FeatureSTFIWX,
                   Feature64BitSupport /*, Feature64BitRegs */,
                   FeatureMFTB]>;
def : ProcessorModel<"g5", G5Model,
                  [Directive970, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt, FeatureSTFIWX,
                   FeatureFRES, FeatureFRSQRTE,
                   Feature64BitSupport /*, Feature64BitRegs */,
                   FeatureMFTB, DeprecatedDST]>;
def : ProcessorModel<"e500", PPCE500Model,
                  [DirectiveE500,
                   FeatureICBT, FeatureBookE,
                   FeatureISEL, FeatureMFTB, FeatureMSYNC, FeatureSPE]>;
def : ProcessorModel<"e500mc", PPCE500mcModel,
                  [DirectiveE500mc,
                   FeatureSTFIWX, FeatureICBT, FeatureBookE,
                   FeatureISEL, FeatureMFTB]>;
def : ProcessorModel<"e5500", PPCE5500Model,
                  [DirectiveE5500, FeatureMFOCRF, Feature64BitSupport,
                   FeatureSTFIWX, FeatureICBT, FeatureBookE,
                   FeatureISEL, FeatureMFTB]>;
def : ProcessorModel<"a2", PPCA2Model,
                  [DirectiveA2, FeatureICBT, FeatureBookE, FeatureMFOCRF,
                   FeatureFCPSGN, FeatureFSqrt, FeatureFRE, FeatureFRES,
                   FeatureFRSQRTE, FeatureFRSQRTES, FeatureRecipPrec,
                   FeatureSTFIWX, FeatureLFIWAX,
                   FeatureFPRND, FeatureFPCVT, FeatureISEL,
                   FeatureSlowPOPCNTD, FeatureCMPB, FeatureLDBRX,
                   Feature64BitSupport /*, Feature64BitRegs */, FeatureMFTB,
                   FeatureISA2_06]>;
def : ProcessorModel<"pwr3", G5Model,
                  [DirectivePwr3, FeatureAltivec,
                   FeatureFRES, FeatureFRSQRTE, FeatureMFOCRF,
                   FeatureSTFIWX, Feature64BitSupport]>;
def : ProcessorModel<"pwr4", G5Model,
                  [DirectivePwr4, FeatureAltivec, FeatureMFOCRF,
                   FeatureFSqrt, FeatureFRES, FeatureFRSQRTE,
                   FeatureSTFIWX, Feature64BitSupport, FeatureMFTB]>;
def : ProcessorModel<"pwr5", G5Model,
                  [DirectivePwr5, FeatureAltivec, FeatureMFOCRF,
                   FeatureFSqrt, FeatureFRE, FeatureFRES,
                   FeatureFRSQRTE, FeatureFRSQRTES,
                   FeatureSTFIWX, Feature64BitSupport,
                   FeatureMFTB, DeprecatedDST]>;
def : ProcessorModel<"pwr5x", G5Model,
                  [DirectivePwr5x, FeatureAltivec, FeatureMFOCRF,
                   FeatureFSqrt, FeatureFRE, FeatureFRES,
                   FeatureFRSQRTE, FeatureFRSQRTES,
                   FeatureSTFIWX, FeatureFPRND, Feature64BitSupport,
                   FeatureMFTB, DeprecatedDST]>;
def : ProcessorModel<"pwr6", G5Model,
                  [DirectivePwr6, FeatureAltivec,
                   FeatureMFOCRF, FeatureFCPSGN, FeatureFSqrt, FeatureFRE,
                   FeatureFRES, FeatureFRSQRTE, FeatureFRSQRTES,
                   FeatureRecipPrec, FeatureSTFIWX, FeatureLFIWAX, FeatureCMPB,
                   FeatureFPRND, Feature64BitSupport /*, Feature64BitRegs */,
                   FeatureMFTB, DeprecatedDST]>;
def : ProcessorModel<"pwr6x", G5Model,
                  [DirectivePwr5x, FeatureAltivec, FeatureMFOCRF,
                   FeatureFCPSGN, FeatureFSqrt, FeatureFRE, FeatureFRES,
                   FeatureFRSQRTE, FeatureFRSQRTES, FeatureRecipPrec,
                   FeatureSTFIWX, FeatureLFIWAX, FeatureCMPB,
                   FeatureFPRND, Feature64BitSupport,
                   FeatureMFTB, DeprecatedDST]>;
def : ProcessorModel<"pwr7", P7Model, ProcessorFeatures.P7Features>;
def : ProcessorModel<"pwr8", P8Model, ProcessorFeatures.P8Features>;
def : ProcessorModel<"pwr9", P9Model, ProcessorFeatures.P9Features>;
def : ProcessorModel<"pwr10", P10Model, ProcessorFeatures.P10Features>;
def : ProcessorModel<"pwr11", P10Model, ProcessorFeatures.P11Features>;
// No scheduler model for future CPU.
def : ProcessorModel<"future", NoSchedModel,
                  ProcessorFeatures.FutureFeatures>;
def : Processor<"ppc", G3Itineraries, [Directive32, FeatureHardFloat,
                                       FeatureMFTB]>;
def : Processor<"ppc32", G3Itineraries, [Directive32, FeatureHardFloat,
                                         FeatureMFTB]>;
def : ProcessorModel<"ppc64", G5Model,
                  [Directive64, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt, FeatureFRES,
                   FeatureFRSQRTE, FeatureSTFIWX,
                   Feature64BitSupport /*, Feature64BitRegs */,
                   FeatureMFTB]>;
def : ProcessorModel<"ppc64le", P8Model, ProcessorFeatures.P8Features>;
