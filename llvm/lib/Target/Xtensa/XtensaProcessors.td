//===- XtensaProcessors.td - Xtensa Processors -------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Xtensa supported processors.
//===----------------------------------------------------------------------===//
class Proc<string Name, list<SubtargetFeature> Features>
    : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

def : Proc<"esp32", [FeatureDensity, FeatureSingleFloat, FeatureLoop, FeatureMAC16, FeatureWindowed, FeatureBoolean, FeatureSEXT,
                     FeatureNSA, FeatureMul16, FeatureMul32, FeatureMul32High, FeatureDFPAccel, FeatureS32C1I, FeatureTHREADPTR, FeatureDiv32,
                     FeatureDebug, FeatureException, FeatureHighPriInterrupts, FeatureHighPriInterruptsLevel7, FeatureCoprocessor,
                     FeatureInterrupt, FeatureDataCache, FeatureRelocatableVector, FeatureTimers3, FeaturePRID, FeatureRegionProtection, FeatureMiscSR,
                     FeatureMINMAX, FeatureCLAMPS]>;

def : Proc<"esp8266", [FeatureDensity, FeatureNSA, FeatureMul16, FeatureMul32, FeatureExtendedL32R, FeatureDebug, FeatureException,
                       FeatureHighPriInterrupts, FeatureHighPriInterruptsLevel3, FeatureInterrupt, FeatureRelocatableVector, FeatureTimers1,
					   FeatureRegionProtection, FeaturePRID]>;

def : Proc<"esp32s2", [FeatureDensity, FeatureWindowed, FeatureSEXT, FeatureNSA,  FeatureMul16, FeatureMul32, FeatureMul32High, FeatureTHREADPTR,
                       FeatureDiv32, FeatureDataCache, FeatureDebug, FeatureException, FeatureHighPriInterrupts, FeatureHighPriInterruptsLevel7,
                       FeatureCoprocessor, FeatureInterrupt, FeatureRelocatableVector,	FeatureTimers3, FeaturePRID, FeatureRegionProtection,
                       FeatureMiscSR, FeatureMINMAX, FeatureCLAMPS, FeatureESP32S2Ops]>;

def : Proc<"esp32s3", [FeatureDensity, FeatureSingleFloat, FeatureLoop, FeatureMAC16, FeatureWindowed, FeatureBoolean, FeatureSEXT,
                       FeatureNSA, FeatureMul16, FeatureMul32, FeatureMul32High, FeatureS32C1I, FeatureTHREADPTR, FeatureDiv32,
                       FeatureDataCache, FeatureDebug, FeatureException, FeatureHighPriInterrupts, FeatureHighPriInterruptsLevel7, FeatureCoprocessor,
                       FeatureInterrupt, FeatureRelocatableVector, FeatureTimers3, FeaturePRID, FeatureRegionProtection, FeatureMiscSR,
                       FeatureMINMAX, FeatureCLAMPS, FeatureESP32S3Ops]>;

//===----------------------------------------------------------------------===//
// Intel Audio DSP processors
//===----------------------------------------------------------------------===//

// intel_tgl_adsp - Tiger Lake Audio DSP (cAVS 2.5)
def : Proc<"intel_tgl_adsp", [FeatureDensity, FeatureWindowed, FeatureBoolean, FeatureLoop, FeatureSEXT,
                               FeatureNSA, FeatureMul16, FeatureMul32, FeatureMul32High, FeatureDiv32,
                               FeatureS32C1I, FeatureSingleFloat,
                               FeatureTHREADPTR, FeatureDebug, FeatureException, FeatureHighPriInterrupts,
                               FeatureHighPriInterruptsLevel7, FeatureCoprocessor, FeatureInterrupt,
                               FeatureDataCache, FeatureRelocatableVector, FeatureTimers3, FeaturePRID,
                               FeatureRegionProtection, FeatureMINMAX, FeatureCLAMPS]>;

// intel_ace15_adsp - ACE 1.5 Audio DSP (MTL / ARL)
// Note: FeatureSingleFloat intentionally omitted for GNU as compatibility
def : Proc<"intel_ace15_adsp", [FeatureDensity, FeatureWindowed, FeatureBoolean, FeatureLoop, FeatureSEXT,
                                FeatureNSA, FeatureMul16, FeatureMul32, FeatureMul32High, FeatureDiv32, FeatureS32C1I,
                                FeatureTHREADPTR, FeatureDebug, FeatureException, FeatureHighPriInterrupts,
                                FeatureHighPriInterruptsLevel5, FeatureCoprocessor, FeatureInterrupt, FeatureDataCache,
                                FeatureRelocatableVector, FeatureTimers1, FeaturePRID,
                                FeatureRegionProtection, FeatureMINMAX, FeatureCLAMPS]>;

def : Proc<"intel_ace15_mtpm", [FeatureDensity, FeatureWindowed, FeatureBoolean, FeatureLoop, FeatureSEXT,
                                FeatureNSA, FeatureMul16, FeatureMul32, FeatureMul32High, FeatureDiv32, FeatureS32C1I,
                                FeatureTHREADPTR, FeatureDebug, FeatureException, FeatureHighPriInterrupts,
                                FeatureHighPriInterruptsLevel5, FeatureCoprocessor, FeatureInterrupt, FeatureDataCache,
                                FeatureRelocatableVector, FeatureTimers1, FeaturePRID,
                                FeatureRegionProtection, FeatureMINMAX, FeatureCLAMPS, FeatureSingleFloat]>;

// intel_ace30_adsp - ACE 3.0 Audio DSP (PTL)
def : Proc<"intel_ace30_adsp", [FeatureDensity, FeatureWindowed, FeatureBoolean, FeatureLoop, FeatureSEXT,
                                FeatureNSA, FeatureMul16, FeatureMul32, FeatureMul32High, FeatureDiv32, FeatureS32C1I,
                                FeatureSingleFloat, FeatureTHREADPTR, FeatureDebug, FeatureException, FeatureHighPriInterrupts,
                                FeatureHighPriInterruptsLevel5, FeatureHighPriInterruptsLevel6, FeatureHighPriInterruptsLevel7,
                                FeatureCoprocessor, FeatureInterrupt, FeatureDataCache, FeatureRelocatableVector, FeatureTimers3,
                                FeaturePRID, FeatureRegionProtection, FeatureMiscSR, FeatureMINMAX, FeatureCLAMPS]>;

def : Proc<"intel_ace30_ptl", [FeatureDensity, FeatureWindowed, FeatureBoolean, FeatureLoop, FeatureSEXT,
                               FeatureNSA, FeatureMul16, FeatureMul32, FeatureMul32High, FeatureDiv32, FeatureS32C1I,
                               FeatureSingleFloat, FeatureTHREADPTR, FeatureDebug, FeatureException, FeatureHighPriInterrupts,
                               FeatureHighPriInterruptsLevel5, FeatureCoprocessor, FeatureInterrupt, FeatureDataCache,
                               FeatureRelocatableVector, FeatureTimers2, FeaturePRID, FeatureRegionProtection, FeatureMiscSR,
                               FeatureMINMAX, FeatureCLAMPS]>;

// intel_ace40_adsp - ACE 4.0 Audio DSP (NVL)
def : Proc<"intel_ace40_adsp", [FeatureDensity, FeatureWindowed, FeatureBoolean, FeatureLoop, FeatureSEXT,
                                FeatureNSA, FeatureMul16, FeatureMul32, FeatureMul32High, FeatureDiv32, FeatureS32C1I,
                                FeatureSingleFloat, FeatureTHREADPTR, FeatureDebug, FeatureException, FeatureHighPriInterrupts,
                                FeatureHighPriInterruptsLevel5, FeatureCoprocessor, FeatureInterrupt, FeatureDataCache,
                                FeatureRelocatableVector, FeatureTimers2, FeaturePRID, FeatureRegionProtection, FeatureMiscSR,
                                FeatureMINMAX, FeatureCLAMPS]>;

def : Proc<"intel_ace40", [FeatureDensity, FeatureWindowed, FeatureBoolean, FeatureLoop, FeatureSEXT,
                           FeatureNSA, FeatureMul16, FeatureMul32, FeatureMul32High, FeatureDiv32, FeatureS32C1I,
                           FeatureSingleFloat, FeatureTHREADPTR, FeatureDebug, FeatureException, FeatureHighPriInterrupts,
                           FeatureHighPriInterruptsLevel5, FeatureCoprocessor, FeatureInterrupt, FeatureDataCache,
                           FeatureRelocatableVector, FeatureTimers2, FeaturePRID, FeatureRegionProtection, FeatureMiscSR,
                           FeatureMINMAX, FeatureCLAMPS]>;
