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
