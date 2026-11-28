//===-- SparcProcessors.td - SPARC Processor Definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
                [FeatureLeon, UMACSMACSupport, LeonCASA]>;

// LEON 3 FT (UT699). Does not support the CASA instruction.
def : Processor<"ut699", LEON3Itineraries,
                [FeatureLeon, InsertNOPLoad, FeatureNoFSMULD, FeatureNoFMULS]>;

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

// LEON 5 generic
def : ProcessorModel<"leon5", LEON5Model,
                     [FeatureLeon, LeonCASA, LeonCycleCounter, FeaturePWRPSR]>;
