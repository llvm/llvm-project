//===-- SystemZ.td - SystemZ processors and features ---------*- tblgen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Processor and feature definitions.
//
//===----------------------------------------------------------------------===//

class SystemZFeature<string extname, string intname, string desc>
  : Predicate<"Subtarget->has"##intname##"()">,
    AssemblerPredicate<"Feature"##intname, extname>,
    SubtargetFeature<extname, "Has"##intname, "true", desc>;

class SystemZMissingFeature<string intname>
  : Predicate<"!Subtarget->has"##intname##"()">;

def FeatureDistinctOps : SystemZFeature<
  "distinct-ops", "DistinctOps",
  "Assume that the distinct-operands facility is installed"
>;

def FeatureLoadStoreOnCond : SystemZFeature<
  "load-store-on-cond", "LoadStoreOnCond",
  "Assume that the load/store-on-condition facility is installed"
>;

def FeatureLoadStoreOnCond2 : SystemZFeature<
  "load-store-on-cond-2", "LoadStoreOnCond2",
  "Assume that the load/store-on-condition facility 2 is installed"
>;

def FeatureHighWord : SystemZFeature<
  "high-word", "HighWord",
  "Assume that the high-word facility is installed"
>;

def FeatureFPExtension : SystemZFeature<
  "fp-extension", "FPExtension",
  "Assume that the floating-point extension facility is installed"
>;

def FeaturePopulationCount : SystemZFeature<
  "population-count", "PopulationCount",
  "Assume that the population-count facility is installed"
>;

def FeatureFastSerialization : SystemZFeature<
  "fast-serialization", "FastSerialization",
  "Assume that the fast-serialization facility is installed"
>;

def FeatureInterlockedAccess1 : SystemZFeature<
  "interlocked-access1", "InterlockedAccess1",
  "Assume that interlocked-access facility 1 is installed"
>;
def FeatureNoInterlockedAccess1 : SystemZMissingFeature<"InterlockedAccess1">;

def FeatureMiscellaneousExtensions : SystemZFeature<
  "miscellaneous-extensions", "MiscellaneousExtensions",
  "Assume that the miscellaneous-extensions facility is installed"
>;

def FeatureTransactionalExecution : SystemZFeature<
  "transactional-execution", "TransactionalExecution",
  "Assume that the transactional-execution facility is installed"
>;

def FeatureProcessorAssist : SystemZFeature<
  "processor-assist", "ProcessorAssist",
  "Assume that the processor-assist facility is installed"
>;

def FeatureVector : SystemZFeature<
  "vector", "Vector",
  "Assume that the vectory facility is installed"
>;
def FeatureNoVector : SystemZMissingFeature<"Vector">;

def : Processor<"generic", NoItineraries, []>;
def : Processor<"z10", NoItineraries, []>;
def : Processor<"z196", NoItineraries,
                [FeatureDistinctOps, FeatureLoadStoreOnCond, FeatureHighWord,
                 FeatureFPExtension, FeaturePopulationCount,
                 FeatureFastSerialization, FeatureInterlockedAccess1]>;
def : Processor<"zEC12", NoItineraries,
                [FeatureDistinctOps, FeatureLoadStoreOnCond, FeatureHighWord,
                 FeatureFPExtension, FeaturePopulationCount,
                 FeatureFastSerialization, FeatureInterlockedAccess1,
                 FeatureMiscellaneousExtensions,
                 FeatureTransactionalExecution, FeatureProcessorAssist]>;
def : Processor<"z13", NoItineraries,
                [FeatureDistinctOps, FeatureLoadStoreOnCond, FeatureHighWord,
                 FeatureFPExtension, FeaturePopulationCount,
                 FeatureFastSerialization, FeatureInterlockedAccess1,
                 FeatureMiscellaneousExtensions,
                 FeatureTransactionalExecution, FeatureProcessorAssist,
                 FeatureVector, FeatureLoadStoreOnCond2]>;
