// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

def MyTarget : Target;

def FeatureA : SubtargetFeature<"NameA", "", "", "">;
def FeatureB : SubtargetFeature<"NameB", "", "", "">;
def FeatureC : SubtargetFeature<"NameC", "", "", "">;

// CHECK: warning: Processor CPU0 has duplicate Features: NameA
def P0 : ProcessorModel<"CPU0", NoSchedModel, [FeatureA, FeatureB, FeatureA]>;
// CHECK: warning: Processor CPU1 has duplicate TuneFeatures: NameB
def P1 : ProcessorModel<"CPU1", NoSchedModel,
                        /*Features=*/[FeatureA, FeatureB, FeatureC],
                        /*TuneFeatures=*/[FeatureB, FeatureC, FeatureB]>;
