// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

def MyTarget : Target;

def FeatureA : SubtargetFeature<"NameA", "", "", "">;
def FeatureB : SubtargetFeature<"NameB", "", "", "">;
def FeatureC : SubtargetFeature<"NameC", "", "", "">;

// CHECK: warning: Processor CPU0 contains duplicate feature 'NameA'
def P0 : ProcessorModel<"CPU0", NoSchedModel, [FeatureA, FeatureB, FeatureA]>;
// CHECK: warning: Processor CPU1 contains duplicate tune feature 'NameB'
// CHECK: warning: Processor CPU1 has 'NameC' in both feature and tune feature sets
def P1 : ProcessorModel<"CPU1", NoSchedModel,
                        /*Features=*/[FeatureC],
                        /*TuneFeatures=*/[FeatureB, FeatureB, FeatureC]>;
