// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s
// Verify that subtarget features with same names result in an error.

include "llvm/Target/Target.td"

def MyTarget : Target;

// CHECK: error: Duplicate key in SubtargetFeatureKV: NameA
def FeatureA : SubtargetFeature<"NameA", "", "", "">;
def FeatureB : SubtargetFeature<"NameA", "", "", "">;
