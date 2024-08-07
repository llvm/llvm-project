// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s
// Verify that subtarget features with same names result in an error.

include "llvm/Target/Target.td"

def MyTarget : Target;

def FeatureA : SubtargetFeature<"NameA", "", "", "">;

// CHECK: [[FILE]]:[[@LINE+2]]:5: error: Feature `NameA` already defined.
// CHECK: [[FILE]]:[[@LINE-3]]:5: note: Previous definition here.
def FeatureB : SubtargetFeature<"NameA", "", "", "">;
