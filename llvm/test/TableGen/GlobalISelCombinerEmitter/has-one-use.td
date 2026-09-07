// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// Test wrapper syntax: (GIHasOneUse (G_MUL ...))
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_MUL)
// CHECK-NEXT: GIM_CheckHasOneUse, /*MI*/1
// CHECK: Combiner Rule #0: test_wrapper
def test_wrapper: GICombineRule<
  (defs root:$d),
  (match (GIHasOneUse (G_MUL $mul, $x, $c)),
         (G_SUB $d, $a, $mul)),
  (apply (GIReplaceReg $d, $x))
>;

// Test standalone syntax: (G_MUL ...), (GIHasOneUse $mul)
// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_MUL)
// CHECK-NEXT: GIM_CheckHasOneUse, /*MI*/1
// CHECK: Combiner Rule #1: test_standalone
def test_standalone: GICombineRule<
  (defs root:$d),
  (match (G_MUL $mul, $x, $c),
         (G_SUB $d, $a, $mul),
         (GIHasOneUse $mul)),
  (apply (GIReplaceReg $d, $x))
>;

def MyCombiner: GICombiner<"MyCombinerImpl", [test_wrapper, test_standalone]>;
