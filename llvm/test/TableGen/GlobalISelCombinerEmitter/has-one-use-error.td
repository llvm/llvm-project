// RUN: not llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s 2>&1 | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// CHECK: GIHasOneUse expects exactly one instruction argument
def test_error_noarg: GICombineRule<
  (defs root:$d),
  (match (GIHasOneUse),
         (G_SUB $d, $a, $b)),
  (apply (GIReplaceReg $d, $a))
>;

def MyCombiner: GICombiner<"MyCombinerImpl", [test_error_noarg]>;
