// RUN: llvm-tblgen -I %p/../../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def HasOneUse : GICombineRule<
  (defs root:$d),
  (match (G_MUL $mul, $x, $c),
         (G_SUB $d, $a, $mul),
         (GIHasOneUse $mul)),
  (apply (GIReplaceReg $d, $x))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  HasOneUse
]>;

// CHECK: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_MUL)
// CHECK: GIM_CheckHasOneUse, /*MI*/1
