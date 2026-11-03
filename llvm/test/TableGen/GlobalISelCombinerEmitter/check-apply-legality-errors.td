// RUN: not llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s 2>&1 | \
// RUN: FileCheck %s -implicit-check-not=error:

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// G_FREEZE gets type0 from $x, but G_ANYEXT's type1 is only available from
// $tmp, which is created by the apply and does not exist at match time.
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: CheckApplyLegality: type1 of 'G_ANYEXT' must come from an operand defined in the 'match'
def TempTypeRule : GICombineRule<
  (defs root:$root),
  (match (G_ZEXT $root, $x)),
  (apply (G_FREEZE GITypeOf<"$x">:$tmp, $x),
         (G_ANYEXT $root, $tmp))> {
  let CheckApplyLegality = true;
}

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse one or more rules
def MyCombiner: GICombiner<"GenMyCombiner", [TempTypeRule]>;