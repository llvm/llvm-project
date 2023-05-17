// RUN: not llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s 2>&1 | \
// RUN:     FileCheck -implicit-check-not=error: %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def dummy;

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;
class I<dag OOps, dag IOps, list<dag> Pat>
  : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
}
def MOV : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;

def missing_match_node : GICombineRule<
  (defs root:$a),
  (dummy),
  (dummy)>;
// CHECK: :[[@LINE-4]]:{{[0-9]+}}: error: Expected match operator
// CHECK-NEXT: def missing_match_node : GICombineRule<
// CHECK: :[[@LINE-6]]:{{[0-9]+}}: error: Failed to parse rule

def null_matcher : GICombineRule<
  (defs root:$a),
  (match),
  (dummy)>;
// CHECK: :[[@LINE-4]]:{{[0-9]+}}: error: Matcher is empty
// CHECK-NEXT: def null_matcher : GICombineRule<
// CHECK: :[[@LINE-6]]:{{[0-9]+}}: error: Failed to parse rule

def unknown_kind1 : GICombineRule<
  (defs root:$a),
  (match 0),
  (dummy)>;
// CHECK: :[[@LINE-4]]:{{[0-9]+}}: error: Expected a subclass of GIMatchKind or a sub-dag whose operator is either of a GIMatchKindWithArgs or Instruction
// CHECK-NEXT: def unknown_kind1 : GICombineRule<
// CHECK: :[[@LINE-6]]:{{[0-9]+}}: error: Failed to parse rule

def unknown_kind2 : GICombineRule<
  (defs root:$a),
  (match (dummy)),
  (dummy)>;
// CHECK: :[[@LINE-4]]:{{[0-9]+}}: error: Expected a subclass of GIMatchKind or a sub-dag whose operator is either of a GIMatchKindWithArgs or Instruction
// CHECK-NEXT: def unknown_kind2 : GICombineRule<
// CHECK: :[[@LINE-6]]:{{[0-9]+}}: error: Failed to parse rule

def multidef : GICombineRule<
  (defs root:$a),
  (match (MOV $a, $b),
         (MOV $a, $b)),
  (dummy)>;
// CHECK: :[[@LINE-5]]:{{[0-9]+}}: error: Two different MachineInstrs cannot def the same vreg
// CHECK-NEXT: def multidef : GICombineRule<
// CHECK: :[[@LINE-7]]:{{[0-9]+}}: error: Failed to parse rule

def multidef_but_not_an_error: GICombineRule<
  (defs root:$a),
  (match (MOV $a, $b),
         (MOV $a, $b)),
  (dummy)>;
// CHECK-NOT: :[[@LINE-5]]:{{[0-9]+}}: error:

def MyCombiner: GICombinerHelper<"GenMyCombiner", [
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: Failed to parse one or more rules
  missing_match_node,
  null_matcher,
  unknown_kind1,
  unknown_kind2,
  multidef
  // Rules omitted from a matcher can be as broken as you like. They will not be read.
  // multidef_but_not_an_error
]>;
