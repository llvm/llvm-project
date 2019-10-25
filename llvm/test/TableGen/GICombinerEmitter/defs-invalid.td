// RUN: not llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s 2>&1 | \
// RUN:     FileCheck -implicit-check-not=error %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def dummy;

def missing_defs_node : GICombineRule<
  (dummy),
  (dummy),
  (dummy)>;
// CHECK: :[[@LINE-4]]:{{[0-9]+}}: error: Expected defs operator
// CHECK-NEXT: def missing_defs_node : GICombineRule<
// CHECK: :[[@LINE-6]]:{{[0-9]+}}: error: Failed to parse rule

def no_roots : GICombineRule<
  (defs),
  (dummy),
  (dummy)>;
// CHECK: :[[@LINE-4]]:{{[0-9]+}}: error: Combine rules must have at least one root
// CHECK-NEXT: def no_roots : GICombineRule<
// CHECK: :[[@LINE-6]]:{{[0-9]+}}: error: Failed to parse rule

def unknown_kind : GICombineRule<
  (defs dummy:$a),
  (dummy),
  (dummy)>;
// CHECK: :[[@LINE-4]]:{{[0-9]+}}: error: Expected a subclass of GIDefKind or a sub-dag whose operator is of type GIDefKindWithArgs
// CHECK-NEXT: def unknown_kind : GICombineRule<
// CHECK: :[[@LINE-6]]:{{[0-9]+}}: error: Failed to parse rule

def MyCombiner: GICombinerHelper<"GenMyCombiner", [
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: Failed to parse one or more rules
  missing_defs_node,
  no_roots,
  unknown_kind
]>;
