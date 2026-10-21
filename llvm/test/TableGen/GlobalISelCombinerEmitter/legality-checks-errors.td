// RUN: not llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s 2>&1 | \
// RUN: FileCheck %s -implicit-check-not=error:

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: LegalityChecks: type 1 of 'G_TRUNC' must use an operand defined in the 'match'
def NotInMatch : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $x)),
  (apply "APPLY")> {
  let LegalityChecks = [(G_TRUNC $dst, $y)];
}

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: LegalityChecks: type 1 of 'G_ZEXT' must be $x or (GIElementTypeOf $x)
def BadTypeExpr : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $x)),
  (apply "APPLY")> {
  let LegalityChecks = [(G_ZEXT $dst, (G_ADD $x))];
}

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: LegalityChecks: 'GIElementTypeOf' is not an instruction
def NotAnInstruction : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $x)),
  (apply "APPLY")> {
  let LegalityChecks = [(GIElementTypeOf $dst)];
}

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse one or more rules
def MyCombiner: GICombiner<"GenMyCombiner", [
  NotInMatch,
  BadTypeExpr,
  NotAnInstruction
]>;