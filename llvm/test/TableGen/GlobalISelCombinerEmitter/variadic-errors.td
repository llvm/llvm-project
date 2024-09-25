// RUN: not llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s 2>&1| \
// RUN: FileCheck %s -implicit-check-not=error:

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'G_BUILD_VECTOR': GIVariadic can only be used on the last operand
def VariadicNotLastInList : GICombineRule<
  (defs root:$dst),
  (match (G_BUILD_VECTOR $dst, $a, GIVariadic<>:$b, $c)),
  (apply (G_ANYEXT $dst, $a))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'G_IMPLICIT_DEF': GIVariadic cannot be used on defs
def VariadicAsDef : GICombineRule<
  (defs root:$dst),
  (match (G_IMPLICIT_DEF GIVariadic<1>:$dst)),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: conflicting types for operand 'args': 'GIVariadic<2,4>' vs 'GIVariadic<3,6>'
def ConflictingInference : GICombineRule<
  (defs root:$dst),
  (match (G_BUILD_VECTOR $dst, GIVariadic<2, 4>:$args)),
  (apply (G_MERGE_VALUES $dst, GIVariadic<3, 6>:$args))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: cannot parse operand type: minimum number of arguments must be greater than zero in GIVariadic
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(G_BUILD_VECTOR ?:$dst, anonymous_{{[0-9]+}}:$a)'
def InvalidBounds0 : GICombineRule<
  (defs root:$dst),
  (match (G_BUILD_VECTOR $dst, GIVariadic<0>:$a)),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: cannot parse operand type: maximum number of arguments (1) must be zero, or greater than the minimum number of arguments (1) in GIVariadic
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(G_BUILD_VECTOR ?:$dst, anonymous_{{[0-9]+}}:$a)'
def InvalidBounds1 : GICombineRule<
  (defs root:$dst),
  (match (G_BUILD_VECTOR $dst, GIVariadic<1,1>:$a)),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: each instance of a GIVariadic operand must have a unique name within the match patterns
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: note: 'c' is used multiple times
def VariadicTypeTestEqOp : GICombineRule<
  (defs root:$a),
  (match (G_MERGE_VALUES $b, $c),
         (G_BUILD_VECTOR $a, $b, GIVariadic<2, 4>:$c)),
  (apply (G_MERGE_VALUES $a, $c))>;

// TODO: We could support this if needed

// CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: GISpecialType is not supported in GICombinePatFrag
// CHECK: :[[@LINE+2]]:{{[0-9]+}}: note: operand 1 of '__PFWithVariadic_alt0_pattern_0' has type 'GIVariadic<1,0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Could not parse GICombinePatFrag 'PFWithVariadic'
def PFWithVariadic: GICombinePatFrag<
    (outs $dst), (ins),
    [(pattern (G_ANYEXT $dst, GIVariadic<>:$b))]>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(PFWithVariadic ?:$dst)'
def UseInPF: GICombineRule<
  (defs root:$dst),
  (match (PFWithVariadic $dst)),
  (apply (G_ANYEXT $dst, (i32 0)))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error:  cannot use a GIVariadic operand on non-variadic instruction 'G_MUL'
def NotVariadicInstMatch : GICombineRule<
  (defs root:$a),
  (match (G_MUL $a, $c, GIVariadic<2, 4>:$b)),
  (apply (G_MERGE_VALUES $a, $b, $c))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error:  cannot use a GIVariadic operand on non-variadic instruction 'G_MUL'
def NotVariadicInstApply : GICombineRule<
  (defs root:$a),
  (match (G_BUILD_VECTOR $a, GIVariadic<2, 4>:$b)),
  (apply (G_MUL $a, $b, $c))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse one or more rules
def MyCombiner: GICombiner<"GenMyCombiner", [
  VariadicNotLastInList,
  VariadicAsDef,
  ConflictingInference,
  InvalidBounds0,
  InvalidBounds1,
  VariadicTypeTestEqOp,
  UseInPF,
  NotVariadicInstMatch,
  NotVariadicInstApply
]>;
