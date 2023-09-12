// RUN: not llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s 2>&1| \
// RUN: FileCheck %s -implicit-check-not=error:

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand name format 'unknown' in GITypeOf: expected '$' followed by an operand name
def NoDollarSign : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $src)),
  (apply (G_ANYEXT $dst, (GITypeOf<"unknown"> 0)))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'unknown' ('GITypeOf<$unknown>') does not refer to a matched operand!
def UnknownOperand : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $src)),
  (apply (G_ANYEXT $dst, (GITypeOf<"$unknown"> 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: GISpecialType is not supported in 'match' patterns
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: note: operand 1 of '__UseInMatch_match_0' has type 'GITypeOf<$dst>'
def UseInMatch : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, (GITypeOf<"$dst"> 0))),
  (apply (G_ANYEXT $dst, (i32 0)))>;

// CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: GISpecialType is not supported in GICombinePatFrag
// CHECK: :[[@LINE+2]]:{{[0-9]+}}: note: operand 1 of '__PFWithTypeOF_alt0_pattern_0' has type 'GITypeOf<$dst>
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Could not parse GICombinePatFrag 'PFWithTypeOF'
def PFWithTypeOF: GICombinePatFrag<
    (outs $dst), (ins),
    [(pattern (G_ANYEXT $dst, (GITypeOf<"$dst"> 0)))]>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(PFWithTypeOF ?:$dst)'
def UseInPF: GICombineRule<
  (defs root:$dst),
  (match (PFWithTypeOF $dst)),
  (apply (G_ANYEXT $dst, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: GISpecialType is not supported in 'match' patterns
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: note: operand 1 of '__InferredUseInMatch_match_0' has type 'GITypeOf<$dst>'
def InferredUseInMatch : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $src)),
  (apply (G_ANYEXT $dst, GITypeOf<"$dst">:$src))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: conflicting types for operand 'src': first seen with 'i32' in '__InferenceConflict_match_0, now seen with 'GITypeOf<$dst>' in '__InferenceConflict_apply_0'
def InferenceConflict : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, i32:$src)),
  (apply (G_ANYEXT $dst, GITypeOf<"$dst">:$src))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'tmp' ('GITypeOf<$tmp>') does not refer to a matched operand!
def TypeOfApplyTmp : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $src)),
  (apply (G_ANYEXT $dst, i32:$tmp),
         (G_ANYEXT $tmp, (GITypeOf<"$tmp"> 0)))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse one or more rules
def MyCombiner: GICombiner<"GenMyCombiner", [
  NoDollarSign,
  UnknownOperand,
  UseInMatch,
  UseInPF,
  InferredUseInMatch,
  InferenceConflict,
  TypeOfApplyTmp
]>;
