// RUN: not llvm-tblgen -I %p/../../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s 2>&1| \
// RUN: FileCheck %s -implicit-check-not=error:

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: expected operand 1 of 'GIReplaceReg' to be a name
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(GIReplaceReg ?:$dst, (i32 0))'
def builtinpat_immop : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $src)),
  (apply (GIReplaceReg $dst, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: expected operand 1 of 'GIReplaceReg' to be a name
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(GIReplaceReg ?:$dst, (i32 0):$k)'
def builtinpat_namedimmop : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $src)),
  (apply (GIReplaceReg $dst, (i32 0):$k))>;


// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'GIEraseRoot' cannot be used in a 'match' pattern
def eraseroot_in_match : GICombineRule<
  (defs root:$dst),
  (match (GIEraseRoot):$mi),
  (apply (COPY $dst, $src))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: 'GIEraseRoot' expected 0 operands, got 1
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(GIEraseRoot ?:$dst)'
def eraseroot_ops : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $src)),
  (apply (GIEraseRoot $dst), (COPY $dst, $src))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: GIEraseRoot must be the only 'apply' pattern
def eraseroot_multiapply : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $src)),
  (apply (GIEraseRoot), (COPY $dst, $src))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: GIEraseRoot can only be used if on roots that do not have any output operand
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: note: 'COPY' has 1 output operands
def eraseroot_root_has_def: GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $src)),
  (apply (GIEraseRoot))>;

def TestPF: GICombinePatFrag<
    (outs root:$def),
    (ins),
    [(pattern (COPY $def, $src))]>;
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: GIEraseRoot can only be used if the root is a CodeGenInstruction or Intrinsic
def eraseroot_notinstmatch: GICombineRule<
  (defs root:$mi),
  (match (TestPF $dst):$mi),
  (apply (GIEraseRoot))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'GIReplaceReg' cannot be used in a 'match' pattern
def replacereg_in_match : GICombineRule<
  (defs root:$dst),
  (match (GIReplaceReg $dst, $src)),
  (apply (COPY $dst, $src))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: 'GIReplaceReg' expected 2 operands, got 1
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(GIReplaceReg ?:$dst)'
def replacereg_ops : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $src)),
  (apply (GIReplaceReg $dst))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: GIReplaceReg cannot replace 'tmp': this builtin can only replace a register defined by the match root
def replacereg_nonroot : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $tmp), (COPY $tmp, $src)),
  (apply (GIReplaceReg $dst, $src), (GIReplaceReg $tmp, $src))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'GIReplaceRegWithConstant' cannot be used in a 'match' pattern
def replaceregconstant_in_match : GICombineRule<
  (defs root:$dst),
  (match (GIReplaceRegWithConstant $dst, 0)),
  (apply (COPY $dst, $src))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: expected operand 1 of 'GIReplaceRegWithConstant' to be an immediate value
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(GIReplaceRegWithConstant ?:$dst, ?:$src)'
def replaceregconstant_nonimm : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $src)),
  (apply (GIReplaceRegWithConstant $dst, $src))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: expected operand 1 of 'GIReplaceRegWithConstant' to be an immediate value
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(GIReplaceRegWithConstant ?:$dst, (i32 0):$k)'
def replaceregconstant_namedimm : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $src)),
  (apply (GIReplaceRegWithConstant $dst, (i32 0):$k))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: GIReplaceRegWithConstant cannot replace 'tmp': this builtin can only replace a register defined by the match root
def replaceregconstant_nonroot : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $tmp), (COPY $tmp, $src)),
  (apply (GIReplaceReg $dst, $src), (GIReplaceRegWithConstant $tmp, 0))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'GIReplaceRegWithFConstant' cannot be used in a 'match' pattern
def replaceregfconstant_in_match : GICombineRule<
  (defs root:$dst),
  (match (GIReplaceRegWithFConstant $dst, 0)),
  (apply (COPY $dst, $src))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: expected operand 1 of 'GIReplaceRegWithFConstant' to be an immediate value
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(GIReplaceRegWithFConstant ?:$dst, ?:$src)'
def replaceregfconstant_nonimm : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $src)),
  (apply (GIReplaceRegWithFConstant $dst, $src))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: expected operand 1 of 'GIReplaceRegWithFConstant' to be an immediate value
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(GIReplaceRegWithFConstant ?:$dst, (i32 0):$k)'
def replaceregfconstant_namedimm : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $src)),
  (apply (GIReplaceRegWithFConstant $dst, (i32 0):$k))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: GIReplaceRegWithFConstant cannot replace 'tmp': this builtin can only replace a register defined by the match root
def replaceregfconstant_nonroot : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, $tmp), (COPY $tmp, $src)),
  (apply (GIReplaceReg $dst, $src), (GIReplaceRegWithFConstant $tmp, 0))>;

// CHECK: error: Failed to parse one or more rules

def MyCombiner: GICombiner<"GenMyCombiner", [
  builtinpat_immop,
  builtinpat_namedimmop,
  eraseroot_in_match,
  eraseroot_ops,
  eraseroot_multiapply,
  eraseroot_root_has_def,
  eraseroot_notinstmatch,
  replacereg_in_match,
  replacereg_ops,
  replacereg_nonroot,
  replaceregconstant_in_match,
  replaceregconstant_nonimm,
  replaceregconstant_namedimm,
  replaceregconstant_nonroot,
  replaceregfconstant_in_match,
  replaceregfconstant_nonimm,
  replaceregfconstant_namedimm,
  replaceregfconstant_nonroot
]>;
