// RUN: not llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s 2>&1| \
// RUN: FileCheck %s -implicit-check-not=error:

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def dummy;

def MatchFooPerms: GICombinePatFrag<
    (outs),
    (ins gi_mo:$foo, gi_imm:$cst),
    [
      (pattern "return foo(${foo}, ${cst})"),
      (pattern "return bar(${foo}, ${cst})"),
      (pattern "return bux(${foo}, ${cst})"),
    ]>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: cannot emit rule 'too_many_perms'; 27 permutations would be emitted, but the max is 16
let MaxPermutations = 16 in
def too_many_perms : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $cst),
         (MatchFooPerms $cst, (i32 0)):$a,
         (MatchFooPerms $cst, (i32 0)):$b,
         (MatchFooPerms $cst, (i32 0)):$c
  ),
  (apply (COPY $dst, (i32 0)), "APPLY ${src}")>;

def DummyCXXPF: GICombinePatFrag<
    (outs),
    (ins gi_mo:$in),
    [
      (pattern "return foo()"),
    ]>;
// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: operand 'foo' (for parameter 'in' of 'DummyCXXPF') cannot be unbound
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: note: one or more alternatives of 'DummyCXXPF' do not bind 'in' to an instruction operand; either use a bound operand or ensure 'DummyCXXPF' binds 'in' in all alternatives
def undef_livein : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $bar),
         (DummyCXXPF $foo)
  ),
  (apply (COPY $dst, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: GICombinePatFrag must have one root in its 'out' operands
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Could not parse GICombinePatFrag 'OutMustBeRoot'
def OutMustBeRoot: GICombinePatFrag<
    (outs $foo, $bar),
    (ins),
    [
      (pattern (G_ZEXT $foo, $bar), (G_FPEXT $bar, $y)),
    ]>;
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(OutMustBeRoot ?:$bar, ?:$foo)'
def out_must_be_root : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $bar),
         (OutMustBeRoot $bar, $foo)
  ),
  (apply (COPY $dst, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: output parameter 'bar' must be 'root' or 'gi_mo'
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Could not parse GICombinePatFrag 'BadOutType'
def BadOutType: GICombinePatFrag<
    (outs root:$foo, gi_imm:$bar),
    (ins),
    [
      (pattern (G_ZEXT $foo, $bar), (G_FPEXT $bar, $y)),
    ]>;
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(BadOutType ?:$bar, ?:$foo)'
def bad_out_type : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $bar),
         (BadOutType $bar, $foo)
  ),
  (apply (COPY $dst, (i32 0)))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: pattern 'dbg' ('G_FPEXT') is unreachable from the pattern root!
def UnreachablePat: GICombinePatFrag<
    (outs root:$foo, $bar),
    (ins),
    [
      (pattern (G_ZEXT $foo, $x), (G_FPEXT $bar, $y):$dbg),
    ]>;
def unreachable_pat : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $bar),
         (UnreachablePat $bar, $foo)
  ),
  (apply (COPY $dst, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: wip_match_opcode cannot be used in GICombinePatFrag
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Could not parse GICombinePatFrag 'WipMatchOpcodePatFrag'
def WipMatchOpcodePatFrag: GICombinePatFrag<
    (outs),
    (ins),
    [
      (pattern (wip_match_opcode G_ZEXT)),
    ]>;
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(WipMatchOpcodePatFrag)'
def wip_match_opcode_patfrag : GICombineRule<
  (defs root:$dst),
  (match (WipMatchOpcodePatFrag)),
  (apply (COPY $dst, (i32 0)))>;

def DummyPF: GICombinePatFrag<(outs),(ins),[]>;
// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: nested GICombinePatFrag are not supported
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Could not parse GICombinePatFrag 'NestingPatFrag'
def NestingPatFrag: GICombinePatFrag<
    (outs),
    (ins),
    [
      (pattern (DummyPF)),
    ]>;
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(NestingPatFrag)'
def nest_pat_frag : GICombineRule<
  (defs root:$dst),
  (match (NestingPatFrag)),
  (apply (COPY $dst, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: input parameter 'k' cannot be redefined!
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Could not parse GICombinePatFrag 'DupParamIn'
def DupParamIn: GICombinePatFrag<
    (outs),
    (ins gi_mo:$k, gi_mo:$k),
    [
      (pattern (G_ZEXT $k, $x)),
    ]>;
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(DupParamIn ?:$dst, ?:$bar)'
def dup_params_in : GICombineRule<
  (defs root:$dst),
  (match (DupParamIn $dst, $bar)),
  (apply (COPY $dst, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: duplicate parameter 'k'
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Could not parse GICombinePatFrag 'DupParamOut'
def DupParamOut: GICombinePatFrag<
  (outs root:$k, root:$k),
  (ins),
  [
    (pattern (G_ZEXT $k, $x)),
  ]>;
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(DupParamOut ?:$dst, ?:$bar)'
def dup_params_out : GICombineRule<
  (defs root:$dst),
  (match (DupParamOut $dst, $bar)),
  (apply (COPY $dst, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: input parameter 'k' cannot be redefined!
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Could not parse GICombinePatFrag 'DupParamInOut'
def DupParamInOut: GICombinePatFrag<
  (outs root:$k),
  (ins gi_mo:$k),
  [
    (pattern (G_ZEXT $k, $x)),
  ]>;
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(DupParamInOut ?:$dst, ?:$bar)'
def dup_params_inout : GICombineRule<
  (defs root:$dst),
  (match (DupParamInOut $dst, $bar)),
  (apply (COPY $dst, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: output parameter 'k' must be defined by all alternative patterns in 'DefByAllAlts'
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Could not parse GICombinePatFrag 'DefByAllAlts'
def DefByAllAlts: GICombinePatFrag<
  (outs root:$k),
  (ins),
  [
    (pattern (G_ZEXT $k, $x)),
    (pattern (G_FPEXT $z, $k))
  ]>;
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(DefByAllAlts ?:$dst)'
def def_by_all_alts : GICombineRule<
  (defs root:$dst),
  (match (DefByAllAlts $dst)),
  (apply (COPY $dst, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: Operand 'x' is defined multiple times in patterns of alternative #1
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Could not parse GICombinePatFrag 'MultiDefInPat'
def MultiDefInPat: GICombinePatFrag<
  (outs root:$k),
  (ins),
  [
    (pattern (G_ZEXT $k, $a)),
    (pattern (G_ZEXT $k, $x), (G_ZEXT $x, $foo), (G_FPEXT $x, $foo)),
  ]>;
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(MultiDefInPat ?:$dst)'
def multi_def_in_pat : GICombineRule<
  (defs root:$dst),
  (match (MultiDefInPat $dst)),
  (apply (COPY $dst, (i32 0)))>;

def ExpectedImm: GICombinePatFrag<
  (outs root:$k),
  (ins gi_imm:$i),
  [
    (pattern (G_ZEXT $k, $i)),
  ]>;
// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: expected operand 1 of 'ExpectedImm' to be an immediate; got MachineOperand $z
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(ExpectedImm ?:$dst, ?:$z)'
def expected_imm : GICombineRule<
  (defs root:$dst),
  (match (ExpectedImm $dst, $z)),
  (apply (COPY $dst, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: operand 1 of 'ExpectedImm' cannot be a named immediate
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(ExpectedImm ?:$dst, (i32 0):$z)'
def expected_imm_namedimm : GICombineRule<
  (defs root:$dst),
  (match (ExpectedImm $dst, (i32 0):$z)),
  (apply (COPY $dst, (i32 0)))>;

def ExpectedMO: GICombinePatFrag<
  (outs root:$k),
  (ins gi_mo:$i),
  [
    (pattern (G_ZEXT $k, $i)),
  ]>;
// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: expected operand 1 of 'ExpectedMO' to be a MachineOperand; got imm 0
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(ExpectedMO ?:$dst, (i32 0))'
def expected_mo : GICombineRule<
  (defs root:$dst),
  (match (ExpectedMO $dst, (i32 0))),
  (apply (COPY $dst, (i32 0)))>;
// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: expected operand 1 of 'ExpectedMO' to be a MachineOperand; got imm 0:$z
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(ExpectedMO ?:$dst, (i32 0):$z)'
def expected_mo_namedimm : GICombineRule<
  (defs root:$dst),
  (match (ExpectedMO $dst, (i32 0):$z)),
  (apply (COPY $dst, (i32 0)))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'dum': using GICombinePatFrag is not supported in apply patterns
def patfrag_in_apply : GICombineRule<
  (defs root:$dst),
  (match (COPY $dst, (i32 0))),
  (apply (DummyPF):$dum)>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: cannot use 'DummyPF as match root
def patfrag_cannot_be_root : GICombineRule<
  (defs root:$root),
  (match (DummyPF):$root),
  (apply (COPY $dst, (i32 0)):$root)>;

def TypedParams: GICombinePatFrag<
  (outs root:$k),
  (ins gi_mo:$i),
  [
    (pattern (G_ZEXT $k, i32:$i)),
  ]>;
// CHECK: :[[@LINE+3]]:{{[0-9]+}}: warning: impossible type constraints: operand 1 of 'broken' has type 'i64', but 'TypedParams' constrains it to 'i32'
// CHECK: :[[@LINE+2]]:{{[0-9]+}}: note: operand 1 of 'broken' is 'k'
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: note: argument 1 of 'TypedParams' is 'i'
def inconsistent_arg_type : GICombineRule<
  (defs root:$dst),
  (match (TypedParams $dst, i64:$k):$broken),
  (apply (COPY $dst, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: all instructions that define root 'foo' in 'RootDefHasMultiDefs' can only have a single output operand
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Could not parse GICombinePatFrag 'RootDefHasMultiDefs'
def RootDefHasMultiDefs: GICombinePatFrag<
    (outs root:$foo),
    (ins gi_imm:$cst),
    [
      (pattern (G_UNMERGE_VALUES $foo, $z, $y))
    ]>;
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(RootDefHasMultiDefs ?:$root, (i32 10))'
def root_def_has_multi_defs : GICombineRule<
  (defs root:$root),
  (match (RootDefHasMultiDefs $root, (i32 10))),
  (apply (COPY $root, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: matching/writing MIFlags is only allowed on CodeGenInstruction patterns
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(DummyCXXPF ?:$x, (MIFlags FmArcp))'
def miflags_in_pf : GICombineRule<
  (defs root:$x),
  (match (COPY $x, $y), (DummyCXXPF $x, (MIFlags FmArcp))),
  (apply (COPY $x, $y))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: '$pf' does not refer to a CodeGenInstruction in MIFlags of '__badtype_for_flagref_in_apply_apply_0'
def badtype_for_flagref_in_apply : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $src), (DummyCXXPF $src):$pf),
  (apply (G_MUL $dst, $src, $src, (MIFlags $pf)))>;

// CHECK: error: Failed to parse one or more rules

def MyCombiner: GICombiner<"GenMyCombiner", [
  too_many_perms,
  undef_livein,
  out_must_be_root,
  bad_out_type,
  unreachable_pat,
  wip_match_opcode_patfrag,
  nest_pat_frag,
  dup_params_in,
  dup_params_out,
  dup_params_inout,
  def_by_all_alts,
  multi_def_in_pat,
  expected_imm,
  expected_imm_namedimm,
  expected_mo,
  expected_mo_namedimm,
  patfrag_in_apply,
  patfrag_cannot_be_root,
  inconsistent_arg_type,
  root_def_has_multi_defs,
  miflags_in_pf,
  badtype_for_flagref_in_apply
]>;
