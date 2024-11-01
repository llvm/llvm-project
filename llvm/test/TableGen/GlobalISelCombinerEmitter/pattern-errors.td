// RUN: not llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s 2>&1| \
// RUN: FileCheck %s -implicit-check-not=error:

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Cannot find root 'missing' in match patterns!
def root_not_found : GICombineRule<
  (defs root:$missing),
  (match (COPY $a, $b):$d),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: apply pattern 'd' is supposed to be a root but it does not redefine any of the defs of the match root
def misleading_root : GICombineRule<
  (defs root:$d),
  (match (COPY $a, $b):$d),
  (apply (COPY $tmp, $b):$d,
         (COPY $a, $tmp))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: C++ code cannot be the root of a rule!
def cxx_root : GICombineRule<
  (defs root:$a),
  (match "return MATCH":$a),
  (apply [{ APPLY }]:$z)>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Cannot use live-in operand 'b' as match pattern root!
def livein_root : GICombineRule<
  (defs root:$b),
  (match (COPY $a, $b)),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: 'COPY' expected 2 operands, got 1
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(COPY ?:$a)'
def not_enough_operands : GICombineRule<
  (defs root:$d),
  (match (COPY $a):$d),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: 'COPY' expected 2 operands, got 3
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(COPY ?:$a, ?:$b, ?:$c)'
def too_many_operands : GICombineRule<
  (defs root:$d),
  (match (COPY $a, $b, $c):$d),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Operand 'd' is defined multiple times in the 'match' patterns
def multi_defs : GICombineRule<
  (defs root:$d),
  (match (COPY $d, $b), (COPY $d, $x)),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Operand 'd' is defined multiple times in the 'match' patterns
def multi_defs_2 : GICombineRule<
  (defs root:$d),
  (match (G_UNMERGE_VALUES $d, $d, $b)),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: pattern 'foo' ('COPY') is unreachable from the pattern root!
def unreachable_pat : GICombineRule<
  (defs root:$d),
  (match (COPY $a, $b):$d, (COPY $z, $k):$foo),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'applytest': wip_match_opcode is not supported in apply patterns
def wip_match_opcode_in_apply : GICombineRule<
  (defs root:$d),
  (match (COPY $a, $b):$d, (wip_match_opcode G_ZEXT)),
  (apply (wip_match_opcode G_ZEXT):$applytest)>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: wip_match_opcode can not be used with instruction patterns!
def wip_match_opcode_with_inst_pat : GICombineRule<
  (defs root:$d),
  (match (COPY $a, $b):$d, (wip_match_opcode G_ZEXT)),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: wip_opcode_match can only be present once
def multiple_wip_match_opcode : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode COPY):$d, (wip_match_opcode G_ZEXT)),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Operand 'a' is defined multiple times in the 'apply' patterns
def multiple_def_in_apply : GICombineRule<
  (defs root:$d),
  (match (COPY $a, $b):$d),
  (apply (COPY $a, $b), (COPY $a, $b))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'd' match pattern defined more than once!
def redef_match : GICombineRule<
  (defs root:$d),
  (match (COPY $a, $b):$d, (COPY $b, $z):$d),
  (apply (COPY $a, $b))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'o' apply pattern defined more than once!
def redef_apply: GICombineRule<
  (defs root:$d),
  (match (COPY $a, $b):$d),
  (apply (COPY $a, $x):$o, (COPY $x, $b):$o)>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: redefining an instruction other than the root is not supported (operand 'b'
def redef_nonroot : GICombineRule<
  (defs root:$a),
  (match (COPY $a, $b), (COPY $b, $z)),
  (apply (COPY $a, $b), (G_ZEXT $b, (i32 0)))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Cannot define live-in operand 'b' in the 'apply' pattern
def cannot_def_match_livein : GICombineRule<
  (defs root:$d),
  (match (COPY $a, $b):$d),
  (apply (COPY $a, $b), (COPY $b, $b))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: invalid output operand 'x': operand is not a live-in of the match pattern, and it has no definition
def undef_in_apply : GICombineRule<
  (defs root:$d),
  (match (COPY $a, $b):$d),
  (apply (COPY $a, $x))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'a' must be redefined in the 'apply' pattern
def no_redef_in_apply : GICombineRule<
  (defs root:$a),
  (match (COPY $a, $b):$foo),
  (apply (COPY $x, $b))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: 'b' must be redefined in the 'apply' pattern
def no_redef_in_apply_multidefroot : GICombineRule<
  (defs root:$a),
  (match (G_UNMERGE_VALUES $a, $b, $c)),
  (apply (COPY $a, $c))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: cannot use wip_match_opcode in combination with apply instruction patterns
def instpat_with_wipmatch : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode COPY):$d),
  (apply (COPY $x, $b):$d)>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: cannot parse operand '(i32)'
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(COPY ?:$x, (i32))
def bad_imm_noarg : GICombineRule<
  (defs root:$a),
  (match (COPY $x, (i32)):$d),
  (apply (COPY $x, $b):$d)>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: cannot parse operand '(i32 0, 0)'
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(COPY ?:$x, (i32 0, 0))
def bad_imm_too_many_args : GICombineRule<
  (defs root:$a),
  (match (COPY $x, (i32 0, 0)):$d),
  (apply (COPY $x, $b):$d)>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: cannot parse immediate '(COPY 0)', 'COPY' is not a ValueType
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(COPY ?:$x, (COPY 0))
def bad_imm_not_a_valuetype : GICombineRule<
  (defs root:$a),
  (match (COPY $x, (COPY 0)):$d),
  (apply (COPY $x, $b):$d)>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: invalid output operand 'imm': output immediates cannot be named
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: note: while emitting pattern 'd' (COPY)
def output_imm_cannot_be_named : GICombineRule<
  (defs root:$x),
  (match (COPY $x, (i32 0)):$d),
  (apply (COPY $x, (i32 0):$imm):$d)>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: 'G_CONSTANT' immediate must be typed!
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: note: while emitting pattern 'd' (G_CONSTANT)
def output_imm_must_be_typed : GICombineRule<
  (defs root:$x),
  (match (COPY $x, (i32 0)):$d),
  (apply (G_CONSTANT $x, 0):$d)>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: 'G_BUILD_VECTOR' expected at least 2 operands, got 1
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(G_BUILD_VECTOR ?:$x)'
def too_few_ops_for_variadic : GICombineRule<
  (defs root:$x),
  (match (G_BUILD_VECTOR $x)),
  (apply (COPY $x, 0))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: expected an operand name after 'i32'
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Failed to parse pattern: '(G_FNEG ?:$x, i32)'
def expected_op_name : GICombineRule<
  (defs root:$x),
  (match (G_FNEG $x, i32)),
  (apply (COPY $x, (i32 0)))>;

// CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: invalid operand type: 'not_a_type' is not a ValueType
// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: Failed to parse pattern: '(G_FNEG ?:$x, not_a_type:$y)'
def not_a_type;
def bad_mo_type_not_a_valuetype : GICombineRule<
  (defs root:$x),
  (match (G_FNEG $x, not_a_type:$y)),
  (apply (COPY $x, (i32 0)))>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: def of a new register 'newreg' in the apply patterns must have a type
def untyped_new_reg_in_apply : GICombineRule<
  (defs root:$x),
  (match (G_FNEG $x, $y)),
  (apply (COPY $newreg, (i32 0)),
         (COPY $x, $newreg))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: 'y' is a named immediate, it cannot be defined by another instruction
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: note: 'y' is defined by 'foo'
def def_named_imm_match : GICombineRule<
  (defs root:$x),
  (match  (G_SEXT $y, $z):$foo,
          (G_FNEG $x, (i32 0):$y)),
  (apply (COPY $x, (i32 0)))>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: invalid output operand 'tmp': output immediates cannot be named
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: note: while emitting pattern 'foo' (COPY)
def def_named_imm_apply : GICombineRule<
  (defs root:$x),
  (match (G_FNEG $x, $y)),
  (apply (COPY i32:$tmp, $y),
         (COPY $x, (i32 0):$tmp):$foo)>;

// CHECK: error: Failed to parse one or more rules

def MyCombiner: GICombiner<"GenMyCombiner", [
  root_not_found,
  misleading_root,
  cxx_root,
  livein_root,
  not_enough_operands,
  too_many_operands,
  multi_defs,
  multi_defs_2,
  unreachable_pat,
  wip_match_opcode_in_apply,
  wip_match_opcode_with_inst_pat,
  multiple_wip_match_opcode,
  multiple_def_in_apply,
  redef_match,
  redef_apply,
  redef_nonroot,
  cannot_def_match_livein,
  undef_in_apply,
  no_redef_in_apply,
  no_redef_in_apply_multidefroot,
  instpat_with_wipmatch,
  bad_imm_noarg,
  bad_imm_too_many_args,
  bad_imm_not_a_valuetype,
  output_imm_cannot_be_named,
  output_imm_must_be_typed,
  too_few_ops_for_variadic,
  expected_op_name,
  bad_mo_type_not_a_valuetype,
  untyped_new_reg_in_apply,
  def_named_imm_match,
  def_named_imm_apply
]>;
