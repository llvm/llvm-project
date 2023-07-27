// RUN: not llvm-tblgen -I %p/../../../include -gen-global-isel-combiner-matchtable \
// RUN:     -combiners=MyCombiner %s 2>&1| \
// RUN: FileCheck %s -implicit-check-not=error:

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
def ADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def SUB : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def MUL : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def TRUNC : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def SEXT : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def ZEXT : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def ICMP : I<(outs GPR32:$dst), (ins GPR32:$tst, GPR32:$src1, GPR32:$src2), []>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Cannot find root 'missing' in match patterns!
def root_not_found : GICombineRule<
  (defs root:$missing),
  (match (MOV $a, $b):$d),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Cannot use live-in operand 'b' as match pattern root!
def livein_root : GICombineRule<
  (defs root:$b),
  (match (MOV $a, $b)),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: 'MOV' expected 2 operands, got 1
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Expected a subclass of GIMatchKind or a sub-dag whose operator is either of a GIMatchKindWithArgs or Instruction
def not_enough_operands : GICombineRule<
  (defs root:$d),
  (match (MOV $a):$d),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: 'MOV' expected 2 operands, got 3
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Expected a subclass of GIMatchKind or a sub-dag whose operator is either of a GIMatchKindWithArgs or Instruction
def too_many_operands : GICombineRule<
  (defs root:$d),
  (match (MOV $a, $b, $c):$d),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Operand 'd' is defined multiple times in the 'match' patterns
def multi_defs : GICombineRule<
  (defs root:$d),
  (match (MOV $d, $b), (MOV $d, $x)),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Instruction pattern '__anon_pat_match_5_0' is unreachable from the pattern root!
def unreachable_pat : GICombineRule<
  (defs root:$d),
  (match (MOV $a, $b):$d, (MOV $z, $k)),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error:  wip_match_opcode can not be used with instruction patterns!
def wip_match_opcode_with_inst_pat : GICombineRule<
  (defs root:$d),
  (match (MOV $a, $b):$d, (wip_match_opcode SEXT)),
  (apply [{ APPLY }])>;

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: error:  wip_opcode_match can only be present once
def multiple_wip_match_opcode : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode MOV):$d, (wip_match_opcode SEXT)),
  (apply [{ APPLY }])>;

// CHECK: error: Failed to parse one or more rules

def MyCombiner: GICombinerHelper<"GenMyCombiner", [
  root_not_found,
  livein_root,
  not_enough_operands,
  too_many_operands,
  multi_defs,
  unreachable_pat,
  wip_match_opcode_with_inst_pat,
  multiple_wip_match_opcode
]>;
