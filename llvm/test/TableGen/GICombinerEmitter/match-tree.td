// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombinerHelper -gicombiner-stop-after-build %s \
// RUN:     -o %t.inc | FileCheck %s

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

def Rule0 : GICombineRule<
  (defs root:$d),
  (match (MUL $t, $s1, $s2),
         (SUB $d, $t, $s3)),
  (apply [{ APPLY }])>;

def Rule1 : GICombineRule<
  (defs root:$d),
  (match (MOV $s1, $s2),
         (MOV $d, $s1)),
  (apply [{ APPLY }])>;

def Rule2 : GICombineRule<
  (defs root:$d),
  (match (MOV $d, $s)),
  (apply [{ APPLY }])>;

def Rule3 : GICombineRule<
  (defs root:$d),
  (match (MUL $t, $s1, $s2),
         (ADD $d, $t, $s3), [{ A }]),
  (apply [{ APPLY }])>;

def Rule4 : GICombineRule<
  (defs root:$d),
  (match (ADD $d, $s1, $s2)),
  (apply [{ APPLY }])>;

def Rule5 : GICombineRule<
  (defs root:$d),
  (match (SUB $d, $s1, $s2)),
  (apply [{ APPLY }])>;

def Rule6 : GICombineRule<
  (defs root:$d),
  (match (SEXT $t, $s1),
         (TRUNC $d, $t)),
  (apply [{ APPLY }])>;

def Rule7 : GICombineRule<
  (defs root:$d),
  (match (ZEXT $t, $s1),
         (TRUNC $d, $t)),
  (apply [{ APPLY }])>;

def MyCombinerHelper: GICombinerHelper<"GenMyCombinerHelper", [
  Rule0,
  Rule1,
  Rule2,
  Rule3,
  Rule4,
  Rule5,
  Rule6,
  Rule7
]>;

// CHECK-LABEL: digraph "matchtree" {
// CHECK-DAG:   Node[[N0:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[0].getOpcode()|4 partitions|Rule0,Rule1,Rule2,Rule3,Rule4,Rule5,Rule6,Rule7}"]
// CHECK-DAG:   Node[[N1:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1] = getVRegDef(MI[0].getOperand(1))|2 partitions|Rule0,Rule5}"]
// CHECK-DAG:   Node[[N2:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1].getOpcode()|2 partitions|Rule0,Rule5}"]
// CHECK-DAG:   Node[[N3:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule0}"]
// CHECK-DAG:   Node[[N4:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule5}"]
// CHECK-DAG:   Node[[N5:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule5}"]
// CHECK-DAG:   Node[[N6:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1] = getVRegDef(MI[0].getOperand(1))|2 partitions|Rule1,Rule2}"]
// CHECK-DAG:   Node[[N7:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1].getOpcode()|2 partitions|Rule1,Rule2}"]
// CHECK-DAG:   Node[[N8:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule1}"]
// CHECK-DAG:   Node[[N9:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule2}"]
// CHECK-DAG:   Node[[N10:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule2}"]
// CHECK-DAG:   Node[[N11:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1] = getVRegDef(MI[0].getOperand(1))|2 partitions|Rule3,Rule4}"]
// CHECK-DAG:   Node[[N12:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1].getOpcode()|2 partitions|Rule3,Rule4}"]
// CHECK-DAG:   Node[[N13:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule3,Rule4}",color=red]
// CHECK-DAG:   Node[[N14:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule4}"]
// CHECK-DAG:   Node[[N15:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule4}"]
// CHECK-DAG:   Node[[N16:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1] = getVRegDef(MI[0].getOperand(1))|1 partitions|Rule6,Rule7}"]
// CHECK-DAG:   Node[[N17:(0x)?[0-9a-fA-F]+]] [shape=record,label="{MI[1].getOpcode()|2 partitions|Rule6,Rule7}"]
// CHECK-DAG:   Node[[N18:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule6}"]
// CHECK-DAG:   Node[[N19:(0x)?[0-9a-fA-F]+]] [shape=record,label="{No partitioner|Rule7}"]

// The most important partitioner is on the first opcode:
// CHECK-DAG:   Node[[N0]] -> Node[[N1]] [label="#0 MyTarget::SUB"]
// CHECK-DAG:   Node[[N0]] -> Node[[N6]] [label="#1 MyTarget::MOV"]
// CHECK-DAG:   Node[[N0]] -> Node[[N11]] [label="#2 MyTarget::ADD"]
// CHECK-DAG:   Node[[N0]] -> Node[[N16]] [label="#3 MyTarget::TRUNC"]

// For, MI[0].getOpcode() == SUB, then has to determine whether it has a reg
// operand and follow that link. If it can't then Rule5 is the only choice as
// that rule is not constrained to a reg.
// CHECK-DAG:   Node[[N1]] -> Node[[N2]] [label="#0 true"]
// CHECK-DAG:   Node[[N1]] -> Node[[N5]] [label="#1 false"]

// For, MI[0].getOpcode() == SUB && MI[0].getOperand(1).isReg(), if MI[1] is a
// MUL then it must be either Rule0 or Rule5. Rule0 is fully tested so Rule5 is
// unreachable. If it's not MUL then it must be Rule5.
// CHECK-DAG:   Node[[N2]] -> Node[[N3]] [label="#0 MyTarget::MUL"]
// CHECK-DAG:   Node[[N2]] -> Node[[N4]] [label="#1 * or nullptr"]

// CHECK-DAG:   Node[[N6]] -> Node[[N7]] [label="#0 true"]
// CHECK-DAG:   Node[[N6]] -> Node[[N10]] [label="#1 false"]

// CHECK-DAG:   Node[[N7]] -> Node[[N8]] [label="#0 MyTarget::MOV"]
// CHECK-DAG:   Node[[N7]] -> Node[[N9]] [label="#1 * or nullptr"]

// CHECK-DAG:   Node[[N11]] -> Node[[N12]] [label="#0 true"]
// CHECK-DAG:   Node[[N11]] -> Node[[N15]] [label="#1 false"]

// CHECK-DAG:   Node[[N12]] -> Node[[N13]] [label="#0 MyTarget::MUL"]
// CHECK-DAG:   Node[[N12]] -> Node[[N14]] [label="#1 * or nullptr"]

// CHECK-DAG:   Node[[N16]] -> Node[[N17]] [label="#0 true"]

// CHECK-DAG:   Node[[N17]] -> Node[[N18]] [label="#0 MyTarget::SEXT"]
// CHECK-DAG:   Node[[N17]] -> Node[[N19]] [label="#1 MyTarget::ZEXT"]
// CHECK-LABEL: {{^}$}}
