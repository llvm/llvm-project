// RUN: not llvm-tblgen -gen-dag-isel -I %p/../../include %s 2>&1 >/dev/null | FileCheck %s

include "llvm/Target/Target.td"

def R0 : Register<"gr0">;
def GPR : RegisterClass<"My", [i32], 32, (add R0)>;

def my_node_1 : SDNode<"MYISD::NODE1", SDTypeProfile<1, 0, []>>;
def my_node_2 : SDNode<"MYISD::NODE2", SDTypeProfile<2, 0, []>>;
def my_node_3 : SDNode<"MYISD::NODE3", SDTypeProfile<2, 1, []>>;

// CHECK: In Instr1: instruction output operand $rd does not appear in pattern outputs
def Instr1 : Instruction {
  let Namespace = "My";
  let OutOperandList = (outs GPR:$rd);
  let InOperandList = (ins GPR:$rs);
  let Pattern = [(set GPR:$rt, (my_node_1))];
}

// CHECK: In Instr2: pattern output operand $rs does not appear in instruction outputs
def Instr2 : Instruction {
  let Namespace = "My";
  let OutOperandList = (outs GPR:$rd);
  let InOperandList = (ins);
  let Pattern = [(set GPR:$rd, GPR:$rs, (my_node_2))];
}

// CHECK: In Instr3: pattern output operand $rs1 does not appear in instruction outputs
def Instr3 : Instruction {
  let Namespace = "My";
  let OutOperandList = (outs GPR:$rd);
  let InOperandList = (ins GPR:$rs1, GPR:$rs2);
  let Pattern = [(set GPR:$rd, GPR:$rs1, (my_node_3 GPR:$rs2))];
}

def MyInstrInfo : InstrInfo;

def My : Target {
  let InstructionSet = MyInstrInfo;
}

// CHECK: llvm-tblgen: 3 errors.
