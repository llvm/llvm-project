// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def REG : Register<"REG">;
def GPR : RegisterClass<"TestTarget", [i32], 32, (add REG)>;

def frmarg : Operand<i8> {}

def INSTR : Instruction {
  let OutOperandList = (outs GPR:$rd);
  let InOperandList = (ins GPR:$rs1, frmarg:$rm);
  let Pattern = [];
}

def SDTTest : SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>]>;
def test_node : SDNode<"ISD::INSTR", SDTTest>;

def : Pat<(test_node (i32 GPR:$rs1), frmarg:$rm),
          (INSTR $rs1, $rm)>;

// CHECK: MatcherTable[] = {
// CHECK:   OPC_CheckOpcode, TARGET_VAL(ISD::INSTR),
// CHECK:   OPC_RecordChild0,
// CHECK:   OPC_RecordChild1,
// CHECK:   OPC_CheckTypeI32,
// CHECK:   OPC_MorphNodeTo1None, TARGET_VAL(::INSTR),
// CHECK:       MVT::i32, 2/*#Ops*/, 0, 1, 
