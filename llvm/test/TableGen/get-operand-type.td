// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s | FileCheck %s

// Check that getOperandType has the expected info in it.

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def X86 : Target {
  let InstructionSet = archInstrInfo;
}

def Reg : Register<"reg">;
def RegClass : RegisterClass<"foo", [i32], 0, (add Reg)>;

def OpA : Operand<i32>;
def OpB : Operand<i32>;

def RegOp : RegisterOperand<RegClass>;

def InstA : Instruction {
  let Size = 1;
  let OutOperandList = (outs OpA:$a);
  let InOperandList = (ins OpB:$b, i32imm:$c);
  field bits<8> Inst;
  field bits<8> SoftFail = 0;
  let Namespace = "X86";
}

def InstB : Instruction {
  let Size = 1;
  let OutOperandList = (outs i32imm:$d);
  let InOperandList = (ins unknown:$x);
  field bits<8> Inst;
  field bits<8> SoftFail = 0;
  let Namespace = "X86";
}

def InstC : Instruction {
  let Size = 1;
  let OutOperandList = (outs RegClass:$d);
  let InOperandList = (ins RegOp:$x);
  field bits<8> Inst;
  field bits<8> SoftFail = 0;
  let Namespace = "X86";
}

// CHECK: #ifdef GET_INSTRINFO_OPERAND_TYPE
// CHECK: static constexpr uint{{.*}}_t Offsets[] = {
// CHECK: static constexpr {{.*}} OpcodeOperandTypes[] = {
// CHECK:        /* InstA */
// CHECK-NEXT:   OpA, OpB, i32imm,
// CHECK-NEXT:   /* InstB */
// CHECK-NEXT:   i32imm, -1,
// CHECK-NEXT:   /* InstC */
// CHECK-NEXT:   RegClass, RegOp,
// CHECK: #endif // GET_INSTRINFO_OPERAND_TYPE
