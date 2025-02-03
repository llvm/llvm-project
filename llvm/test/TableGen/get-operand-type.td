// RUN: llvm-tblgen -gen-instr-info -emit-operand-types \
// RUN:      -I %p/../../include %s | FileCheck %s

// Check that getOperandType has the expected info in it

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def arch : Target {
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
  let Namespace = "MyNamespace";
}

def InstB : Instruction {
  let Size = 1;
  let OutOperandList = (outs i32imm:$d);
  let InOperandList = (ins unknown:$x);
  field bits<8> Inst;
  field bits<8> SoftFail = 0;
  let Namespace = "MyNamespace";
}

def InstC : Instruction {
  let Size = 1;
  let OutOperandList = (outs RegClass:$d);
  let InOperandList = (ins RegOp:$x);
  field bits<8> Inst;
  field bits<8> SoftFail = 0;
  let Namespace = "MyNamespace";
}

// CHECK: #ifdef GET_INSTRINFO_OPERAND_TYPE
// CHECK: static const uint{{.*}}_t Offsets[] = {
// CHECK: static const {{.*}} OpcodeOperandTypes[] = {
// CHECK:        /* InstA */
// CHECK-NEXT:   OpA, OpB, i32imm,
// CHECK-NEXT:   /* InstB */
// CHECK-NEXT:   i32imm, -1,
// CHECK-NEXT:   /* InstC */
// CHECK-NEXT:   RegClass, RegOp,
// CHECK: #endif // GET_INSTRINFO_OPERAND_TYPE
