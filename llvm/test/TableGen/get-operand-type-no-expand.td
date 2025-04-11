// Test -instr-info-expand-mi-operand-info=0 mode which keeps complex operands
// that contain a DAG of basic operands unexpanded (the default is to expand).

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def X86 : Target {
  let InstructionSet = archInstrInfo;
}

def Reg : Register<"reg">;
def RegClass : RegisterClass<"foo", [i32], 0, (add Reg)>;

class ComplexOperand<int size> : Operand<iPTR> {
  let MIOperandInfo = (ops i8imm, i32imm);
  int Size = size;
}

def i8complex : ComplexOperand<8>;
def i512complex: ComplexOperand<512>;

def InstA : Instruction {
  let Size = 1;
  let OutOperandList = (outs i512complex:$a);
  let InOperandList = (ins i8complex:$b, i32imm:$c);
  field bits<8> Inst;
  field bits<8> SoftFail = 0;
  let Namespace = "X86";
}

// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s \
// RUN:   -instr-info-expand-mi-operand-info=1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-EXPAND
// CHECK-EXPAND: #ifdef GET_INSTRINFO_OPERAND_TYPE
// CHECK-EXPAND: OpcodeOperandTypes[] = {
// CHECK-EXPAND:        /* InstA */
// CHECK-EXPAND-NEXT:   i8imm, i32imm, i8imm, i32imm, i32imm,
// CHECK-EXPAND: #endif // GET_INSTRINFO_OPERAND_TYPE

// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s \
// RUN:   -instr-info-expand-mi-operand-info=0 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOEXPAND
// CHECK-NOEXPAND: #ifdef GET_INSTRINFO_OPERAND_TYPE
// CHECK-NOEXPAND: OpcodeOperandTypes[] = {
// CHECK-NOEXPAND:        /* InstA */
// CHECK-NOEXPAND-NEXT:   i512complex, i8complex, i32imm,
// CHECK-NOEXPAND: #endif // GET_INSTRINFO_OPERAND_TYPE
