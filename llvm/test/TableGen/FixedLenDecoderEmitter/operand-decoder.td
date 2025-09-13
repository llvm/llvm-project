// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def R0 : Register<"r0">;
def RC : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

def MyInstrInfo : InstrInfo;

def MyTarget : Target {
  let InstructionSet = MyInstrInfo;
}

// CHECK-LABEL: case 0:
// CHECK-NEXT:    if (!Check(S, DecodeRCRegisterClass(MI, Decoder)))
// CHECK-NEXT:      return MCDisassembler::Fail;
// CHECK-NEXT:    tmp = fieldFromInstruction(insn, 2, 4);
// CHECK-NEXT:    MI.addOperand(MCOperand::createImm(tmp));
// CHECK-NEXT:    tmp = 0x0;
// CHECK-NEXT:    insertBits(tmp, fieldFromInstruction(insn, 0, 2), 0, 2);
// CHECK-NEXT:    insertBits(tmp, fieldFromInstruction(insn, 6, 2), 2, 2);
// CHECK-NEXT:    MI.addOperand(MCOperand::createImm(tmp));
// CHECK-NEXT:    tmp = 0x0;
// CHECK-NEXT:    MI.addOperand(MCOperand::createImm(tmp));
// CHECK-NEXT:    tmp = fieldFromInstruction(insn, 13, 2) << 1;
// CHECK-NEXT:    MI.addOperand(MCOperand::createImm(tmp));
// CHECK-NEXT:    tmp = 0x0;
// CHECK-NEXT:    insertBits(tmp, fieldFromInstruction(insn, 17, 1), 1, 1);
// CHECK-NEXT:    insertBits(tmp, fieldFromInstruction(insn, 19, 1), 3, 1);
// CHECK-NEXT:    MI.addOperand(MCOperand::createImm(tmp));
// CHECK-NEXT:    tmp = 0x5;
// CHECK-NEXT:    MI.addOperand(MCOperand::createImm(tmp));
// CHECK-NEXT:    tmp = 0x2;
// CHECK-NEXT:    insertBits(tmp, fieldFromInstruction(insn, 26, 2), 2, 2);
// CHECK-NEXT:    MI.addOperand(MCOperand::createImm(tmp));
// CHECK-NEXT:    tmp = 0xa;
// CHECK-NEXT:    insertBits(tmp, fieldFromInstruction(insn, 28, 1), 0, 1);
// CHECK-NEXT:    insertBits(tmp, fieldFromInstruction(insn, 30, 1), 2, 1);
// CHECK-NEXT:    MI.addOperand(MCOperand::createImm(tmp));
// CHECK-NEXT:    return S;

def I : Instruction {
  let OutOperandList = (outs RC:$op0);
  let InOperandList = (ins i32imm:$op1, i32imm:$op2, i32imm:$op3, i32imm:$op4,
                           i32imm:$op5, i32imm:$op6, i32imm:$op7, i32imm:$op8);
  let Size = 4;
  bits<32> Inst;
  bits<0> op0;                  // no init, no variable parts
  bits<4> op1;                  // no init, 1 variable part
  bits<4> op2;                  // no init, 2 variable parts
  bits<4> op3 = 0b0000;         // zero init, no variable parts
  bits<4> op4 = {0, ?, ?, 0};   // zero init, 1 variable part
  bits<4> op5 = {?, 0, ?, 0};   // zero init, 2 variable parts
  bits<4> op6 = 0b0101;         // non-zero init, no variable parts
  bits<4> op7 = {?, ?, 1, 0};   // non-zero init, 1 variable part
  bits<4> op8 = {1, ?, 1, ?};   // non-zero init, 2 variable parts
  let Inst{5...2} = op1;
  let Inst{1...0} = op2{1...0};
  let Inst{7...6} = op2{3...2};
  let Inst{11...8} = op3;
  let Inst{15...12} = op4;
  let Inst{19...16} = op5;
  let Inst{23...20} = op6;
  let Inst{27...24} = op7;
  let Inst{31...28} = op8;
}
