// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s --check-prefixes=CHECK,CHECK-SMALL
// RUN: llvm-tblgen -gen-disassembler --large-decoder-table -I %p/../../../include %s | FileCheck %s --check-prefixes=CHECK,CHECK-LARGE

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def Reg : Register<"reg">;

def RegClass : RegisterClass<"foo", [i64], 0, (add Reg)>;

def GR64 : RegisterOperand<RegClass>;

class MyMemOperand<dag sub_ops> : Operand<iPTR> {
  let MIOperandInfo = sub_ops;
  dag Base;
  dag Extension;
}

def MemOp16: MyMemOperand<(ops GR64:$reg, i16imm:$offset)>;

def MemOp32: MyMemOperand<(ops GR64:$reg, i32imm:$offset)>;

class MyVarInst<MyMemOperand memory_op> : Instruction {
  dag Inst;

  let OutOperandList = (outs GR64:$dst);
  let InOperandList  = (ins memory_op:$src);
}

def FOO16 : MyVarInst<MemOp16> {
  let Inst = (ascend
      (descend (operand "$dst", 3), 0b01000, (operand "$src.reg", 3)),
      (slice "$src.offset", 15, 0, (decoder "myCustomDecoder"))
  );
}
def FOO32 : MyVarInst<MemOp32> {
  let Inst = (ascend
      (descend (operand "$dst", 3), 0b01001,
      (operand "$src.reg", 3, (decoder "myCustomDecoder"))),
      (slice "$src.offset", 31, 16),
      (slice "$src.offset", 15, 0)
  );
}

// Instruction length table
// CHECK: InstrLenTable
// CHECK: 27,
// CHECK-NEXT: 43,
// CHECK-NEXT: };

// CHECK-SMALL:      /* 0 */       OPC_ExtractField, 3, 5,                // Field = Inst{7-3}
// CHECK-SMALL-NEXT: /* 3 */       OPC_FilterValueOrSkip, 8, 4, 0,        // if Field != 0x8 skip to 11
// CHECK-SMALL-NEXT: /* 7 */       OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 0, // Opcode: FOO16
// CHECK-SMALL-NEXT: /* 11 */      OPC_FilterValue, 9,                    // if Field != 0x9 pop scope
// CHECK-SMALL-NEXT: /* 13 */      OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 1, // Opcode: FOO32
// CHECK-SMALL-NEXT: };

// CHECK-LARGE:      /* 0 */       OPC_ExtractField, 3, 5,                // Field = Inst{7-3}
// CHECK-LARGE-NEXT: /* 3 */       OPC_FilterValueOrSkip, 8, 4, 0, 0,     // if Field != 0x8 skip to 12
// CHECK-LARGE-NEXT: /* 8 */       OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 0, // Opcode: FOO16
// CHECK-LARGE-NEXT: /* 12 */      OPC_FilterValue, 9,                    // if Field != 0x9 pop scope
// CHECK-LARGE-NEXT: /* 14 */      OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 1, // Opcode: FOO32
// CHECK-LARGE-NEXT: };

// CHECK:      case 0:
// CHECK-NEXT: tmp = fieldFromInstruction(insn, 8, 3);
// CHECK-NEXT: if (!Check(S, DecodeRegClassRegisterClass(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
// CHECK-NEXT: tmp = fieldFromInstruction(insn, 0, 3);
// CHECK-NEXT: if (!Check(S, DecodeRegClassRegisterClass(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
// CHECK-NEXT: tmp = fieldFromInstruction(insn, 11, 16);
// CHECK-NEXT: if (!Check(S, myCustomDecoder(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
// CHECK-NEXT: return S;
// CHECK-NEXT: case 1:
// CHECK-NEXT: tmp = fieldFromInstruction(insn, 8, 3);
// CHECK-NEXT: if (!Check(S, DecodeRegClassRegisterClass(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
// CHECK-NEXT: tmp = fieldFromInstruction(insn, 0, 3);
// CHECK-NEXT: if (!Check(S, myCustomDecoder(MI, tmp, Address, Decoder))) { return MCDisassembler::Fail; }
// CHECK-NEXT: tmp = 0x0;
// CHECK-NEXT: tmp |= fieldFromInstruction(insn, 11, 16) << 16;
// CHECK-NEXT: tmp |= fieldFromInstruction(insn, 27, 16);
// CHECK-NEXT: MI.addOperand(MCOperand::createImm(tmp));
// CHECK-NEXT: return S;

// CHECK-LABEL: case OPC_ExtractField: {
// CHECK: makeUp(insn, Start + Len);

// CHECK-LABEL: case OPC_CheckField: {
// CHECK: makeUp(insn, Start + Len);

// CHECK-LABEL: case OPC_Decode: {
// CHECK: Len = InstrLenTable[Opc];
// CHECK-NEXT: makeUp(insn, Len);
