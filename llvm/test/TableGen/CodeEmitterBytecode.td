// Check that InstrInfo.UseBytecodeEncoder emits a program table and an
// interpreter loop in place of the opcode switch, for both the APInt and the
// uint64_t shapes.

// RUN: llvm-tblgen -gen-emitter -I %p/../../include %s \
// RUN:   | FileCheck %s --check-prefix=SWITCH
// RUN: llvm-tblgen -gen-emitter -I %p/../../include -DBYTECODE %s \
// RUN:   | FileCheck %s --check-prefix=APINT
// RUN: llvm-tblgen -gen-emitter -I %p/../../include -DBYTECODE -DNARROW %s \
// RUN:   | FileCheck %s --check-prefix=U64

include "llvm/Target/Target.td"

#ifdef BYTECODE
def TestInstrInfo : InstrInfo { let UseBytecodeEncoder = 1; }
#else
def TestInstrInfo : InstrInfo;
#endif

def Test : Target {
  let InstructionSet = TestInstrInfo;
}

// The APInt shape is selected by the widest encoding, so the same instructions
// are declared at 128 bits by default and at 32 bits under -DNARROW.
#ifdef NARROW
class TestEncoding { bits<32> Inst; }
#else
class TestEncoding { bits<128> Inst; }
#endif

class TestInst<bits<8> Opc, dag Ins = (ins i32imm:$src, i32imm:$other)>
    : Instruction, TestEncoding {
  bits<8> src;
  bits<8> other;
  let Inst{7...0} = Opc;
  let OutOperandList = (outs);
  let InOperandList = Ins;
}

// A and B encode their operands identically, so they share a sequence. D
// encodes the same bits from the operand at the same position, under another
// name, so it shares that sequence too. C has a statement of its own, and E is
// the only one with a post-encoder.
def A : TestInst<1> { let Inst{15...8} = src; }
def B : TestInst<2> { let Inst{15...8} = src; }
def C : TestInst<3> { let Inst{23...16} = src; }
def D : TestInst<4, (ins i32imm:$renamed)> {
  bits<8> renamed;
  let Inst{15...8} = renamed;
}
def E : TestInst<5> {
  let Inst{15...8} = src;
  let PostEncoderMethod = "postEncode";
}

// The switch form is unchanged.
// SWITCH-NOT: OpcodeToOffset
// SWITCH:     switch (opcode) {
// SWITCH:     // op: src

// OpcodeToOffset is indexed unchecked, so reject out-of-range opcodes first.
// APINT:      unsigned TableIndex = opcode - FirstSupportedOpcode;
// APINT-NEXT: if (TableIndex >= 5)
// APINT-NEXT:   reportUnsupportedInst(MI);

// A, B and D share one sequence, as the operand's name does not reach the
// program, and E reuses its statement.
// APINT:      static const uint8_t OpcodeToOffset[] = {
// APINT-NEXT:   1, // ::A
// APINT-NEXT:   1, // ::B
// APINT-NEXT:   3, // ::C
// APINT-NEXT:   1, // ::D
// APINT-NEXT:   5, // ::E
// APINT-NEXT: };
// APINT-NEXT: static const uint8_t EncodeProgram[] = {
// APINT-NEXT:   /* 0 */ 0,
// APINT-NEXT:   /* 1 */ 2, 1,
// APINT-NEXT:   /* 3 */ 3, 1,
// APINT-NEXT:   /* 5 */ 2, 4, 1,
// APINT-NEXT: };
// APINT-NEXT: for (uint8_t Idx = OpcodeToOffset[TableIndex];; ++Idx) {
// APINT-NEXT:   switch (EncodeProgram[Idx]) {
// APINT-NEXT:   default: llvm_unreachable("Unexpected bytecode command.");
// APINT-NEXT:   case 0:
// APINT-NEXT:     reportUnsupportedInst(MI);
// APINT-NEXT:     break;
// APINT-NEXT:   case 1:
// APINT-NEXT:     return;
// APINT-NEXT:   case 2:
// APINT-NEXT:     op.clearAllBits();
// APINT-NEXT:     getMachineOpValue(MI, MI.getOperand(0), op, Fixups, STI);
// APINT-NEXT:     Value.insertBits(op.extractBitsAsZExtValue(8, 0), 8, 8);
// APINT-NEXT:     break;
// APINT-NEXT:   case 3:
// APINT-NEXT:     op.clearAllBits();
// APINT-NEXT:     getMachineOpValue(MI, MI.getOperand(0), op, Fixups, STI);
// APINT-NEXT:     Value.insertBits(op.extractBitsAsZExtValue(8, 0), 16, 8);
// APINT-NEXT:     break;
// APINT-NEXT:   case 4:
// APINT-NEXT:     Value = postEncode(MI, Value, STI);
// APINT-NEXT:     break;
// APINT-NEXT:   }
// APINT-NEXT: }
// APINT-NEXT: }

// The uint64_t shape returns the value instead.
// U64:      for (uint8_t Idx = OpcodeToOffset[TableIndex];; ++Idx) {
// U64:        case 1:
// U64-NEXT:     return Value;
// U64-NEXT:   case 2:
// U64-NEXT:     op = getMachineOpValue(MI, MI.getOperand(0), Fixups, STI);
// U64-NEXT:     Value |= (op & 0xff) << 8;
// U64-NEXT:     break;
