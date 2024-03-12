// RUN: llvm-tblgen -gen-emitter -I %p/../../include %s | \
// RUN:     FileCheck %s --check-prefix=ENCODER
// RUN: llvm-tblgen -gen-disassembler -I %p/../../include %s | \
// RUN:     FileCheck %s --check-prefix=DECODER

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def arch : Target {
  let InstructionSet = archInstrInfo;
}

def Myi32 : Operand<i32> {
  let DecoderMethod = "DecodeMyi32";
}

def HasA : Predicate<"Subtarget->hasA()">;
def HasB : Predicate<"Subtarget->hasB()">;

def ModeA : HwMode<"+a", [HasA]>; // hwmode ID 1
def ModeB : HwMode<"+b", [HasB]>; // hwmode ID 2


def fooTypeEncDefault : InstructionEncoding {
  let Size = 8;
  field bits<64> SoftFail = 0;
  bits<64> Inst;
  bits<8> factor;
  let Inst{7...0} = factor;
  let Inst{3...2} = 0b10;
  let Inst{1...0} = 0b00;
}

def fooTypeEncA : InstructionEncoding {
  let Size = 4;
  field bits<32> SoftFail = 0;
  bits<32> Inst;
  bits<8> factor;
  let Inst{7...0} = factor;
  let Inst{3...2} = 0b11;
  let Inst{1...0} = 0b00;
}

def fooTypeEncB : InstructionEncoding {
  let Size = 4;
  field bits<32> SoftFail = 0;
  bits<32> Inst;
  bits<8> factor;
  let Inst{15...8} = factor;
  let Inst{1...0} = 0b11;
}

// Test for DefaultMode as a selector.
def foo : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins i32imm:$factor);
  let EncodingInfos = EncodingByHwMode<
    [ModeA, ModeB, DefaultMode], [fooTypeEncA, fooTypeEncB, fooTypeEncDefault]
  >;
  let AsmString = "foo  $factor";
}

def bar: Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins i32imm:$factor);
  let Size = 4;
  bits<32> Inst;
  bits<32> SoftFail;
  bits<8> factor;
  let Inst{31...24} = factor;
  let Inst{1...0} = 0b10;
  let AsmString = "bar  $factor";
}

def baz : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins i32imm:$factor);
  bits<32> Inst;
  let EncodingInfos = EncodingByHwMode<
    [ModeB], [fooTypeEncA]
  >;
  let AsmString = "foo  $factor";
}

def unrelated: Instruction {
  let OutOperandList = (outs);
  let DecoderNamespace = "Alt";
  let InOperandList = (ins i32imm:$factor);
  let Size = 4;
  bits<32> Inst;
  bits<32> SoftFail;
  bits<8> factor;
  let Inst{31...24} = factor;
  let Inst{1...0} = 0b10;
  let AsmString = "unrelated  $factor";
}

// DecoderTable64 is assigned with 'DefaultMode',
// the table name should remain the same as before.
// DECODER-LABEL: DecoderTable64[] =
// DECODER-DAG: Opcode: fooTypeEncDefault:foo

// DecoderTable without any valid HwModes
// should not have any suffix in table name.
// The table should remain the same as before.
// DECODER-LABEL: DecoderTableAlt32[] =
// DECODER-DAG: Opcode: unrelated
// DECODER-NOT: DecoderTableAlt_ModeA32[]
// DECODER-NOT: DecoderTableAlt_ModeB32[]

// The DecoderTable32 contains two valid hwmodes, we will
// generate two tables corresponding to these hwmodes.
// Still want to say we didn't assign 'DefaultMode' to DecoderTable32,
// instead we assign 'DefaultMode' to DecoderTable64, so
// DecoderTable32 won't appear here.
// DECODER-LABEL: DecoderTable_ModeA32[] =
// DECODER-DAG: Opcode: fooTypeEncA:foo
// DECODER-DAG: Opcode: bar
// DECODER-LABEL: DecoderTable_ModeB32[] =
// DECODER-DAG: Opcode: fooTypeEncB:foo
// DECODER-DAG: Opcode: fooTypeEncA:baz
// DECODER-DAG: Opcode: bar


// For 'bar' and 'unrelated', we didn't assign any hwmodes for them,
// they should keep the same in the following three tables.
// For 'foo' we assigned three hwmodes(includes 'DefaultMode')
// it's encodings should be different in the following three tables.
// For 'baz' we only assigned ModeB for it, to avoid empty encoding
// we assigned the encoding of ModeB to ModeA and DefaultMode(Even though
// they will not be used).
// ENCODER-LABEL:   static const uint64_t InstBits[] = {
// ENCODER:         UINT64_C(2),        // bar
// To avoid empty encoding, we choose the encoding of ModeB for baz here
// ENCODER:         UINT64_C(12),        // baz
// ENCODER:         UINT64_C(8),        // foo
// ENCODER:         UINT64_C(2),        // unrelated

// ENCODER-LABEL:   static const uint64_t InstBits_ModeA[] = {
// ENCODER:         UINT64_C(2),        // bar
// To avoid empty encoding, we choose the encoding of ModeB for baz here
// ENCODER:         UINT64_C(12),        // baz
// ENCODER:         UINT64_C(12),       // foo
// ENCODER:         UINT64_C(2),        // unrelated

// ENCODER-LABEL:   static const uint64_t InstBits_ModeB[] = {
// ENCODER:         UINT64_C(2),        // bar
// ENCODER:         UINT64_C(12),       // baz
// ENCODER:         UINT64_C(3),        // foo
// ENCODER:         UINT64_C(2),        // unrelated

// Sink the logic of hwmodes selection to per instruction level.
// ENCODER-LABEL: case ::bar:
// ENCODER: op = getMachineOpValue
// ENCODER-LABEL: case ::baz: {
// ENCODER:  unsigned HwMode = STI.getHwMode();
// ENCODER:  HwMode &= 2;
// ENCODER:  switch (HwMode) {
// ENCODER:  default: llvm_unreachable("Unknown hardware mode!"); break;
// ENCODER:  case 2: InstBitsByHw = InstBits_ModeB; break;
// ENCODER:  };
// ENCODER:  Value = InstBitsByHw[opcode];
// ENCODER:  switch (HwMode) {
// ENCODER:  default: llvm_unreachable("Unhandled HwMode");
// ENCODER:  case 2: {
// ENCODER:  op = getMachineOpValue
// ENCODER-LABEL: case ::foo: {
// ENCODER:  unsigned HwMode = STI.getHwMode();
// ENCODER:  HwMode &= 3;
// ENCODER:  switch (HwMode) {
// ENCODER:  default: llvm_unreachable("Unknown hardware mode!"); break;
// ENCODER:  case 0: InstBitsByHw = InstBits; break;
// ENCODER:  case 1: InstBitsByHw = InstBits_ModeA; break;
// ENCODER:  case 2: InstBitsByHw = InstBits_ModeB; break;
// ENCODER:  };
// ENCODER:  Value = InstBitsByHw[opcode];
// ENCODER:  switch (HwMode) {
// ENCODER:  default: llvm_unreachable("Unhandled HwMode");
// ENCODER:  case 0: {
// ENCODER:  op = getMachineOpValue
// ENCODER:  case 1: {
// ENCODER:  op = getMachineOpValue
// ENCODER:  case 2: {
// ENCODER:  op = getMachineOpValue



