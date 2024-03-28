// RUN: llvm-tblgen -gen-emitter -I %p/../../include %s | \
// RUN:     FileCheck %s --check-prefix=ENCODER
// RUN: llvm-tblgen -gen-disassembler -I %p/../../include %s | \
// RUN:     FileCheck %s --check-prefix=DECODER
// RUN: llvm-tblgen -gen-disassembler --suppress-per-hwmode-duplicates=O1 -I \
// RUN:     %p/../../include %s | FileCheck %s --check-prefix=DECODER-SUPPRESS-O1
// RUN: llvm-tblgen -gen-disassembler --suppress-per-hwmode-duplicates=O2 -I \
// RUN:     %p/../../include %s | FileCheck %s --check-prefix=DECODER-SUPPRESS-O2

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

def ModeA : HwMode<"+a", [HasA]>;
def ModeB : HwMode<"+b", [HasB]>;


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


// Under default settings, using 'HwMode' to dictate instruction encodings results in
// significant duplication of DecoderTables. The three tables ‘DecoderTableAlt32’,
// ‘DecoderTableAlt_ModeA32’, and ‘DecoderTableAlt_ModeB32’ are exact duplicates and
// could effectively be merged into one.
// DECODER-LABEL: DecoderTable32[] =
// DECODER-DAG: Opcode: bar
// DECODER-LABEL: DecoderTable64[] =
// DECODER-DAG: Opcode: fooTypeEncDefault:foo
// DECODER-LABEL: DecoderTableAlt32[] =
// DECODER-DAG: Opcode: unrelated
// DECODER-LABEL: DecoderTableAlt_ModeA32[] =
// DECODER-DAG: Opcode: unrelated
// DECODER-LABEL: DecoderTableAlt_ModeB32[] =
// DECODER-DAG: Opcode: unrelated
// DECODER-LABEL: DecoderTable_ModeA32[] =
// DECODER-DAG: Opcode: fooTypeEncA:foo
// DECODER-DAG: Opcode: bar
// DECODER-LABEL: DecoderTable_ModeB32[] =
// DECODER-DAG: Opcode: fooTypeEncB:foo
// DECODER-DAG: Opcode: fooTypeEncA:baz
// DECODER-DAG: Opcode: bar

// Under the 'O1' optimization level, unnecessary duplicate tables will be eliminated,
// reducing the three ‘Alt’ tables down to just one.
// DECODER-SUPPRESS-O1-LABEL: DecoderTable32[] =
// DECODER-SUPPRESS-O1-DAG: Opcode: bar
// DECODER-SUPPRESS-O1-LABEL: DecoderTable64[] =
// DECODER-SUPPRESS-O1-DAG: Opcode: fooTypeEncDefault:foo
// DECODER-SUPPRESS-O1-LABEL: DecoderTableAlt32[] =
// DECODER-SUPPRESS-O1-DAG: Opcode: unrelated
// DECODER-SUPPRESS-O1-LABEL: DecoderTable_ModeA32[] =
// DECODER-SUPPRESS-O1-DAG: Opcode: fooTypeEncA:foo
// DECODER-SUPPRESS-O1-DAG: Opcode: bar
// DECODER-SUPPRESS-O1-LABEL: DecoderTable_ModeB32[] =
// DECODER-SUPPRESS-O1-DAG: Opcode: fooTypeEncB:foo
// DECODER-SUPPRESS-O1-DAG: Opcode: fooTypeEncA:baz
// DECODER-SUPPRESS-O1-DAG: Opcode: bar

// Under the 'O2' optimization condition, instructions possessing the 'EncodingByHwMode'
// attribute will be extracted from their original DecoderNamespace and placed into their
// respective HwMode tables. Meanwhile, other instructions that do not have the 'EncodingByHwMode'
// attribute but are within the same DecoderNamespace will be stored in the 'Default' table. This
// approach will significantly reduce instruction redundancy, but it necessitates users to thoroughly
// consider the interplay between HwMode and DecoderNamespace for their instructions.
// DECODER-SUPPRESS-O2-LABEL: DecoderTable32[] =
// DECODER-SUPPRESS-O2-DAG: Opcode: bar
// DECODER-SUPPRESS-O2-LABEL: DecoderTable64[] =
// DECODER-SUPPRESS-O2-NOT: Opcode: bar
// DECODER-SUPPRESS-O2-DAG: Opcode: fooTypeEncDefault:foo
// DECODER-SUPPRESS-O2-LABEL: DecoderTableAlt32[] =
// DECODER-SUPPRESS-O2-DAG: Opcode: unrelated
// DECODER-SUPPRESS-O2-LABEL: DecoderTable_ModeA32[] =
// DECODER-SUPPRESS-O2-DAG: Opcode: fooTypeEncA:foo
// DECODER-SUPPRESS-O2-NOT: Opcode: bar
// DECODER-SUPPRESS-O2-LABEL: DecoderTable_ModeB32[] =
// DECODER-SUPPRESS-O2-DAG: Opcode: fooTypeEncB:foo
// DECODER-SUPPRESS-O2-DAG: Opcode: fooTypeEncA:baz
// DECODER-SUPPRESS-O2-NOT: Opcode: bar

// ENCODER-LABEL:   static const uint64_t InstBits_DefaultMode[] = {
// ENCODER:         UINT64_C(2),        // bar
// ENCODER:         UINT64_C(0),        // baz
// ENCODER:         UINT64_C(8),        // foo
// ENCODER:         UINT64_C(2),        // unrelated

// ENCODER-LABEL:   static const uint64_t InstBits_ModeA[] = {
// ENCODER:         UINT64_C(2),        // bar
// ENCODER:         UINT64_C(0),        // baz
// ENCODER:         UINT64_C(12),       // foo
// ENCODER:         UINT64_C(2),        // unrelated

// ENCODER-LABEL:   static const uint64_t InstBits_ModeB[] = {
// ENCODER:         UINT64_C(2),        // bar
// ENCODER:         UINT64_C(12),       // baz
// ENCODER:         UINT64_C(3),        // foo
// ENCODER:         UINT64_C(2),        // unrelated

// ENCODER:  unsigned HwMode = STI.getHwMode();
// ENCODER:  switch (HwMode) {
// ENCODER:  default: llvm_unreachable("Unknown hardware mode!"); break;
// ENCODER:  case 0: InstBits = InstBits_DefaultMode; break;
// ENCODER:  case 1: InstBits = InstBits_ModeA; break;
// ENCODER:  case 2: InstBits = InstBits_ModeB; break;
// ENCODER:  };

// ENCODER:     case ::foo: {
// ENCODER:      switch (HwMode) {
// ENCODER:      default: llvm_unreachable("Unhandled HwMode");
// ENCODER:      case 0: {
// ENCODER:      case 1: {
// ENCODER:      case 2: {

