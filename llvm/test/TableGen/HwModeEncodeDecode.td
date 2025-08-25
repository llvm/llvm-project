// RUN: llvm-tblgen -gen-emitter -I %p/../../include %s | FileCheck %s --check-prefix=ENCODER
// RUN: llvm-tblgen -gen-disassembler -I %p/../../include %s | FileCheck %s --check-prefix=DECODER

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def arch : Target {
    let InstructionSet = archInstrInfo;
}

def  Myi32  : Operand<i32> {
  let DecoderMethod = "DecodeMyi32";
}

def HasA : Predicate<"Subtarget->hasA()">;
def HasB : Predicate<"Subtarget->hasB()">;

def ModeA : HwMode<"+a", [HasA]>;
def ModeB : HwMode<"+b", [HasB]>;


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

let OutOperandList = (outs) in {
def foo : Instruction {
  let InOperandList = (ins i32imm:$factor);
  let EncodingInfos = EncodingByHwMode<
    [ModeA, ModeB], [fooTypeEncA,
                      fooTypeEncB]
  >;
  let AsmString = "foo  $factor";
}

def bar: Instruction {
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
  let InOperandList = (ins i32imm:$factor);
  bits<32> Inst;
  let EncodingInfos = EncodingByHwMode<
    [ModeB], [fooTypeEncA]
  >;
  let AsmString = "foo  $factor";
}
}

// DECODER-LABEL: DecoderTable_ModeA32[22] =
// DECODER-DAG: Opcode: fooTypeEncA:foo
// DECODER-DAG: Opcode: bar
// DECODER-LABEL: DecoderTable_ModeB32[30] =
// DECODER-DAG: Opcode: fooTypeEncB:foo
// DECODER-DAG: Opcode: fooTypeEncA:baz
// DECODER-DAG: Opcode: bar

// ENCODER-LABEL:   static const uint64_t InstBits_ModeA[] = {
// ENCODER:         UINT64_C(2),        // bar
// ENCODER:         UINT64_C(12),       // foo

// ENCODER-LABEL:   static const uint64_t InstBits_ModeB[] = {
// ENCODER:         UINT64_C(2),        // bar
// ENCODER:         UINT64_C(3),        // foo

// ENCODER:     case ::foo: {
// ENCODER:      switch (HwMode) {
// ENCODER:      default: llvm_unreachable("Unhandled HwMode");
// ENCODER:      case 1: {
