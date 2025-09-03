// RUN: llvm-tblgen -gen-disassembler -I %p/../../include %s | \
// RUN:     FileCheck %s --check-prefix=DECODER
// RUN: llvm-tblgen -gen-disassembler --suppress-per-hwmode-duplicates=O2 -I \
// RUN:     %p/../../include %s | FileCheck %s --check-prefix=DECODER-SUPPRESS

// Test duplicate table suppression for per-HwMode decoders.

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
      [ModeA, ModeB], [fooTypeEncA, fooTypeEncB]
    >;
    let AsmString = "foo  $factor";
  }

  // Encoding not overridden, same namespace:
  // In the default case, this instruction is duplicated into both ModeA and
  // ModeB decoder tables.
  // In the suppressed case, this instruction appears in a single decoder table.
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

  // Encoding not overridden, different namespace:
  // In the default case, this instruction is duplicated into two Alt decoder
  // tables (ModeA and ModeB).
  // In the suppressed case, this instruction appears in a single decoder table.
  def unrelated: Instruction {
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
}

// DECODER-LABEL: DecoderTable_ModeA32
// DECODER-DAG: Opcode: fooTypeEncA:foo
// DECODER-DAG: Opcode: bar
// DECODER-LABEL: DecoderTable_ModeB32
// DECODER-DAG: Opcode: fooTypeEncB:foo
// DECODER-DAG: Opcode: fooTypeEncA:baz
// DECODER-DAG: Opcode: bar
// DECODER-LABEL: DecoderTableAlt_ModeA32
// DECODER-DAG: Opcode: unrelated
// DECODER-LABEL: DecoderTableAlt_ModeB32
// DECODER-DAG: Opcode: unrelated

// DECODER-SUPPRESS-LABEL: DecoderTable32
// DECODER-SUPPRESS-DAG: Opcode: bar
// DECODER-SUPPRESS-LABEL: DecoderTable_ModeA32
// DECODER-SUPPRESS-DAG: Opcode: fooTypeEncA:foo
// DECODER-SUPPRESS-NOT: Opcode: bar
// DECODER-SUPPRESS-LABEL: DecoderTable_ModeB32
// DECODER-SUPPRESS-DAG: Opcode: fooTypeEncB:foo
// DECODER-SUPPRESS-DAG: Opcode: fooTypeEncA:baz
// DECODER-SUPPRESS-NOT: Opcode: bar
// DECODER-SUPPRESS-LABEL: DecoderTableAlt32
// DECODER-SUPPRESS-DAG: Opcode: unrelated
