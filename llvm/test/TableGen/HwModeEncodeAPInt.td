// This testcase is to test the correctness of HwMode encoding under the 'APInt' Mode.
// RUN: llvm-tblgen -gen-emitter -I %p/../../include %s | \
// RUN:     FileCheck %s --check-prefix=ENCODER

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

def ModeA : HwMode<[HasA]>; // Mode 1
def ModeB : HwMode<[HasB]>; // Mode 2
def ModeC : HwMode<[]>;     // Mode 3


def fooTypeEncDefault : InstructionEncoding {
  let Size = 16;
  field bits<128> SoftFail = 0;
  bits<128> Inst;
  bits<8> factor;
  let Inst{127...120} = factor;
  let Inst{3...2} = 0b10;
  let Inst{1...0} = 0b00;
}

def fooTypeEncA : InstructionEncoding {
  let Size = 16;
  field bits<128> SoftFail = 0;
  bits<128> Inst;
  bits<8> factor;
  let Inst{119...112} = factor;
  let Inst{3...2} = 0b11;
  let Inst{1...0} = 0b00;
}

def fooTypeEncB : InstructionEncoding {
  let Size = 16;
  field bits<128> SoftFail = 0;
  bits<128> Inst;
  bits<8> factor;
  let Inst{119...112} = factor;
  let Inst{111...110} = 0b11;
}

def fooTypeEncC : InstructionEncoding {
  let Size = 16;
  field bits<128> SoftFail = 0;
  bits<128> Inst;
  bits<8> factor;
  let Inst{31...24} = factor;
  let Inst{23...21} = 0b110;
  let Inst{1...0} = 0b11;
}

// Test for DefaultMode as a selector.
def foo : Instruction {
  bits<128> Inst;
  let OutOperandList = (outs);
  let InOperandList = (ins i32imm:$factor);
  let EncodingInfos = EncodingByHwMode<
  [ModeC, ModeA, ModeB, DefaultMode],
  [fooTypeEncC, fooTypeEncA, fooTypeEncB, fooTypeEncDefault]>;
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

// For 'bar' and 'unrelated', we didn't assign any HwModes for them,
// they should keep the same in the following four tables.
// For 'foo' we assigned four HwModes( includes 'DefaultMode' ),
// it's encodings should be different in the following four tables.
// For 'baz' we only assigned ModeB for it, so it will be presented
// as '0' in the tables of ModeA, ModeC and Default Mode.
// ENCODER-LABEL:   static const uint64_t InstBits[] = {
// ENCODER-NEXT:    UINT64_C(2), UINT64_C(0),       // bar
// ENCODER-NEXT:    UINT64_C(0), UINT64_C(0),       // baz
// ENCODER-NEXT:    UINT64_C(8), UINT64_C(0),       // foo
// ENCODER-NEXT:    UINT64_C(2), UINT64_C(0),       // unrelated
// ENCODER-NEXT:    };
// ENCODER-LABEL:   static const uint64_t InstBits_ModeA[] = {
// ENCODER-NEXT:    UINT64_C(2), UINT64_C(0),       // bar
// ENCODER-NEXT:    UINT64_C(0), UINT64_C(0),       // baz
// ENCODER-NEXT:    UINT64_C(12), UINT64_C(0),      // foo
// ENCODER-NEXT:    UINT64_C(2), UINT64_C(0),       // unrelated
// ENCODER-NEXT:    };
// ENCODER-LABEL:   static const uint64_t InstBits_ModeB[] = {
// ENCODER-NEXT:    UINT64_C(2), UINT64_C(0),       // bar
// ENCODER-NEXT:    UINT64_C(12), UINT64_C(0),      // baz
// ENCODER-NEXT:    UINT64_C(0), UINT64_C(211106232532992),  // foo
// ENCODER-NEXT:    UINT64_C(2), UINT64_C(0),       // unrelated
// ENCODER-NEXT:    };
// ENCODER-LABEL:   static const uint64_t InstBits_ModeC[] = {
// ENCODER-NEXT:    UINT64_C(2), UINT64_C(0),      // bar
// ENCODER-NEXT:    UINT64_C(0), UINT64_C(0),      // baz
// ENCODER-NEXT:    UINT64_C(12582915),  UINT64_C(0),  // foo
// ENCODER-NEXT:    UINT64_C(2),  UINT64_C(0),     // unrelated
// ENCODER-NEXT:    };

// ENCODER: const uint64_t *InstBitsByHw;
// ENCODER: constexpr unsigned FirstSupportedOpcode
// ENCODER: const unsigned opcode = MI.getOpcode();
// ENCODER: if (opcode < FirstSupportedOpcode)
// ENCODER: unsigned TableIndex = opcode - FirstSupportedOpcode
// ENCODER: if (Scratch.getBitWidth() != 128)
// ENCODER:   Scratch = Scratch.zext(128);
// ENCODER: Inst = APInt(128, ArrayRef(InstBits + TableIndex * 2, 2));
// ENCODER: APInt &Value = Inst;
// ENCODER: APInt &op = Scratch;
// ENCODER: switch (opcode) {
// ENCODER-LABEL: case ::bar:
// ENCODER-LABEL: case ::unrelated:
// ENCODER-NOT: getHwMode
// ENCODER-LABEL: case ::foo: {
// ENCODER: unsigned HwMode = STI.getHwMode(MCSubtargetInfo::HwMode_EncodingInfo);
// ENCODER: switch (HwMode) {
// ENCODER: default: llvm_unreachable("Unknown hardware mode!"); break;
// ENCODER: case 0: InstBitsByHw = InstBits; break;
// ENCODER: case 1: InstBitsByHw = InstBits_ModeA; break;
// ENCODER: case 2: InstBitsByHw = InstBits_ModeB; break;
// ENCODER: case 3: InstBitsByHw = InstBits_ModeC; break;
// ENCODER: };
// ENCODER: Inst = APInt(128, ArrayRef(InstBitsByHw + TableIndex * 2, 2));
// ENCODER: Value = Inst;
// ENCODER: switch (HwMode) {
// ENCODER: default: llvm_unreachable("Unhandled HwMode");
// ENCODER: case 0: {
// ENCODER: op.clearAllBits();
// ENCODER: getMachineOpValue(MI, MI.getOperand(0), op, Fixups, STI);
// ENCODER: Value.insertBits(op.extractBitsAsZExtValue(8, 0), 120, 8);
// ENCODER: break;
// ENCODER: }
// ENCODER: case 1: {
// ENCODER: op.clearAllBits();
// ENCODER: getMachineOpValue(MI, MI.getOperand(0), op, Fixups, STI);
// ENCODER: Value.insertBits(op.extractBitsAsZExtValue(8, 0), 112, 8);
// ENCODER: break;
// ENCODER: }
// ENCODER: case 2: {
// ENCODER: op.clearAllBits();
// ENCODER: getMachineOpValue(MI, MI.getOperand(0), op, Fixups, STI);
// ENCODER: Value.insertBits(op.extractBitsAsZExtValue(8, 0), 112, 8);
// ENCODER: break;
// ENCODER: }
// ENCODER: case 3: {
// ENCODER: op.clearAllBits();
// ENCODER: getMachineOpValue(MI, MI.getOperand(0), op, Fixups, STI);
// ENCODER: Value.insertBits(op.extractBitsAsZExtValue(8, 0), 24, 8);
// ENCODER: break;
// ENCODER: }
// ENCODER-LABEL: case ::baz: {
// ENCODER: unsigned HwMode = STI.getHwMode(MCSubtargetInfo::HwMode_EncodingInfo);
// ENCODER: switch (HwMode) {
// ENCODER: default: llvm_unreachable("Unknown hardware mode!"); break;
// ENCODER: case 2: InstBitsByHw = InstBits_ModeB; break;
// ENCODER: };
// ENCODER: Inst = APInt(128, ArrayRef(InstBitsByHw + TableIndex * 2, 2));
// ENCODER: Value = Inst;
// ENCODER: switch (HwMode) {
// ENCODER: default: llvm_unreachable("Unhandled HwMode");
// ENCODER: case 2: {
// ENCODER: getMachineOpValue(MI, MI.getOperand(0), op, Fixups, STI);
// ENCODER: Value.insertBits(op.extractBitsAsZExtValue(8, 0), 112, 8);
// ENCODER: break;
// ENCODER: }

// ENCODER-LABEL: uint32_t archMCCodeEmitter::getOperandBitOffset
// ENCODER: switch (MI.getOpcode()) {
// ENCODER-LABEL: case ::bar:
// ENCODER-LABEL: case ::unrelated: {
// ENCODER-NOT: getHwMode
// ENCODER-LABEL: case ::foo: {
// ENCODER:   unsigned HwMode = STI.getHwMode(MCSubtargetInfo::HwMode_EncodingInfo);
// ENCODER:   switch (HwMode) {
// ENCODER:   default: llvm_unreachable("Unhandled HwMode");
// ENCODER:   case 0: {
// ENCODER:   switch (OpNum) {
// ENCODER:   case 0:
// ENCODER:     return 120;
// ENCODER:   }
// ENCODER:   break;
// ENCODER:   }
// ENCODER:   case 1: {
// ENCODER:   switch (OpNum) {
// ENCODER:   case 0:
// ENCODER:     return 112;
// ENCODER:   }
// ENCODER:   break;
// ENCODER:   }
// ENCODER:   case 2: {
// ENCODER:   switch (OpNum) {
// ENCODER:   case 0:
// ENCODER:     return 112;
// ENCODER:   }
// ENCODER:   break;
// ENCODER:   }
// ENCODER:   case 3: {
// ENCODER:   switch (OpNum) {
// ENCODER:   case 0:
// ENCODER:     return 24;
// ENCODER:   }
// ENCODER:   break;
// ENCODER:   }
// ENCODER:   }
// ENCODER:   break;
// ENCODER: }
