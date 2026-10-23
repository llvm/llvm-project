// Check that the per-HwMode encoding of an instruction runs as one statement of
// the bytecode program, scoped so that its HwMode local does not reach the
// statements next to it.
// RUN: llvm-tblgen -gen-emitter -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { let UseBytecodeEncoder = 1; }

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


// CHECK:      case 2:
// CHECK-NEXT:   {
// CHECK-NEXT:   unsigned HwMode = STI.getHwMode(MCSubtargetInfo::HwMode_EncodingInfo);
// CHECK:        Inst = APInt(128, ArrayRef(InstBitsByHw + TableIndex * 2, 2));
// CHECK:        default: llvm_unreachable("Unhandled HwMode");
// CHECK:        Value.insertBits(op.extractBitsAsZExtValue(8, 0), 24, 8);
// CHECK-NEXT:     break;
// CHECK-NEXT:   }
// CHECK-NEXT:   }
// CHECK-NEXT:   }
// CHECK-NEXT:   break;
// CHECK-NEXT: }
