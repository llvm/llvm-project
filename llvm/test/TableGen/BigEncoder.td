// RUN: llvm-tblgen -gen-emitter -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def arch : Target {
    let InstructionSet = archInstrInfo;
}

def  Myi32  : Operand<i32> {
  let DecoderMethod = "DecodeMyi32";
}


let OutOperandList = (outs), Size = 2 in {

def foo : Instruction {
    let InOperandList = (ins i32imm:$factor);
    field bits<65> Inst;
    bits<32> factor;
    let Inst{7-0} = 0xAA;
    let Inst{14-8} = factor{6-0}; // no offset
    let AsmString = "foo  $factor";
    field bits<16> SoftFail = 0;
    }

def bar : Instruction {
    let InOperandList = (ins i32imm:$factor);
    field bits<65> Inst;
    bits<32> factor;
    let Inst{7-0} = 0xBB;
    let Inst{15-8} = factor{10-3}; // offset by 3
    let AsmString = "bar  $factor";
    field bits<16> SoftFail = 0;
    }

def biz : Instruction {
    let InOperandList = (ins i32imm:$factor);
    field bits<65> Inst;
    bits<32> factor;
    let Inst{7-0} = 0xCC;
    let Inst{11-8,15-12} = factor{10-3}; // offset by 3, multipart
    let AsmString = "biz  $factor";
    field bits<16> SoftFail = 0;
    }

}
// CHECK-LABEL: case ::biz: {
// CHECK:      Value.insertBits(op.extractBitsAsZExtValue(4, 3), 12, 4);
// CHECK-NEXT: Value.insertBits(op.extractBitsAsZExtValue(4, 7), 8, 4);

// CHECK-LABEL: case ::foo: {
// CHECK:      Value.insertBits(op.extractBitsAsZExtValue(7, 0), 8, 7);

// CHECK-LABEL: case ::bar: {
// CHECK:      Value.insertBits(op.extractBitsAsZExtValue(8, 3), 8, 8);
