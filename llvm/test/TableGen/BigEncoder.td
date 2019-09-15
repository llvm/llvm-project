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
// CHECK: const APInt [[x:M[0-9]+]] = APInt::getBitsSet(65, 3, 7);
// CHECK-NEXT: Value |= (op & [[x]]) << 9;
// CHECK-NEXT: const APInt [[y:M[0-9]+]] = APInt::getBitsSet(65, 7, 11);
// CHECK-NEXT: Value |= (op & [[y]]) << 1;

// CHECK-LABEL: case ::foo: {
// CHECK: const APInt [[x:M[0-9]+]] = APInt::getBitsSet(65, 0, 7);
// CHECK-NEXT: op &= [[x]];
// CHECK-NEXT: op <<= 8;
// CHECK-NEXT: Value |= op;

// CHECK-LABEL: case ::bar: {
// CHECK: const APInt [[x:M[0-9]+]] = APInt::getBitsSet(65, 3, 11);
// CHECK-NEXT: op &= [[x]];
// CHECK-NEXT: op <<= 5;
// CHECK-NEXT: Value |= op;
