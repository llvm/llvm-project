// RUN: llvm-tblgen -gen-disassembler -I %p/../../include %s | FileCheck %s

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
    field bits<16> Inst;
    bits<32> factor;
    let Inst{7-0} = 0xAA;
    let Inst{14-8} = factor{6-0}; // no offset
    let AsmString = "foo  $factor";
    field bits<16> SoftFail = 0;
    }

def bar : Instruction {
    let InOperandList = (ins i32imm:$factor);
    field bits<16> Inst;
    bits<32> factor;
    let Inst{7-0} = 0xBB;
    let Inst{15-8} = factor{10-3}; // offset by 3
    let AsmString = "bar  $factor";
    field bits<16> SoftFail = 0;
    }

def biz : Instruction {
    let InOperandList = (ins i32imm:$factor);
    field bits<16> Inst;
    bits<32> factor;
    let Inst{7-0} = 0xCC;
    let Inst{11-8,15-12} = factor{10-3}; // offset by 3, multipart
    let AsmString = "biz  $factor";
    field bits<16> SoftFail = 0;
    }

def baz : Instruction {
    let InOperandList = (ins Myi32:$factor);
    field bits<16> Inst;
    bits<32> factor;
    let Inst{7-0} = 0xDD;
    let Inst{15-8} = factor{11-4}; // offset by 4 + custom decode
    let AsmString = "baz  $factor";
    field bits<16> SoftFail = 0;
    }

}

// CHECK: tmp = fieldFromInstruction(insn, 8, 7);
// CHECK: tmp = fieldFromInstruction(insn, 8, 8) << 3;
// CHECK: tmp |= fieldFromInstruction(insn, 8, 4) << 7;
// CHECK: tmp |= fieldFromInstruction(insn, 12, 4) << 3;
// CHECK: tmp = fieldFromInstruction(insn, 8, 8) << 4;
