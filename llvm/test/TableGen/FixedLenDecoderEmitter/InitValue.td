// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def arch : Target {
    let InstructionSet = archInstrInfo;
}

let OutOperandList = (outs), Size = 2 in {

def foo : Instruction {
    let InOperandList = (ins i32imm:$factor);
    field bits<16> Inst;
    field bits<16> SoftFail = 0;
    bits<8> factor;
    let factor{0} = 0; // zero initial value
    let Inst{15...8} = factor{7...0};
    }

def bar : Instruction {
    let InOperandList = (ins i32imm:$factor);
    field bits<16> Inst;
    field bits<16> SoftFail = 0;
    bits<8> factor;
    let factor{0} = 1; // non-zero initial value
    let Inst{15...8} = factor{7...0};
    }

def bax : Instruction {
    let InOperandList = (ins i32imm:$factor);
    field bits<16> Inst;
    field bits<16> SoftFail = 0;
    bits<33> factor;
    let factor{32} = 1; // non-zero initial value
    let Inst{15...8} = factor{32...25};
    }

}

// CHECK: tmp = fieldFromInstruction(insn, 9, 7) << 1;
// CHECK: tmp = 0x1;
// CHECK: tmp |= fieldFromInstruction(insn, 9, 7) << 1;
// CHECK: tmp = 0x100000000;
// CHECK: tmp |= fieldFromInstruction(insn, 8, 7) << 25;
