// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s

include "llvm/Target/Target.td"

class I<dag outs, dag ins, bits<5> opcode> : Instruction {
  let InOperandList = ins;
  let OutOperandList = outs;

  bits<5> dst;
  bits<5> src0;
  bits<5> src1;

  int Size = 16;
  bits<128> Inst;

  let Inst{4...0} = opcode;
  let Inst{9...5} = dst;
  let Inst{14...10} = src0;
  let Inst{19...15} = src1;

  let Inst{127...20} = 0xdeadbeef;
}

def Reg : Register<"reg">;

def Regs : RegisterClass<"foo", [i32], 0, (add Reg)>;

def IAdd : I<(outs Regs:$dst), (ins Regs:$src0, Regs:$src1), 0b10101>;

// CHECK-LABEL: static const uint8_t DecoderTable128[20] = {
// CHECK-NEXT:   OPC_CheckField, 84, 44, 0,                 //  0: check Inst[127:84] == 0x0
// CHECK-NEXT:   OPC_CheckField, 20, 64, 239, 253, 182, 245, 13, 

def II : InstrInfo;

def MyTarget : Target {
  let InstructionSet = II;
}
