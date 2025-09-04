// RUN: llvm-tblgen -gen-disassembler -use-fn-table-in-decode-to-mcinst -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def arch : Target {
  let InstructionSet = archInstrInfo;
}

let Namespace = "arch" in {
  def R0 : Register<"r0">;
  def R1 : Register<"r1">;
  def R2 : Register<"r2">;
  def R3 : Register<"r3">;
}
def Regs : RegisterClass<"Regs", [i32], 32, (add R0, R1, R2, R3)>;

class TestInstruction : Instruction {
  let Size = 1;
  let OutOperandList = (outs);
  field bits<8> Inst;
  field bits<8> SoftFail = 0;
}

// Define instructions to generate 4 cases in decodeToMCInst.
// Lower 2 bits define the number of operands. Each register operand
// needs 2 bits to encode.

// An instruction with no inputs. Encoded with lower 2 bits = 0 and upper
// 6 bits = 0 as well.
def Inst0 : TestInstruction {
  let Inst = 0x0;
  let InOperandList = (ins);
  let AsmString = "Inst0";
}

// An instruction with a single input. Encoded with lower 2 bits = 1 and the
// single input in bits 2-3.
def Inst1 : TestInstruction {
  bits<2> r0;
  let Inst{1-0} = 1;
  let Inst{3-2} = r0;
  let InOperandList = (ins Regs:$r0);
  let AsmString = "Inst1";
}

// An instruction with two inputs. Encoded with lower 2 bits = 2 and the
// inputs in bits 2-3 and 4-5.
def Inst2 : TestInstruction {
  bits<2> r0;
  bits<2> r1;
  let Inst{1-0} = 2;
  let Inst{3-2} = r0;
  let Inst{5-4} = r1;
  let InOperandList = (ins Regs:$r0, Regs:$r1);
  let AsmString = "Inst2";
}

// An instruction with three inputs. Encoded with lower 2 bits = 3 and the
// inputs in bits 2-3 and 4-5 and 6-7.
def Inst3 : TestInstruction {
  bits<2> r0;
  bits<2> r1;
  bits<2> r2;
  let Inst{1-0} = 3;
  let Inst{3-2} = r0;
  let Inst{5-4} = r1;
  let Inst{7-6} = r2;
  let InOperandList = (ins Regs:$r0, Regs:$r1, Regs:$r2);
  let AsmString = "Inst3";
}

// CHECK-LABEL: DecodeStatus decodeFn_0(DecodeStatus S, InsnType insn, MCInst &MI, uint64_t Address, const MCDisassembler *Decoder, bool &DecodeComplete)
// CHECK-LABEL: DecodeStatus decodeFn_1(DecodeStatus S, InsnType insn, MCInst &MI, uint64_t Address, const MCDisassembler *Decoder, bool &DecodeComplete)
// CHECK-LABEL: DecodeStatus decodeFn_2(DecodeStatus S, InsnType insn, MCInst &MI, uint64_t Address, const MCDisassembler *Decoder, bool &DecodeComplete)
// CHECK-LABEL: DecodeStatus decodeFn_3(DecodeStatus S, InsnType insn, MCInst &MI, uint64_t Address, const MCDisassembler *Decoder, bool &DecodeComplete)
// CHECK-LABEL: decodeToMCInst(unsigned Idx, DecodeStatus S, InsnType insn, MCInst &MI, uint64_t Address, const MCDisassembler *Decoder, bool &DecodeComplete)
// CHECK: static constexpr DecodeFnTy decodeFnTable[]
// CHECK-NEXT: decodeFn_0,
// CHECK-NEXT: decodeFn_1,
// CHECK-NEXT: decodeFn_2,
// CHECK-NEXT: decodeFn_3,
// CHECK: return decodeFnTable[Idx](S, insn, MI, Address, Decoder, DecodeComplete)
