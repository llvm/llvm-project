// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s

// Verify the per-operand decode debug path for an operand whose encoding bits
// are spread across several disjoint islands of the instruction. Each island
// becomes one (base, width) pair; the call site passes them as raw integers
// and reportOperandDecodeFailure joins them with " + " at runtime, so no unique
// "field(b,w) + ..." string literal is baked into the table.

include "llvm/Target/Target.td"

def R0 : Register<"r0">;
def RC : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

def MyInstrInfo : InstrInfo;

def MyTarget : Target {
  let InstructionSet = MyInstrInfo;
}

// Operand `op` (6 bits) is split across three disjoint islands in Inst:
//   op{1:0} -> Inst{1:0}, op{3:2} -> Inst{9:8}, op{5:4} -> Inst{20:19}.
def I : Instruction {
  let OutOperandList = (outs RC:$op);
  let InOperandList = (ins);
  let Size = 4;
  bits<32> Inst;
  bits<6> op;
  let Inst{1...0}   = op{1...0};
  let Inst{9...8}   = op{3...2};
  let Inst{20...19} = op{5...4};
  let Inst{31} = 0;
}

// The operand is assembled from the three islands, then the field overload of
// reportOperandDecodeFailure is called with all three (base, width) pairs.
// CHECK-LABEL: case 0:
// CHECK-NEXT:    tmp = 0x0;
// CHECK-NEXT:    tmp |= fieldFromInstruction(insn, 0, 2);
// CHECK-NEXT:    tmp |= fieldFromInstruction(insn, 8, 2) << 2;
// CHECK-NEXT:    tmp |= fieldFromInstruction(insn, 19, 2) << 4;
// CHECK-NEXT:    if (!Check(S, DecodeRCRegisterClass(MI, tmp, Address, Decoder))) {
// CHECK-NEXT:      LLVM_DEBUG(reportOperandDecodeFailure("op", "DecodeRCRegisterClass", {0, 2, 8, 2, 19, 2}));
// CHECK-NEXT:      return MCDisassembler::Fail;
// CHECK-NEXT:    }
// CHECK-NEXT:    return S;
