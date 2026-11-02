// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s --check-prefix=NODBG
// RUN: llvm-tblgen -gen-disassembler -emit-decoder-debug -I %p/../../../include %s | FileCheck %s
// RUN: llvm-tblgen -gen-disassembler -emit-decoder-debug -specialize-decoders-per-bitwidth \
// RUN:   -I %p/../../../include %s | FileCheck %s --check-prefix=SPEC

// Decode-failure debug logging is emitted only with -emit-decoder-debug.

// NODBG-NOT: {{report(Operand|Insn)}}

include "llvm/Target/Target.td"

def R0 : Register<"r0">;
def RC : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

def MyInstrInfo : InstrInfo;

def MyTarget : Target {
  let InstructionSet = MyInstrInfo;
}

def ImmOp : Operand<i32> {
  let DecoderMethod = "DecodeImmOp";
  let hasCompleteDecoder = 0;
}

def I : Instruction {
  let OutOperandList = (outs RC:$dst);
  let InOperandList = (ins RC:$none, ImmOp:$k);
  let Size = 4;
  bits<32> Inst;
  bits<4> dst;
  bits<0> none;
  bits<4> k = 0b0101;
  let Inst{3...0} = dst;
  let Inst{31...4} = 0;
}

def J : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins);
  let Size = 4;
  bits<32> Inst;
  let Inst{31...0} = 1;
  let DecoderMethod = "DecodeJ";
  let hasCompleteDecoder = 0;
}

// CHECK: #ifndef NDEBUG

// CHECK{LITERAL}: [[maybe_unused]] void reportOperandDecodeFailure(const char *OperandName,
// CHECK-NEXT:                          const char *Desc,
// CHECK-NEXT:                          const char *Decoder) {
// CHECK-NEXT:   dbgs() << "OPC_Decode for " << OperandName << ": " << Desc << " using "
// CHECK-NEXT:          << Decoder << " : FAIL\n";
// CHECK-NEXT: }

// CHECK{LITERAL}: [[maybe_unused]] void
// CHECK-NEXT: reportOperandDecodeFailure(const char *OperandName, const char *Decoder,
// CHECK-NEXT:                          ArrayRef<std::pair<uint32_t, uint32_t>> Fields) {
// CHECK-NEXT:   dbgs() << "OPC_Decode for " << OperandName << ": ";
// CHECK-NEXT:   bool First = true;
// CHECK-NEXT:   for (auto [Base, Width] : Fields) {
// CHECK-NEXT:     if (!First)
// CHECK-NEXT:       dbgs() << " + ";
// CHECK-NEXT:     First = false;
// CHECK-NEXT:     dbgs() << "field(" << Base << ',' << Width << ')';
// CHECK-NEXT:   }
// CHECK-NEXT:   dbgs() << " using " << Decoder << " : FAIL\n";
// CHECK-NEXT: }

// CHECK{LITERAL}: [[maybe_unused]] void reportInsnDecodeFailure(unsigned Opcode,
// CHECK-NEXT:                          const char *Decoder) {
// CHECK-NEXT:   dbgs() << "OPC_Decode for opcode " << Opcode << ": insn using " << Decoder
// CHECK-NEXT:          << " : FAIL\n";
// CHECK-NEXT: }

// CHECK{LITERAL}: [[maybe_unused]] void reportInsnBits(const InsnType &Insn, unsigned NumBits) {
// CHECK-NEXT:   dbgs() << "decodeInstruction: insn = 0x";
// CHECK-NEXT:   for (unsigned Hi = NumBits; Hi != 0;) {
// CHECK-NEXT:     unsigned Width = std::min(Hi, 64U);
// CHECK-NEXT:     Hi -= Width;
// CHECK-NEXT:     uint64_t Chunk = fieldFromInstruction(Insn, Hi, Width);
// CHECK-NEXT:     for (int Shift = (Width + 3) / 4 * 4 - 4; Shift >= 0; Shift -= 4)
// CHECK-NEXT:       dbgs() << "0123456789abcdef"[(Chunk >> Shift) & 0xf];
// CHECK-NEXT:   }
// CHECK-NEXT:   dbgs() << '\n';
// CHECK-NEXT: }
// CHECK-NEXT: #endif // NDEBUG

// One call per operand kind: fields, 'bits<0>', and a constant ('imm').
// CHECK-LABEL: static DecodeStatus decodeToMCInst(
// CHECK:       case 0:
// CHECK-NEXT:    if (!Check(S, DecodeJ(MI, insn, Address, Decoder))) {
// CHECK-NEXT:      LLVM_DEBUG(reportInsnDecodeFailure(MI.getOpcode(), "DecodeJ"));
// CHECK-NEXT:      DecodeComplete = false;
// CHECK-NEXT:      return MCDisassembler::Fail;
// CHECK-NEXT:    }
// CHECK-NEXT:    return S;
// CHECK-NEXT:  case 1:
// CHECK-NEXT:    tmp = fieldFromInstruction(insn, 0, 4);
// CHECK-NEXT:    if (!Check(S, DecodeRCRegisterClass(MI, tmp, Address, Decoder))) {
// CHECK-NEXT{LITERAL}:      LLVM_DEBUG(reportOperandDecodeFailure("dst", "DecodeRCRegisterClass", {{0, 4}}));
// CHECK-NEXT:      return MCDisassembler::Fail;
// CHECK-NEXT:    }
// CHECK-NEXT:    if (!Check(S, DecodeRCRegisterClass(MI, Decoder))) {
// CHECK-NEXT:      LLVM_DEBUG(reportOperandDecodeFailure("none", "bits<0>", "DecodeRCRegisterClass"));
// CHECK-NEXT:      return MCDisassembler::Fail;
// CHECK-NEXT:    }
// CHECK-NEXT:    tmp = 0x5;
// CHECK-NEXT:    if (!Check(S, DecodeImmOp(MI, tmp, Address, Decoder))) {
// CHECK-NEXT:      LLVM_DEBUG(reportOperandDecodeFailure("k", "imm", "DecodeImmOp"));
// CHECK-NEXT:      DecodeComplete = false;
// CHECK-NEXT:      return MCDisassembler::Fail;
// CHECK-NEXT:    }
// CHECK-NEXT:    return S;

// CHECK: LLVM_DEBUG(reportInsnBits(insn, std::is_integral_v<InsnType> ? sizeof(InsnType) * 8 : 64));
// SPEC: LLVM_DEBUG(reportInsnBits(insn, BitWidth));
