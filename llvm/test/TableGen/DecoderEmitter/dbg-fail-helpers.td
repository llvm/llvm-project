// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s

// Verify the DecoderEmitter emits the reportOperandDecodeFailure / reportInsnDecodeFailure helpers used
// by per-operand and whole-instruction decode debug logging.

include "llvm/Target/Target.td"

def R0 : Register<"r0">;
def RC : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

def MyInstrInfo : InstrInfo;

def MyTarget : Target {
  let InstructionSet = MyInstrInfo;
}

def I : Instruction {
  let OutOperandList = (outs RC:$dst);
  let InOperandList = (ins);
  let Size = 4;
  bits<32> Inst;
  bits<4> dst;
  let Inst{3...0} = dst;
  let Inst{31...4} = 0;
}

// CHECK: #ifndef NDEBUG

// CHECK{LITERAL}: [[maybe_unused]] void reportOperandDecodeFailure(const char *Op,
// CHECK-NEXT:                          const char *Desc,
// CHECK-NEXT:                          const char *Decoder) {
// CHECK-NEXT:   dbgs() << "OPC_Decode for " << Op << ": " << Desc << " using " << Decoder
// CHECK-NEXT:          << " : FAIL\n";
// CHECK-NEXT: }

// CHECK{LITERAL}: [[maybe_unused]] void reportOperandDecodeFailure(const char *Op,
// CHECK-NEXT:                          const char *Decoder,
// CHECK-NEXT:                          ArrayRef<uint32_t> Fields) {
// CHECK-NEXT:   dbgs() << "OPC_Decode for " << Op << ": ";
// CHECK-NEXT:   for (unsigned I = 0, E = Fields.size(); I != E; I += 2) {
// CHECK-NEXT:     if (I)
// CHECK-NEXT:       dbgs() << " + ";
// CHECK-NEXT:     dbgs() << "field(" << Fields[I] << ',' << Fields[I + 1] << ')';
// CHECK-NEXT:   }
// CHECK-NEXT:   dbgs() << " using " << Decoder << " : FAIL\n";
// CHECK-NEXT: }

// CHECK{LITERAL}: [[maybe_unused]] void reportInsnDecodeFailure(const char *Name,
// CHECK-NEXT:                          const char *Decoder) {
// CHECK-NEXT:   dbgs() << "OPC_Decode for " << Name << ": insn using " << Decoder
// CHECK-NEXT:          << " : FAIL\n";
// CHECK-NEXT: }
// CHECK-NEXT: #endif // NDEBUG

// CHECK-LABEL: static DecodeStatus decodeToMCInst(
// CHECK:       case 0:
// CHECK-NEXT:    tmp = fieldFromInstruction(insn, 0, 4);
// CHECK-NEXT:    if (!Check(S, DecodeRCRegisterClass(MI, tmp, Address, Decoder))) {
// CHECK-NEXT:      LLVM_DEBUG(reportOperandDecodeFailure("dst", "DecodeRCRegisterClass", {0, 4}));
// CHECK-NEXT:      return MCDisassembler::Fail;
// CHECK-NEXT:    }
// CHECK-NEXT:    return S;
