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

// CHECK{LITERAL}: [[maybe_unused]] void reportInsnDecodeFailure(const char *InstName,
// CHECK-NEXT:                          const char *Decoder) {
// CHECK-NEXT:   dbgs() << "OPC_Decode for " << InstName << ": insn using " << Decoder
// CHECK-NEXT:          << " : FAIL\n";
// CHECK-NEXT: }
// CHECK-NEXT: #endif // NDEBUG

// CHECK-LABEL: static DecodeStatus decodeToMCInst(
// CHECK:       case 0:
// CHECK-NEXT:    tmp = fieldFromInstruction(insn, 0, 4);
// CHECK-NEXT:    if (!Check(S, DecodeRCRegisterClass(MI, tmp, Address, Decoder))) {
// CHECK-NEXT{LITERAL}:      LLVM_DEBUG(reportOperandDecodeFailure("dst", "DecodeRCRegisterClass", {{0, 4}}));
// CHECK-NEXT:      return MCDisassembler::Fail;
// CHECK-NEXT:    }
// CHECK-NEXT:    return S;
