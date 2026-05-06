// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s

// Verify the DecoderEmitter emits the dbgFailOpd / dbgFailInsn helpers used
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

// dbgFailOpd (no-field overload): the description is a single shared literal.
// CHECK:      DecodeStatus dbgFailOpd(const char *Op, const char *Desc, const char *Decoder,
// CHECK-NEXT:                         bool *DecodeComplete = nullptr) {
// CHECK-NEXT:   LLVM_DEBUG(dbgs() << "OPC_Decode for " << Op << ": " << Desc << " using "
// CHECK-NEXT:                     << Decoder << " : FAIL\n");
// CHECK-NEXT:   if (DecodeComplete)
// CHECK-NEXT:     *DecodeComplete = false;
// CHECK-NEXT:   return MCDisassembler::Fail;
// CHECK-NEXT: }

// dbgFailOpd (field overload): forms "field(b,w)" from raw (base, width)
// integers rather than baking a unique string literal into each call site.
// CHECK:      DecodeStatus dbgFailOpd(const char *Op, const char *Decoder,
// CHECK-NEXT:                         ArrayRef<uint32_t> Fields,
// CHECK-NEXT:                         bool *DecodeComplete = nullptr) {
// CHECK-NEXT:   LLVM_DEBUG({
// CHECK-NEXT:     dbgs() << "OPC_Decode for " << Op << ": ";
// CHECK-NEXT:     for (unsigned I = 0, E = Fields.size(); I != E; I += 2) {
// CHECK-NEXT:       if (I)
// CHECK-NEXT:         dbgs() << " + ";
// CHECK-NEXT:       dbgs() << "field(" << Fields[I] << ',' << Fields[I + 1] << ')';
// CHECK-NEXT:     }
// CHECK-NEXT:     dbgs() << " using " << Decoder << " : FAIL\n";
// CHECK-NEXT:   });
// CHECK-NEXT:   if (DecodeComplete)
// CHECK-NEXT:     *DecodeComplete = false;
// CHECK-NEXT:   return MCDisassembler::Fail;
// CHECK-NEXT: }

// dbgFailInsn: whole-instruction custom-decoder check-site helper.
// CHECK{LITERAL}: [[maybe_unused]] DecodeStatus dbgFailInsn(const char *Name, const char *Decoder,
// CHECK-NEXT:                                           bool *DecodeComplete = nullptr) {
// CHECK-NEXT:   LLVM_DEBUG(dbgs() << "OPC_Decode for " << Name << ": insn using " << Decoder
// CHECK-NEXT:                     << " : FAIL\n");
// CHECK-NEXT:   if (DecodeComplete)
// CHECK-NEXT:     *DecodeComplete = false;
// CHECK-NEXT:   return MCDisassembler::Fail;
// CHECK-NEXT: }

// And the per-operand check site inside decodeToMCInst calls dbgFailOpd
// with the operand name, field description, and decoder-method name as
// separate string-literal arguments.
// CHECK-LABEL: static DecodeStatus decodeToMCInst(
// CHECK:       case 0:
// CHECK-NEXT:    tmp = fieldFromInstruction(insn, 0, 4);
// CHECK-NEXT:    if (!Check(S, DecodeRCRegisterClass(MI, tmp, Address, Decoder))) return dbgFailOpd("dst", "DecodeRCRegisterClass", {0, 4});
// CHECK-NEXT:    return S;
