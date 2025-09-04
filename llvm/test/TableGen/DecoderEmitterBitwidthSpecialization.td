// RUN: llvm-tblgen -gen-disassembler -I %p/../../include %s | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: llvm-tblgen -gen-disassembler -specialize-decoders-per-bitwidth -I %p/../../include %s | FileCheck %s --check-prefix=CHECK-SPECIALIZE-NO-TABLE
// RUN: llvm-tblgen -gen-disassembler -specialize-decoders-per-bitwidth -use-fn-table-in-decode-to-mcinst -I %p/../../include %s | FileCheck %s --check-prefix=CHECK-SPECIALIZE-TABLE


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

// Bit 0 of the encoding determines the size (8 or 16 bits).
// Bits {3..1} define the number of operands encoded.
class Instruction8Bit<int NumOps> : Instruction {
  let Size = 1;
  let OutOperandList = (outs);
  field bits<8> Inst;
  let Inst{0} = 0;
  let Inst{3-1} = NumOps;
}

class Instruction16Bit<int NumOps> : Instruction {
  let Size = 2;
  let OutOperandList = (outs);
  field bits<16> Inst;
  let Inst{0} = 1;
  let Inst{3-1} = NumOps;
}

// Define instructions to generate 4 cases in decodeToMCInst.
// Each register operand needs 2 bits to encode.

// An instruction with no inputs.
def Inst0 : Instruction8Bit<0> {
  let Inst{7-4} = 0;
  let InOperandList = (ins);
  let AsmString = "Inst0";
}

// An instruction with a single input.
def Inst1 : Instruction8Bit<1> {
  bits<2> r0;
  let Inst{5-4} = r0;
  let Inst{7-6} = 0;
  let InOperandList = (ins Regs:$r0);
  let AsmString = "Inst1";
}

// An instruction with two inputs.
def Inst2 : Instruction16Bit<2> {
  bits<2> r0;
  bits<2> r1;
  let Inst{5-4} = r0;
  let Inst{7-6} = r1;
  let Inst{15-8} = 0;
  let InOperandList = (ins Regs:$r0, Regs:$r1);
  let AsmString = "Inst2";
}

// An instruction with three inputs. .
def Inst3 : Instruction16Bit<3> {
  bits<2> r0;
  bits<2> r1;
  bits<2> r2;
  let Inst{5-4} = r0;
  let Inst{7-6} = r1;
  let Inst{9-8} = r2;
  let Inst{15-10} = 0;
  let InOperandList = (ins Regs:$r0, Regs:$r1, Regs:$r2);
  let AsmString = "Inst3";
}

// -----------------------------------------------------------------------------
// In the default case, we emit a single decodeToMCinst function and DecodeIdx
// is shared across all bitwidths.

// CHECK-DEFAULT-LABEL: DecoderTable8[25]
// CHECK-DEFAULT:         DecodeIdx: 0
// CHECK-DEFAULT:         DecodeIdx: 1
// CHECK-DEFAULT:       };

// CHECK-DEFAULT-LABEL: DecoderTable16[25]
// CHECK-DEFAULT:         DecodeIdx: 2
// CHECK-DEFAULT:         DecodeIdx: 3
// CHECK-DEFAULT:       };

// CHECK-DEFAULT-LABEL: template <typename InsnType>
// CHECK-DEFAULT-NEXT:  static DecodeStatus decodeToMCInst
// CHECK-DEFAULT:         case 0
// CHECK-DEFAULT:         case 1
// CHECK-DEFAULT:         case 2
// CHECK-DEFAULT:         case 3

// -----------------------------------------------------------------------------
// When we specialize per bitwidth, we emit 2 decodeToMCInst functions and
// DecodeIdx is assigned per bit width.

// CHECK-SPECIALIZE-NO-TABLE-LABEL:   DecoderTable8[26]
// CHECK-SPECIALIZE-NO-TABLE:           /* 0 */ 8, // Bitwidth 8
// CHECK-SPECIALIZE-NO-TABLE:           DecodeIdx: 0
// CHECK-SPECIALIZE-NO-TABLE:           DecodeIdx: 1
// CHECK-SPECIALIZE-NO-TABLE:         };

// CHECK-SPECIALIZE-NO-TABLE-LABEL:   template <typename InsnType>
// CHECK-SPECIALIZE-NO-TABLE-NEXT:    static std::enable_if_t<InsnBitWidth<InsnType> == 8, DecodeStatus>
// CHECK-SPECIALIZE-NO-TABLE-NEXT:    decodeToMCInst
// CHECK-SPECIALIZE-NO-TABLE:           case 0
// CHECK-SPECIALIZE-NO-TABLE:           case 1

// CHECK-SPECIALIZE-NO-TABLE-LABEL:   DecoderTable16[26]
// CHECK-SPECIALIZE-NO-TABLE:           /* 0 */ 16, // Bitwidth 16
// CHECK-SPECIALIZE-NO-TABLE:           DecodeIdx: 0
// CHECK-SPECIALIZE-NO-TABLE:           DecodeIdx: 1
// CHECK-SPECIALIZE-NO-TABLE:         };

// CHECK-SPECIALIZE-NO-TABLE-LABEL:   template <typename InsnType>
// CHECK-SPECIALIZE-NO-TABLE-NEXT:    static std::enable_if_t<InsnBitWidth<InsnType> == 16, DecodeStatus>
// CHECK-SPECIALIZE-NO-TABLE-NEXT:    decodeToMCInst
// CHECK-SPECIALIZE-NO-TABLE:           case 0
// CHECK-SPECIALIZE-NO-TABLE:           case 1

// CHECK-SPECIALIZE-NO-TABLE-LABEL:   template <typename InsnType>
// CHECK-SPECIALIZE-NO-TABLE-NEXT:    decodeInstruction
// CHECK-SPECIALIZE-NO-TABLE:         uint32_t BitWidth = decodeULEB128AndIncUnsafe(Ptr);
// CHECK-SPECIALIZE-NO-TABLE-NEXT:    assert(InsnBitWidth<InsnType> == BitWidth &&

// -----------------------------------------------------------------------------
// Per bitwidth specialization with function table.

// 8 bit deccoder table, functions, and function table.
// CHECK-SPECIALIZE-TABLE-LABEL:    DecoderTable8[26]
// CHECK-SPECIALIZE-TABLE:           /* 0 */ 8, // Bitwidth 8
// CHECK-SPECIALIZE-TABLE:            DecodeIdx: 0
// CHECK-SPECIALIZE-TABLE:            DecodeIdx: 1
// CHECK-SPECIALIZE-TABLE:          };

// CHECK-SPECIALIZE-TABLE-LABEL:    template <typename InsnType>
// CHECK-SPECIALIZE-TABLE-NEXT:     static std::enable_if_t<InsnBitWidth<InsnType> == 8, DecodeStatus>
// CHECK-SPECIALIZE-TABLE-NEXT:     decodeFn_8bit_0

// CHECK-SPECIALIZE-TABLE-LABEL:    template <typename InsnType>
// CHECK-SPECIALIZE-TABLE-NEXT:     static std::enable_if_t<InsnBitWidth<InsnType> == 8, DecodeStatus>
// CHECK-SPECIALIZE-TABLE-NEXT:     decodeFn_8bit_1

// CHECK-SPECIALIZE-TABLE-LABEL:    template <typename InsnType>
// CHECK-SPECIALIZE-TABLE-NEXT:     static std::enable_if_t<InsnBitWidth<InsnType> == 8, DecodeStatus>
// CHECK-SPECIALIZE-TABLE-NEXT:     decodeToMCInst
// CHECK-SPECIALIZE-TABLE-LABEL:      static constexpr DecodeFnTy decodeFnTable[] = {
// CHECK-SPECIALIZE-TABLE-NEXT:       decodeFn_8bit_0,
// CHECK-SPECIALIZE-TABLE-NEXT:       decodeFn_8bit_1,
// CHECK-SPECIALIZE-TABLE-NEXT:     };

// 16 bit deccoder table, functions, and function table.
// CHECK-SPECIALIZE-TABLE-LABEL:    DecoderTable16[26]
// CHECK-SPECIALIZE-TABLE:            /* 0 */ 16, // Bitwidth 16
// CHECK-SPECIALIZE-TABLE:            DecodeIdx: 0
// CHECK-SPECIALIZE-TABLE:            DecodeIdx: 1
// CHECK-SPECIALIZE-TABLE:          };

// CHECK-SPECIALIZE-TABLE-LABEL:    template <typename InsnType>
// CHECK-SPECIALIZE-TABLE-NEXT:     static std::enable_if_t<InsnBitWidth<InsnType> == 16, DecodeStatus>
// CHECK-SPECIALIZE-TABLE-NEXT:     decodeFn_16bit_0

// CHECK-SPECIALIZE-TABLE-LABEL:    template <typename InsnType>
// CHECK-SPECIALIZE-TABLE-NEXT:     static std::enable_if_t<InsnBitWidth<InsnType> == 16, DecodeStatus>
// CHECK-SPECIALIZE-TABLE-NEXT:     decodeFn_16bit_1

// CHECK-SPECIALIZE-TABLE-LABEL:    template <typename InsnType>
// CHECK-SPECIALIZE-TABLE-NEXT:     static std::enable_if_t<InsnBitWidth<InsnType> == 16, DecodeStatus>
// CHECK-SPECIALIZE-TABLE-NEXT:     decodeToMCInst
// CHECK-SPECIALIZE-TABLE-LABEL:      static constexpr DecodeFnTy decodeFnTable[] = {
// CHECK-SPECIALIZE-TABLE-NEXT:       decodeFn_16bit_0,
// CHECK-SPECIALIZE-TABLE-NEXT:       decodeFn_16bit_1,
// CHECK-SPECIALIZE-TABLE-NEXT:     };

// CHECK-SPECIALIZE-TABLE-LABEL:    template <typename InsnType>
// CHECK-SPECIALIZE-TABLE-NEXT:     decodeInstruction
// CHECK-SPECIALIZE-TABLE:          uint32_t BitWidth = decodeULEB128AndIncUnsafe(Ptr);
// CHECK-SPECIALIZE-TABLE-NEXT:     assert(InsnBitWidth<InsnType> == BitWidth &&
