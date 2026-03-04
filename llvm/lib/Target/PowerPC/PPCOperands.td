//===-- PPCOperands.td - PowerPC instruction operands -------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines PowerPC instruction operands, including immediate
// operands and addressing modes.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Immediate operand base classes
//===----------------------------------------------------------------------===//

// Base class for immediate AsmOperandClass definitions.
class ImmediateAsmOperand<string predicate, string render="addImmOperands">
: AsmOperandClass {
  let Name = NAME;
  let PredicateMethod = predicate;
  let RenderMethod = render;
}

// Base class for immediate operands with optional encoder.
class ImmediateOp<ValueType vt, string asmop, int width, bit is_signed = 0,
                  string encoder = "", string decoder = ""> : Operand<vt> {
  let PrintMethod = "print"#asmop#"Operand";
  let ParserMatchClass = !cast<AsmOperandClass>(asmop);
  let OperandType = "OPERAND_IMMEDIATE";

  // Set decoder method based on signedness if not explicitly provided
  let DecoderMethod = !if(!eq(decoder, ""),
                          !if(is_signed,
                              "decodeSImmOperand<"#width#">",
                              "decodeUImmOperand<"#width#">"),
                          decoder);

  // Set encoder method if provided
  let EncoderMethod = !if(!eq(encoder, ""),
                          "",
                          "getImmEncoding<" # encoder # ">");
}

//===----------------------------------------------------------------------===//
// Multiclasses for complete immediate definitions
// (AsmOperand + Operand + ImmLeaf).
//===----------------------------------------------------------------------===//

// Helper multiclass that generates operand + patterns together while keeping
// them as separate definitions for GlobalISel compatibility.
multiclass ImmOpWithPatterns<ValueType vt, string asmop, int width,
                             bit is_signed, code pred, SDNodeXForm xform,
                             string encoder = ""> {
  // Operand definition (for instruction operands)
  def "" : ImmediateOp<vt, asmop, width, is_signed, encoder>;

  // ImmLeaf for imm nodes (for DAG pattern matching)
  def _pat : ImmLeaf<vt, pred, xform>;

  // TImmLeaf for timm nodes (for target-specific pattern matching, e.g., intrinsics)
  def _timm : TImmLeaf<vt, pred, xform>;
}

// Creates Operand and separate ImmLeaf definitions for unsigned immediates.
multiclass UnsignedImmediate<ValueType vt, code pred, SDNodeXForm xform,
                             string asmop, int width, string encoder = "">
      : ImmOpWithPatterns<vt, asmop, width, 0, pred, xform, encoder>;

// Creates Operand and separate ImmLeaf definitions for signed immediates.
multiclass SignedImmediate<ValueType vt, code pred, SDNodeXForm xform,
                           string asmop, int width, string encoder = "">
      : ImmOpWithPatterns<vt, asmop, width, 1, pred, xform, encoder>;

// Multiclass for signed immediate operands with both regular and
// PC-relative versions.
multiclass SignedImmediateWithPCRel<ValueType vt, code pred,
                                    SDNodeXForm xform, string asmop,
                                    int width, string fixup_imm,
                                    string fixup_pcrel>
      : SignedImmediate<vt, pred, xform, asmop, width, fixup_imm> {
  // PC-relative immediate: instantiate SignedImmediate with PC-relative fixup
  defm _pcrel : SignedImmediate<vt, pred, xform, asmop, width, fixup_pcrel>;
}

//===----------------------------------------------------------------------===//
// Immediate AsmOperand definitions
//===----------------------------------------------------------------------===//
def U1Imm : ImmediateAsmOperand<"isUImm<1>">;
def U2Imm : ImmediateAsmOperand<"isUImm<2>">;
def U3Imm : ImmediateAsmOperand<"isUImm<3>">;
def U4Imm : ImmediateAsmOperand<"isUImm<4>">;
def U5Imm : ImmediateAsmOperand<"isUImm<5>">;
def U6Imm : ImmediateAsmOperand<"isUImm<6>">;
def U7Imm : ImmediateAsmOperand<"isUImm<7>">;
def U8Imm : ImmediateAsmOperand<"isUImm<8>">;
def U10Imm : ImmediateAsmOperand<"isUImm<10>">;
def U12Imm : ImmediateAsmOperand<"isUImm<12>">;
def S5Imm : ImmediateAsmOperand<"isSImm<5>">;

// Special cases that have custom predicat and/or render method.
def ATBitsAsHint : ImmediateAsmOperand<"isATBitsAsHint">; //Predicate always fails.
def ImmZero : ImmediateAsmOperand<"isImmZero">;
def U16Imm : ImmediateAsmOperand<"isU16Imm","addU16ImmOperands">;
def S16Imm : ImmediateAsmOperand<"isS16Imm","addS16ImmOperands">;
def S17Imm : ImmediateAsmOperand<"isS17Imm","addS16ImmOperands">;
def S32Imm : ImmediateAsmOperand<"isS32Imm">;
def S34Imm : ImmediateAsmOperand<"isS34Imm">;

//===----------------------------------------------------------------------===//
// i32 immediate operands
//===----------------------------------------------------------------------===//

defm u1imm : UnsignedImmediate<i32,
  [{ return isUInt<1>(Imm); }], NOOP_SDNodeXForm,
  "U1Imm", 1>;
defm u2imm : UnsignedImmediate<i32,
  [{ return isUInt<2>(Imm); }], NOOP_SDNodeXForm,
  "U2Imm", 2>;
defm u3imm : UnsignedImmediate<i32,
  [{ return isUInt<3>(Imm); }], NOOP_SDNodeXForm,
  "U3Imm", 3>;
defm u4imm : UnsignedImmediate<i32,
  [{ return isUInt<4>(Imm); }], NOOP_SDNodeXForm,
  "U4Imm", 4>;
defm u5imm : UnsignedImmediate<i32,
  [{ return isUInt<5>(Imm); }], NOOP_SDNodeXForm,
  "U5Imm", 5>;
defm u6imm : UnsignedImmediate<i32,
  [{ return isUInt<6>(Imm); }], NOOP_SDNodeXForm,
  "U6Imm", 6>;
defm u7imm : UnsignedImmediate<i32,
  [{ return isUInt<7>(Imm); }], NOOP_SDNodeXForm,
  "U7Imm", 7>;
defm u8imm : UnsignedImmediate<i32,
  [{ return isUInt<8>(Imm); }], NOOP_SDNodeXForm,
  "U8Imm", 8>;
defm u10imm : UnsignedImmediate<i32,
  [{ return isUInt<10>(Imm); }], NOOP_SDNodeXForm,
  "U10Imm", 10>;
defm u12imm : UnsignedImmediate<i32,
  [{ return isUInt<12>(Imm); }], NOOP_SDNodeXForm,
  "U12Imm", 12>;

defm s5imm : SignedImmediate<i32,
  [{ return isInt<5>(Imm); }], NOOP_SDNodeXForm,
  "S5Imm", 5>;

defm s32imm : SignedImmediateWithPCRel<i32,
  [{ return isInt<34>(Imm); }], NOOP_SDNodeXForm,
  "S34Imm", 34, "PPC::fixup_ppc_imm32", "PPC::fixup_ppc_pcrel32">;
defm s34imm : SignedImmediate<i32,
  [{ return isInt<34>(Imm); }], NOOP_SDNodeXForm,
  "S34Imm", 34, "PPC::fixup_ppc_imm34">;

//===----------------------------------------------------------------------===//
// i64 immediate operands
//===----------------------------------------------------------------------===//

defm s32imm64 : SignedImmediateWithPCRel<i64,
  [{ return isInt<32>(Imm); }], NOOP_SDNodeXForm,
  "S32Imm", 32, "PPC::fixup_ppc_imm32", "PPC::fixup_ppc_pcrel32">;
defm s34imm64 : SignedImmediateWithPCRel<i64,
  [{ return isInt<34>(Imm); }], NOOP_SDNodeXForm,
  "S34Imm", 34, "PPC::fixup_ppc_imm34", "PPC::fixup_ppc_pcrel34">;

//===----------------------------------------------------------------------===//
// Special case immediate operands
//===----------------------------------------------------------------------===//

// immZero represents a hardcoded zero register encoding, it is NOT used in DAG
// patterns. It is only used as an instruction operand for assembly/disassembly,
// specifically to represent a hardcoded zero register value in PC-relative
// addressing modes.
def immZero : Operand<i32> {
  let PrintMethod = "printImmZeroOperand";
  let ParserMatchClass = ImmZero;
  let DecoderMethod = "decodeImmZeroOperand";
  let OperandType = "OPERAND_IMMEDIATE";
}

// atimm is used to represent branch prediction hints, not a general immediate
// value. It's a 2-bit AT (Address Translation) field in PPC branch instructions
// and is an assembly-only operand that prints as `+` or `-` symbols, not
// numeric values.
def atimm : Operand<i32> {
  let PrintMethod = "printATBitsAsHint";
  let ParserMatchClass = ATBitsAsHint;
  let OperandType = "OPERAND_IMMEDIATE";
}

// Special cases: s16imm and u16imm have custom encoder methods.
defm s16imm : SignedImmediate<i32,
  [{ return isInt<16>(Imm); }], NOOP_SDNodeXForm,
  "S16Imm", 16, "PPC::fixup_ppc_half16">;
defm u16imm : UnsignedImmediate<i32,
  [{ return isUInt<16>(Imm); }], NOOP_SDNodeXForm,
  "U16Imm", 16, "PPC::fixup_ppc_half16">;

// s16imm64 uses imm64SExt16 pattern to match the operand type.
def s16imm64 : ImmediateOp<i64, "S16Imm", 16, 1, "PPC::fixup_ppc_half16">;

// u16imm64 uses two different patterns depending on the instruction context:
// * immZExt16 - For low 16-bit immediates
// * imm16ShiftedZExt - For high 16-bit immediates (shifted)
def u16imm64 : ImmediateOp<i64, "U16Imm", 16, 0, "PPC::fixup_ppc_half16">;

// Special case: s17imm uses S16Imm print method but accepts wider range.
// This operand type is used for addis/lis to allow the assembler parser
// to accept immediates in the range -65536..65535 for compatibility
// with the GNU assembler. The operand is treated as 16-bit otherwise.
def s17imm : ImmediateOp<i32, "S17Imm", 16, 1, "PPC::fixup_ppc_half16"> {
  let PrintMethod = "printS16ImmOperand";
}
def s17imm64 : ImmediateOp<i64, "S17Imm", 16, 1, "PPC::fixup_ppc_half16"> {
  let PrintMethod = "printS16ImmOperand";
}
