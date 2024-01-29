//===-- X86InstrUtils.td - X86 Instruction Utilities --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for simplifying the instruction definitions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Classes for setting the fields of X86Inst
//===----------------------------------------------------------------------===//

// Prefix byte classes which are used to indicate to the ad-hoc machine code
// emitter that various prefix bytes are required.
class OpSize16 { OperandSize OpSize = OpSize16; }
class OpSize32 { OperandSize OpSize = OpSize32; }
class AdSize16 { AddressSize AdSize = AdSize16; }
class AdSize32 { AddressSize AdSize = AdSize32; }
class AdSize64 { AddressSize AdSize = AdSize64; }
class REX_W  { bit hasREX_W = 1; }
class LOCK   { bit hasLockPrefix = 1; }
class REP    { bit hasREPPrefix = 1; }
class TB     { Map OpMap = TB; }
class T8     { Map OpMap = T8; }
class TA     { Map OpMap = TA; }
class XOP8   { Map OpMap = XOP8; Prefix OpPrefix = PS; }
class XOP9   { Map OpMap = XOP9; Prefix OpPrefix = PS; }
class XOPA   { Map OpMap = XOPA; Prefix OpPrefix = PS; }
class ThreeDNow { Map OpMap = ThreeDNow; }
class T_MAP4     { Map OpMap = T_MAP4; }
class T_MAP4PS : T_MAP4 { Prefix OpPrefix = PS; } // none
class T_MAP4PD : T_MAP4 { Prefix OpPrefix = PD; } // 0x66
class T_MAP4XS : T_MAP4 { Prefix OpPrefix = XS; } // 0xF3
class T_MAP4XD : T_MAP4 { Prefix OpPrefix = XD; } // 0xF2
class T_MAP5     { Map OpMap = T_MAP5; }
class T_MAP5PS : T_MAP5 { Prefix OpPrefix = PS; } // none
class T_MAP5PD : T_MAP5 { Prefix OpPrefix = PD; } // 0x66
class T_MAP5XS : T_MAP5 { Prefix OpPrefix = XS; } // 0xF3
class T_MAP5XD : T_MAP5 { Prefix OpPrefix = XD; } // 0xF2
class T_MAP6     { Map OpMap = T_MAP6; }
class T_MAP6PS : T_MAP6 { Prefix OpPrefix = PS; }
class T_MAP6PD : T_MAP6 { Prefix OpPrefix = PD; }
class T_MAP6XS : T_MAP6 { Prefix OpPrefix = XS; }
class T_MAP6XD : T_MAP6 { Prefix OpPrefix = XD; }
class T_MAP7     { Map OpMap = T_MAP7; }
class T_MAP7XS : T_MAP7 { Prefix OpPrefix = XS; } // 0xF3
class T_MAP7XD : T_MAP7 { Prefix OpPrefix = XD; } // 0xF2
class OBXS   { Prefix OpPrefix = XS; }
class PS   : TB { Prefix OpPrefix = PS; }
class PD   : TB { Prefix OpPrefix = PD; }
class XD   : TB { Prefix OpPrefix = XD; }
class XS   : TB { Prefix OpPrefix = XS; }
class T8PS : T8 { Prefix OpPrefix = PS; }
class T8PD : T8 { Prefix OpPrefix = PD; }
class T8XD : T8 { Prefix OpPrefix = XD; }
class T8XS : T8 { Prefix OpPrefix = XS; }
class TAPS : TA { Prefix OpPrefix = PS; }
class TAPD : TA { Prefix OpPrefix = PD; }
class TAXD : TA { Prefix OpPrefix = XD; }
class TAXS : TA { Prefix OpPrefix = XS; }
class VEX    { Encoding OpEnc = EncVEX; }
class WIG  { bit IgnoresW = 1; }
// Special version of REX_W that can be changed to VEX.W==0 for EVEX2VEX.
class VEX_W1X  { bit hasREX_W = 1; bit EVEX_W1_VEX_W0 = 1; }
class VEX_4V : VEX { bit hasVEX_4V = 1; }
class VEX_L  { bit hasVEX_L = 1; }
class VEX_LIG { bit ignoresVEX_L = 1; }
class EVEX   { Encoding OpEnc = EncEVEX; }
class EVEX_4V : EVEX { bit hasVEX_4V = 1; }
class EVEX_K { bit hasEVEX_K = 1; }
class EVEX_KZ : EVEX_K { bit hasEVEX_Z = 1; }
class EVEX_B { bit hasEVEX_B = 1; }
class EVEX_RC { bit hasEVEX_RC = 1; }
class EVEX_V512 { bit hasEVEX_L2 = 1; bit hasVEX_L = 0; }
class EVEX_V256 { bit hasEVEX_L2 = 0; bit hasVEX_L = 1; }
class EVEX_V128 { bit hasEVEX_L2 = 0; bit hasVEX_L = 0; }
class NOTRACK { bit hasNoTrackPrefix = 1; }
class SIMD_EXC { list<Register> Uses = [MXCSR]; bit mayRaiseFPException = 1; }
// Specify AVX512 8-bit compressed displacement encoding based on the vector
// element size in bits (8, 16, 32, 64) and the CDisp8 form.
class EVEX_CD8<int esize, CD8VForm form> {
  int CD8_EltSize = !srl(esize, 3);
  bits<3> CD8_Form = form.Value;
}
class EVEX_NoCD8 : EVEX { bits<7> CD8_Scale = 0; }
class XOP { Encoding OpEnc = EncXOP; }
class XOP_4V : XOP { bit hasVEX_4V = 1; }
class EVEX2VEXOverride<string VEXInstrName> {
  string EVEX2VEXOverride = VEXInstrName;
}
class AVX512BIi8Base : PD {
  Domain ExeDomain = SSEPackedInt;
  ImmType ImmT = Imm8;
}
class AVX512XSIi8Base : XS {
  Domain ExeDomain = SSEPackedInt;
  ImmType ImmT = Imm8;
}
class AVX512XDIi8Base : XD {
  Domain ExeDomain = SSEPackedInt;
  ImmType ImmT = Imm8;
}
class AVX512PSIi8Base : PS {
  Domain ExeDomain = SSEPackedSingle;
  ImmType ImmT = Imm8;
}
class AVX512PDIi8Base : PD {
  Domain ExeDomain = SSEPackedDouble;
  ImmType ImmT = Imm8;
}
class NotEVEX2VEXConvertible { bit notEVEX2VEXConvertible = 1; }
class ExplicitREX2Prefix { ExplicitOpPrefix explicitOpPrefix = ExplicitREX2; }
class ExplicitVEXPrefix { ExplicitOpPrefix explicitOpPrefix = ExplicitVEX; }
class ExplicitEVEXPrefix { ExplicitOpPrefix explicitOpPrefix = ExplicitEVEX; }

// SchedModel info for instruction that loads one value and gets the second
// (and possibly third) value from a register.
// This is used for instructions that put the memory operands before other
// uses.
class SchedLoadReg<X86FoldableSchedWrite Sched> : Sched<[Sched.Folded,
  // Memory operand.
  ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault,
  // Register reads (implicit or explicit).
  Sched.ReadAfterFold, Sched.ReadAfterFold]>;

//===----------------------------------------------------------------------===//
// X86 Type infomation definitions
//===----------------------------------------------------------------------===//

/// X86TypeInfo - This is a bunch of information that describes relevant X86
/// information about value types.  For example, it can tell you what the
/// register class and preferred load to use.
class X86TypeInfo<ValueType vt, string instrsuffix, RegisterClass regclass,
                  PatFrag loadnode, X86MemOperand memoperand, ImmType immkind,
                  Operand immoperand, SDPatternOperator immoperator,
                  SDPatternOperator immnosuoperator, Operand imm8operand,
                  SDPatternOperator imm8operator, SDPatternOperator imm8nosuoperator,
                  bit hasOddOpcode, OperandSize opSize,
                  bit hasREX_W> {
  /// VT - This is the value type itself.
  ValueType VT = vt;

  /// InstrSuffix - This is the suffix used on instructions with this type.  For
  /// example, i8 -> "b", i16 -> "w", i32 -> "l", i64 -> "q".
  string InstrSuffix = instrsuffix;

  /// RegClass - This is the register class associated with this type.  For
  /// example, i8 -> GR8, i16 -> GR16, i32 -> GR32, i64 -> GR64.
  RegisterClass RegClass = regclass;

  /// LoadNode - This is the load node associated with this type.  For
  /// example, i8 -> loadi8, i16 -> loadi16, i32 -> loadi32, i64 -> loadi64.
  PatFrag LoadNode = loadnode;

  /// MemOperand - This is the memory operand associated with this type.  For
  /// example, i8 -> i8mem, i16 -> i16mem, i32 -> i32mem, i64 -> i64mem.
  X86MemOperand MemOperand = memoperand;

  /// ImmEncoding - This is the encoding of an immediate of this type.  For
  /// example, i8 -> Imm8, i16 -> Imm16, i32 -> Imm32.  Note that i64 -> Imm32
  /// since the immediate fields of i64 instructions is a 32-bit sign extended
  /// value.
  ImmType ImmEncoding = immkind;

  /// ImmOperand - This is the operand kind of an immediate of this type.  For
  /// example, i8 -> i8imm, i16 -> i16imm, i32 -> i32imm.  Note that i64 ->
  /// i64i32imm since the immediate fields of i64 instructions is a 32-bit sign
  /// extended value.
  Operand ImmOperand = immoperand;

  /// ImmOperator - This is the operator that should be used to match an
  /// immediate of this kind in a pattern (e.g. imm, or i64immSExt32).
  SDPatternOperator ImmOperator = immoperator;

  SDPatternOperator ImmNoSuOperator = immnosuoperator;

  /// Imm8Operand - This is the operand kind to use for an imm8 of this type.
  /// For example, i8 -> <invalid>, i16 -> i16i8imm, i32 -> i32i8imm.  This is
  /// only used for instructions that have a sign-extended imm8 field form.
  Operand Imm8Operand = imm8operand;

  /// Imm8Operator - This is the operator that should be used to match an 8-bit
  /// sign extended immediate of this kind in a pattern (e.g. imm16immSExt8).
  SDPatternOperator Imm8Operator = imm8operator;

  SDPatternOperator Imm8NoSuOperator = imm8nosuoperator;

  /// HasOddOpcode - This bit is true if the instruction should have an odd (as
  /// opposed to even) opcode.  Operations on i8 are usually even, operations on
  /// other datatypes are odd.
  bit HasOddOpcode = hasOddOpcode;

  /// OpSize - Selects whether the instruction needs a 0x66 prefix based on
  /// 16-bit vs 32-bit mode. i8/i64 set this to OpSizeFixed. i16 sets this
  /// to Opsize16. i32 sets this to OpSize32.
  OperandSize OpSize = opSize;

  /// HasREX_W - This bit is set to true if the instruction should have
  /// the 0x40 REX prefix.  This is set for i64 types.
  bit HasREX_W = hasREX_W;
}

def invalid_node : SDNode<"<<invalid_node>>", SDTIntLeaf,[],"<<invalid_node>>">;

def Xi8  : X86TypeInfo<i8, "b", GR8, loadi8, i8mem, Imm8, i8imm,
                       imm_su, imm, i8imm, invalid_node, invalid_node,
                       0, OpSizeFixed, 0>;
def Xi16 : X86TypeInfo<i16, "w", GR16, loadi16, i16mem, Imm16, i16imm,
                       imm_su, imm, i16i8imm, i16immSExt8_su, i16immSExt8,
                       1, OpSize16, 0>;
def Xi32 : X86TypeInfo<i32, "l", GR32, loadi32, i32mem, Imm32, i32imm,
                       imm_su, imm, i32i8imm, i32immSExt8_su, i32immSExt8,
                       1, OpSize32, 0>;
def Xi64 : X86TypeInfo<i64, "q", GR64, loadi64, i64mem, Imm32S, i64i32imm,
                      i64immSExt32_su, i64immSExt32, i64i8imm, i64immSExt8_su,
                      i64immSExt8, 1, OpSizeFixed, 1>;

// Group template arguments that can be derived from the vector type (EltNum x
// EltVT).  These are things like the register class for the writemask, etc.
// The idea is to pass one of these as the template argument rather than the
// individual arguments.
// The template is also used for scalar types, in this case numelts is 1.
class X86VectorVTInfo<int numelts, ValueType eltvt, RegisterClass rc,
                      string suffix = ""> {
  RegisterClass RC = rc;
  ValueType EltVT = eltvt;
  int NumElts = numelts;

  // Corresponding mask register class.
  RegisterClass KRC = !cast<RegisterClass>("VK" # NumElts);

  // Corresponding mask register pair class.
  RegisterOperand KRPC = !if (!gt(NumElts, 16), ?,
                              !cast<RegisterOperand>("VK" # NumElts # "Pair"));

  // Corresponding write-mask register class.
  RegisterClass KRCWM = !cast<RegisterClass>("VK" # NumElts # "WM");

  // The mask VT.
  ValueType KVT = !cast<ValueType>("v" # NumElts # "i1");

  // Suffix used in the instruction mnemonic.
  string Suffix = suffix;

  // VTName is a string name for vector VT. For vector types it will be
  // v # NumElts # EltVT, so for vector of 8 elements of i32 it will be v8i32
  // It is a little bit complex for scalar types, where NumElts = 1.
  // In this case we build v4f32 or v2f64
  string VTName = "v" # !if (!eq (NumElts, 1),
                        !if (!eq (EltVT.Size, 16), 8,
                        !if (!eq (EltVT.Size, 32), 4,
                        !if (!eq (EltVT.Size, 64), 2, NumElts))), NumElts) # EltVT;

  // The vector VT.
  ValueType VT = !cast<ValueType>(VTName);

  string EltTypeName = !cast<string>(EltVT);
  // Size of the element type in bits, e.g. 32 for v16i32.
  string EltSizeName = !subst("i", "", !subst("f", "", !subst("b", "", EltTypeName)));
  int EltSize = EltVT.Size;

  // "i" for integer types and "f" for floating-point types
  string TypeVariantName = !subst("b", "", !subst(EltSizeName, "", EltTypeName));

  // Size of RC in bits, e.g. 512 for VR512.
  int Size = VT.Size;

  // The corresponding memory operand, e.g. i512mem for VR512.
  X86MemOperand MemOp = !cast<X86MemOperand>(TypeVariantName # Size # "mem");
  X86MemOperand ScalarMemOp = !cast<X86MemOperand>(!subst("b", "", EltTypeName) # "mem");
  // FP scalar memory operand for intrinsics - ssmem/sdmem.
  Operand IntScalarMemOp = !if (!eq (EltTypeName, "f16"), !cast<Operand>("shmem"),
                           !if (!eq (EltTypeName, "bf16"), !cast<Operand>("shmem"),
                           !if (!eq (EltTypeName, "f32"), !cast<Operand>("ssmem"),
                           !if (!eq (EltTypeName, "f64"), !cast<Operand>("sdmem"), ?))));

  // Load patterns
  PatFrag LdFrag = !cast<PatFrag>("load" # VTName);

  PatFrag AlignedLdFrag = !cast<PatFrag>("alignedload" # VTName);

  PatFrag ScalarLdFrag = !cast<PatFrag>("load" # !subst("b", "", EltTypeName));
  PatFrag BroadcastLdFrag = !cast<PatFrag>("X86VBroadcastld" # EltSizeName);

  PatFrags ScalarIntMemFrags = !if (!eq (EltTypeName, "f16"), !cast<PatFrags>("sse_load_f16"),
                               !if (!eq (EltTypeName, "bf16"), !cast<PatFrags>("sse_load_f16"),
                               !if (!eq (EltTypeName, "f32"), !cast<PatFrags>("sse_load_f32"),
                               !if (!eq (EltTypeName, "f64"), !cast<PatFrags>("sse_load_f64"), ?))));

  // The string to specify embedded broadcast in assembly.
  string BroadcastStr = "{1to" # NumElts # "}";

  // 8-bit compressed displacement tuple/subvector format.  This is only
  // defined for NumElts <= 8.
  CD8VForm CD8TupleForm = !if (!eq (!srl(NumElts, 4), 0),
                               !cast<CD8VForm>("CD8VT" # NumElts), ?);

  SubRegIndex SubRegIdx = !if (!eq (Size, 128), sub_xmm,
                          !if (!eq (Size, 256), sub_ymm, ?));

  Domain ExeDomain = !if (!eq (EltTypeName, "f32"), SSEPackedSingle,
                     !if (!eq (EltTypeName, "f64"), SSEPackedDouble,
                     !if (!eq (EltTypeName, "f16"), SSEPackedSingle, // FIXME?
                     !if (!eq (EltTypeName, "bf16"), SSEPackedSingle, // FIXME?
                     SSEPackedInt))));

  RegisterClass FRC = !if (!eq (EltTypeName, "f32"), FR32X,
                      !if (!eq (EltTypeName, "f16"), FR16X,
                      !if (!eq (EltTypeName, "bf16"), FR16X,
                      FR64X)));

  dag ImmAllZerosV = (VT immAllZerosV);

  string ZSuffix = !if (!eq (Size, 128), "Z128",
                   !if (!eq (Size, 256), "Z256", "Z"));
}

def v64i8_info  : X86VectorVTInfo<64,  i8, VR512, "b">;
def v32i16_info : X86VectorVTInfo<32, i16, VR512, "w">;
def v16i32_info : X86VectorVTInfo<16, i32, VR512, "d">;
def v8i64_info  : X86VectorVTInfo<8,  i64, VR512, "q">;
def v32f16_info : X86VectorVTInfo<32, f16, VR512, "ph">;
def v32bf16_info: X86VectorVTInfo<32, bf16, VR512, "pbf">;
def v16f32_info : X86VectorVTInfo<16, f32, VR512, "ps">;
def v8f64_info  : X86VectorVTInfo<8,  f64, VR512, "pd">;

// "x" in v32i8x_info means RC = VR256X
def v32i8x_info  : X86VectorVTInfo<32,  i8, VR256X, "b">;
def v16i16x_info : X86VectorVTInfo<16, i16, VR256X, "w">;
def v8i32x_info  : X86VectorVTInfo<8,  i32, VR256X, "d">;
def v4i64x_info  : X86VectorVTInfo<4,  i64, VR256X, "q">;
def v16f16x_info : X86VectorVTInfo<16, f16, VR256X, "ph">;
def v16bf16x_info: X86VectorVTInfo<16, bf16, VR256X, "pbf">;
def v8f32x_info  : X86VectorVTInfo<8,  f32, VR256X, "ps">;
def v4f64x_info  : X86VectorVTInfo<4,  f64, VR256X, "pd">;

def v16i8x_info  : X86VectorVTInfo<16,  i8, VR128X, "b">;
def v8i16x_info  : X86VectorVTInfo<8,  i16, VR128X, "w">;
def v4i32x_info  : X86VectorVTInfo<4,  i32, VR128X, "d">;
def v2i64x_info  : X86VectorVTInfo<2,  i64, VR128X, "q">;
def v8f16x_info  : X86VectorVTInfo<8,  f16, VR128X, "ph">;
def v8bf16x_info : X86VectorVTInfo<8,  bf16, VR128X, "pbf">;
def v4f32x_info  : X86VectorVTInfo<4,  f32, VR128X, "ps">;
def v2f64x_info  : X86VectorVTInfo<2,  f64, VR128X, "pd">;

// We map scalar types to the smallest (128-bit) vector type
// with the appropriate element type. This allows to use the same masking logic.
def i32x_info    : X86VectorVTInfo<1,  i32, GR32, "si">;
def i64x_info    : X86VectorVTInfo<1,  i64, GR64, "sq">;
def f16x_info    : X86VectorVTInfo<1,  f16, VR128X, "sh">;
def bf16x_info   : X86VectorVTInfo<1,  bf16, VR128X, "sbf">;
def f32x_info    : X86VectorVTInfo<1,  f32, VR128X, "ss">;
def f64x_info    : X86VectorVTInfo<1,  f64, VR128X, "sd">;

class AVX512VLVectorVTInfo<X86VectorVTInfo i512, X86VectorVTInfo i256,
                           X86VectorVTInfo i128> {
  X86VectorVTInfo info512 = i512;
  X86VectorVTInfo info256 = i256;
  X86VectorVTInfo info128 = i128;
}

def avx512vl_i8_info  : AVX512VLVectorVTInfo<v64i8_info, v32i8x_info,
                                             v16i8x_info>;
def avx512vl_i16_info : AVX512VLVectorVTInfo<v32i16_info, v16i16x_info,
                                             v8i16x_info>;
def avx512vl_i32_info : AVX512VLVectorVTInfo<v16i32_info, v8i32x_info,
                                             v4i32x_info>;
def avx512vl_i64_info : AVX512VLVectorVTInfo<v8i64_info, v4i64x_info,
                                             v2i64x_info>;
def avx512vl_f16_info : AVX512VLVectorVTInfo<v32f16_info, v16f16x_info,
                                             v8f16x_info>;
def avx512vl_bf16_info : AVX512VLVectorVTInfo<v32bf16_info, v16bf16x_info,
                                             v8bf16x_info>;
def avx512vl_f32_info : AVX512VLVectorVTInfo<v16f32_info, v8f32x_info,
                                             v4f32x_info>;
def avx512vl_f64_info : AVX512VLVectorVTInfo<v8f64_info, v4f64x_info,
                                             v2f64x_info>;

class X86KVectorVTInfo<RegisterClass _krc, RegisterClass _krcwm,
                       ValueType _vt> {
  RegisterClass KRC = _krc;
  RegisterClass KRCWM = _krcwm;
  ValueType KVT = _vt;
}

def v1i1_info : X86KVectorVTInfo<VK1, VK1WM, v1i1>;
def v2i1_info : X86KVectorVTInfo<VK2, VK2WM, v2i1>;
def v4i1_info : X86KVectorVTInfo<VK4, VK4WM, v4i1>;
def v8i1_info : X86KVectorVTInfo<VK8, VK8WM, v8i1>;
def v16i1_info : X86KVectorVTInfo<VK16, VK16WM, v16i1>;
def v32i1_info : X86KVectorVTInfo<VK32, VK32WM, v32i1>;
def v64i1_info : X86KVectorVTInfo<VK64, VK64WM, v64i1>;

// Subclasses of X86Inst
class PseudoI<dag oops, dag iops, list<dag> pattern>
  : X86Inst<0, Pseudo, NoImm, oops, iops, ""> {
  let Pattern = pattern;
}

class I<bits<8> o, Format f, dag outs, dag ins, string asm,
        list<dag> pattern, Domain d = GenericDomain>
  : X86Inst<o, f, NoImm, outs, ins, asm, d> {
  let Pattern = pattern;
}
class Ii8<bits<8> o, Format f, dag outs, dag ins, string asm,
          list<dag> pattern, Domain d = GenericDomain>
  : X86Inst<o, f, Imm8, outs, ins, asm, d> {
  let Pattern = pattern;
}
class Ii8Reg<bits<8> o, Format f, dag outs, dag ins, string asm,
             list<dag> pattern, Domain d = GenericDomain>
  : X86Inst<o, f, Imm8Reg, outs, ins, asm, d> {
  let Pattern = pattern;
}
class Ii8PCRel<bits<8> o, Format f, dag outs, dag ins, string asm,
               list<dag> pattern>
  : X86Inst<o, f, Imm8PCRel, outs, ins, asm> {
  let Pattern = pattern;
}
class Ii16<bits<8> o, Format f, dag outs, dag ins, string asm,
           list<dag> pattern>
  : X86Inst<o, f, Imm16, outs, ins, asm> {
  let Pattern = pattern;
}
class Ii32<bits<8> o, Format f, dag outs, dag ins, string asm,
           list<dag> pattern>
  : X86Inst<o, f, Imm32, outs, ins, asm> {
  let Pattern = pattern;
}
class Ii32S<bits<8> o, Format f, dag outs, dag ins, string asm,
            list<dag> pattern>
  : X86Inst<o, f, Imm32S, outs, ins, asm> {
  let Pattern = pattern;
}

class Ii64<bits<8> o, Format f, dag outs, dag ins, string asm,
           list<dag> pattern>
  : X86Inst<o, f, Imm64, outs, ins, asm> {
  let Pattern = pattern;
}

class Ii16PCRel<bits<8> o, Format f, dag outs, dag ins, string asm,
           list<dag> pattern>
           : X86Inst<o, f, Imm16PCRel, outs, ins, asm> {
  let Pattern = pattern;
}

class Ii32PCRel<bits<8> o, Format f, dag outs, dag ins, string asm,
           list<dag> pattern>
  : X86Inst<o, f, Imm32PCRel, outs, ins, asm> {
  let Pattern = pattern;
}

// FPStack Instruction Templates:
// FPI - Floating Point Instruction template.
class FPI<bits<8> o, Format F, dag outs, dag ins, string asm>
  : I<o, F, outs, ins, asm, []> {
  let Defs = [FPSW];
  let Predicates = [HasX87];
}

// FpI_ - Floating Point Pseudo Instruction template.
class FpI_<dag outs, dag ins, FPFormat fp, list<dag> pattern>
  : PseudoI<outs, ins, pattern> {
  let FPForm = fp;
  let Defs = [FPSW];
  let Predicates = [HasX87];
}

// Templates for instructions that use a 16- or 32-bit segmented address as
//  their only operand: lcall (FAR CALL) and ljmp (FAR JMP)
//
//   Iseg16 - 16-bit segment selector, 16-bit offset
//   Iseg32 - 16-bit segment selector, 32-bit offset

class Iseg16 <bits<8> o, Format f, dag outs, dag ins, string asm,
              list<dag> pattern>
      : X86Inst<o, f, Imm16, outs, ins, asm> {
  let Pattern = pattern;
}

class Iseg32 <bits<8> o, Format f, dag outs, dag ins, string asm,
              list<dag> pattern>
      : X86Inst<o, f, Imm32, outs, ins, asm> {
  let Pattern = pattern;
}

// SI - SSE 1 & 2 scalar instructions
class SI<bits<8> o, Format F, dag outs, dag ins, string asm,
         list<dag> pattern, Domain d = GenericDomain>
      : I<o, F, outs, ins, asm, pattern, d> {
  let Predicates = !if(!eq(OpEnc.Value, EncEVEX.Value), [HasAVX512],
                   !if(!eq(OpEnc.Value, EncVEX.Value), [UseAVX],
                   !if(!eq(OpPrefix.Value, XS.Value), [UseSSE1],
                   !if(!eq(OpPrefix.Value, XD.Value), [UseSSE2],
                   !if(!eq(OpPrefix.Value, PD.Value), [UseSSE2],
                   [UseSSE1])))));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(!eq(OpEnc.Value, EncEVEX.Value), !strconcat("v", asm),
                  !if(!eq(OpEnc.Value, EncVEX.Value), !strconcat("v", asm),
                  asm));
}

// SI - SSE 1 & 2 scalar intrinsics - vex form available on AVX512
class SI_Int<bits<8> o, Format F, dag outs, dag ins, string asm,
         list<dag> pattern, Domain d = GenericDomain>
      : I<o, F, outs, ins, asm, pattern, d> {
  let Predicates = !if(!eq(OpEnc.Value, EncEVEX.Value), [HasAVX512],
                   !if(!eq(OpEnc.Value, EncVEX.Value), [UseAVX],
                   !if(!eq(OpPrefix.Value, XS.Value), [UseSSE1],
                   !if(!eq(OpPrefix.Value, XD.Value), [UseSSE2],
                   !if(!eq(OpPrefix.Value, PD.Value), [UseSSE2],
                   [UseSSE1])))));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(!eq(OpEnc.Value, EncEVEX.Value), !strconcat("v", asm),
                  !if(!eq(OpEnc.Value, EncVEX.Value), !strconcat("v", asm),
                  asm));
}
// SIi8 - SSE 1 & 2 scalar instructions - vex form available on AVX512
class SIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern> {
  let Predicates = !if(!eq(OpEnc.Value, EncEVEX.Value), [HasAVX512],
                   !if(!eq(OpEnc.Value, EncVEX.Value), [HasAVX],
                   !if(!eq(OpPrefix.Value, XS.Value), [UseSSE1],
                   [UseSSE2])));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(!eq(OpEnc.Value, EncEVEX.Value), !strconcat("v", asm),
                  !if(!eq(OpEnc.Value, EncVEX.Value), !strconcat("v", asm),
                  asm));
}

// PI - SSE 1 & 2 packed instructions
class PI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern,
         Domain d>
      : I<o, F, outs, ins, asm, pattern, d> {
  let Predicates = !if(!eq(OpEnc.Value, EncEVEX.Value), [HasAVX512],
                   !if(!eq(OpEnc.Value, EncVEX.Value), [HasAVX],
                   !if(!eq(OpPrefix.Value, PD.Value), [UseSSE2],
                   [UseSSE1])));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(!eq(OpEnc.Value, EncEVEX.Value), !strconcat("v", asm),
                  !if(!eq(OpEnc.Value, EncVEX.Value), !strconcat("v", asm),
                  asm));
}

// MMXPI - SSE 1 & 2 packed instructions with MMX operands
class MMXPI<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pattern,
            Domain d>
      : I<o, F, outs, ins, asm, pattern, d> {
  let Predicates = !if(!eq(OpPrefix.Value, PD.Value), [HasMMX, HasSSE2],
                       [HasMMX, HasSSE1]);
}

// PIi8 - SSE 1 & 2 packed instructions with immediate
class PIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern, Domain d>
      : Ii8<o, F, outs, ins, asm, pattern, d> {
  let Predicates = !if(!eq(OpEnc.Value, EncEVEX.Value), [HasAVX512],
                   !if(!eq(OpEnc.Value, EncVEX.Value), [HasAVX],
                   !if(!eq(OpPrefix.Value, PD.Value), [UseSSE2],
                   [UseSSE1])));

  // AVX instructions have a 'v' prefix in the mnemonic
  let AsmString = !if(!eq(OpEnc.Value, EncEVEX.Value), !strconcat("v", asm),
                  !if(!eq(OpEnc.Value, EncVEX.Value), !strconcat("v", asm),
                  asm));
}

// SSE1 Instruction Templates:
//
//   SSI   - SSE1 instructions with XS prefix.
//   PSI   - SSE1 instructions with PS prefix.
//   PSIi8 - SSE1 instructions with ImmT == Imm8 and PS prefix.
//   VSSI  - SSE1 instructions with XS prefix in AVX form.
//   VPSI  - SSE1 instructions with PS prefix in AVX form, packed single.

class SSI<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, XS, Requires<[UseSSE1]>;
class SSIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[UseSSE1]>;
class PSI<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedSingle>, PS,
        Requires<[UseSSE1]>;
class PSIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedSingle>, PS,
        Requires<[UseSSE1]>;
class VSSI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern>, XS,
        Requires<[HasAVX]>;
class VPSI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern, SSEPackedSingle>, PS,
        Requires<[HasAVX]>;

// SSE2 Instruction Templates:
//
//   SDI    - SSE2 instructions with XD prefix.
//   SDIi8  - SSE2 instructions with ImmT == Imm8 and XD prefix.
//   S2SI   - SSE2 instructions with XS prefix.
//   SSDIi8 - SSE2 instructions with ImmT == Imm8 and XS prefix.
//   PDI    - SSE2 instructions with PD prefix, packed double domain.
//   PDIi8  - SSE2 instructions with ImmT == Imm8 and PD prefix.
//   VSDI   - SSE2 scalar instructions with XD prefix in AVX form.
//   VPDI   - SSE2 vector instructions with PD prefix in AVX form,
//                 packed double domain.
//   VS2I   - SSE2 scalar instructions with PD prefix in AVX form.
//   S2I    - SSE2 scalar instructions with PD prefix.
//   MMXSDIi8  - SSE2 instructions with ImmT == Imm8 and XD prefix as well as
//               MMX operands.
//   MMXSSDIi8 - SSE2 instructions with ImmT == Imm8 and XS prefix as well as
//               MMX operands.

class SDI<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, XD, Requires<[UseSSE2]>;
class SDIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XD, Requires<[UseSSE2]>;
class S2SI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, XS, Requires<[UseSSE2]>;
class S2SIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[UseSSE2]>;
class PDI<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedDouble>, PD,
        Requires<[UseSSE2]>;
class PDIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedDouble>, PD,
        Requires<[UseSSE2]>;
class VSDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern>, XD,
        Requires<[UseAVX]>;
class VS2SI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern>, XS,
        Requires<[HasAVX]>;
class VPDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern, SSEPackedDouble>,
        PD, Requires<[HasAVX]>;
class VS2I<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, !strconcat("v", asm), pattern>, PD,
        Requires<[UseAVX]>;
class S2I<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, PD, Requires<[UseSSE2]>;
class MMXSDIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
               list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XD, Requires<[HasMMX, HasSSE2]>;
class MMXS2SIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
                list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[HasMMX, HasSSE2]>;

// SSE3 Instruction Templates:
//
//   S3I   - SSE3 instructions with PD prefixes.
//   S3SI  - SSE3 instructions with XS prefix.
//   S3DI  - SSE3 instructions with XD prefix.

class S3SI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedSingle>, XS,
        Requires<[UseSSE3]>;
class S3DI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedDouble>, XD,
        Requires<[UseSSE3]>;
class S3I<bits<8> o, Format F, dag outs, dag ins, string asm,
          list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedDouble>, PD,
        Requires<[UseSSE3]>;


// SSSE3 Instruction Templates:
//
//   SS38I - SSSE3 instructions with T8 prefix.
//   SS3AI - SSSE3 instructions with TA prefix.
//   MMXSS38I - SSSE3 instructions with T8 prefix and MMX operands.
//   MMXSS3AI - SSSE3 instructions with TA prefix and MMX operands.
//
// Note: SSSE3 instructions have 64-bit and 128-bit versions. The 64-bit version
// uses the MMX registers. The 64-bit versions are grouped with the MMX
// classes. They need to be enabled even if AVX is enabled.

class SS38I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[UseSSSE3]>;
class SS3AI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[UseSSSE3]>;
class MMXSS38I<bits<8> o, Format F, dag outs, dag ins, string asm,
               list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PS,
        Requires<[HasMMX, HasSSSE3]>;
class MMXSS3AI<bits<8> o, Format F, dag outs, dag ins, string asm,
               list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPS,
        Requires<[HasMMX, HasSSSE3]>;

// SSE4.1 Instruction Templates:
//
//   SS48I - SSE 4.1 instructions with T8 prefix.
//   SS41AIi8 - SSE 4.1 instructions with TA prefix and ImmT == Imm8.
//
class SS48I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[UseSSE41]>;
class SS4AIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[UseSSE41]>;

// SSE4.2 Instruction Templates:
//
//   SS428I - SSE 4.2 instructions with T8 prefix.
class SS428I<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[UseSSE42]>;

//   SS42AI = SSE 4.2 instructions with TA prefix
class SS42AI<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[UseSSE42]>;

//   CRC32I - SSE 4.2 CRC32 instructions.
// NOTE: 'HasCRC32' is used as CRC32 instructions are GPR only and not directly
// controlled by the SSE42 flag.
class CRC32I<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, T8XD, Requires<[HasCRC32]>;

// AVX Instruction Templates:
//   Instructions introduced in AVX (no SSE equivalent forms)
//
//   AVX8I - AVX instructions with T8PD prefix.
//   AVXAIi8 - AVX instructions with TAPD prefix and ImmT = Imm8.
class AVX8I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[HasAVX]>;
class AVXAIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[HasAVX]>;

// AVX2 Instruction Templates:
//   Instructions introduced in AVX2 (no SSE equivalent forms)
//
//   AVX28I - AVX2 instructions with T8PD prefix.
//   AVX2AIi8 - AVX2 instructions with TAPD prefix and ImmT = Imm8.
class AVX28I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[HasAVX2]>;
class AVX2AIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[HasAVX2]>;


// AVX-512 Instruction Templates:
//   Instructions introduced in AVX-512 (no SSE equivalent forms)
//
//   AVX5128I - AVX-512 instructions with T8PD prefix.
//   AVX512AIi8 - AVX-512 instructions with TAPD prefix and ImmT = Imm8.
//   AVX512PDI  - AVX-512 instructions with PD, double packed.
//   AVX512PSI  - AVX-512 instructions with PS, single packed.
//   AVX512XS8I - AVX-512 instructions with T8 and XS prefixes.
//   AVX512XSI  - AVX-512 instructions with XS prefix, generic domain.
//   AVX512BI   - AVX-512 instructions with PD, int packed domain.
//   AVX512SI   - AVX-512 scalar instructions with PD prefix.

class AVX5128I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[HasAVX512]>;
class AVX5128IBase : T8PD {
  Domain ExeDomain = SSEPackedInt;
}
class AVX512XS8I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8XS,
        Requires<[HasAVX512]>;
class AVX512XSI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, XS,
        Requires<[HasAVX512]>;
class AVX512XDI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, XD,
        Requires<[HasAVX512]>;
class AVX512BI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, PD,
        Requires<[HasAVX512]>;
class AVX512BIBase : PD {
  Domain ExeDomain = SSEPackedInt;
}
class AVX512BIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, PD,
        Requires<[HasAVX512]>;
class AVX512AIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[HasAVX512]>;
class AVX512AIi8Base : TAPD {
  ImmType ImmT = Imm8;
}
class AVX512Ii8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>,
        Requires<[HasAVX512]>;
class AVX512PDI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedDouble>, PD,
        Requires<[HasAVX512]>;
class AVX512PSI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedSingle>, PS,
        Requires<[HasAVX512]>;
class AVX512PIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern, Domain d>
      : Ii8<o, F, outs, ins, asm, pattern, d>, Requires<[HasAVX512]>;
class AVX512PI<bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern, Domain d>
      : I<o, F, outs, ins, asm, pattern, d>, Requires<[HasAVX512]>;
class AVX512FMA3S<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern>
      : I<o, F, outs, ins, asm, pattern>, T8PD,
        EVEX_4V, Requires<[HasAVX512]>;

class AVX512<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern>
      : I<o, F, outs, ins, asm, pattern>, Requires<[HasAVX512]>;

// AES Instruction Templates:
//
// AES8I
// These use the same encoding as the SSE4.2 T8 and TA encodings.
class AES8I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag>pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedInt>, T8PD,
        Requires<[NoAVX, HasAES]>;

class AESAI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        Requires<[NoAVX, HasAES]>;

// PCLMUL Instruction Templates
class PCLMULIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
               list<dag>pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD;

// FMA3 Instruction Templates
class FMA3<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern>
      : I<o, F, outs, ins, asm, pattern>, T8PD,
        VEX_4V, FMASC, Requires<[HasFMA, NoFMA4, NoVLX]>;
class FMA3S<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag>pattern>
      : I<o, F, outs, ins, asm, pattern>, T8PD,
        VEX_4V, FMASC, Requires<[HasFMA, NoFMA4, NoAVX512]>;
class FMA3S_Int<bits<8> o, Format F, dag outs, dag ins, string asm,
                list<dag>pattern>
      : I<o, F, outs, ins, asm, pattern>, T8PD,
        VEX_4V, FMASC, Requires<[HasFMA, NoAVX512]>;

// FMA4 Instruction Templates
class FMA4<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern>
      : Ii8Reg<o, F, outs, ins, asm, pattern>, TAPD,
        VEX_4V, FMASC, Requires<[HasFMA4, NoVLX]>;
class FMA4S<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag>pattern>
      : Ii8Reg<o, F, outs, ins, asm, pattern>, TAPD,
        VEX_4V, FMASC, Requires<[HasFMA4, NoAVX512]>;
class FMA4S_Int<bits<8> o, Format F, dag outs, dag ins, string asm,
                list<dag>pattern>
      : Ii8Reg<o, F, outs, ins, asm, pattern>, TAPD,
        VEX_4V, FMASC, Requires<[HasFMA4]>;

// XOP 2, 3 and 4 Operand Instruction Template
class IXOP<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern, SSEPackedDouble>,
         XOP9, Requires<[HasXOP]>;

// XOP 2 and 3 Operand Instruction Templates with imm byte
class IXOPi8<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern, SSEPackedDouble>,
         XOP8, Requires<[HasXOP]>;
// XOP 4 Operand Instruction Templates with imm byte
class IXOPi8Reg<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : Ii8Reg<o, F, outs, ins, asm, pattern, SSEPackedDouble>,
         XOP8, Requires<[HasXOP]>;

//  XOP 5 operand instruction (VEX encoding!)
class IXOP5<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag>pattern>
      : Ii8Reg<o, F, outs, ins, asm, pattern, SSEPackedInt>, TAPD,
        VEX_4V, Requires<[HasXOP]>;

// X86-64 Instruction templates...
//

class RI<bits<8> o, Format F, dag outs, dag ins, string asm,
         list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, REX_W;
class RIi8 <bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, REX_W;
class RIi16 <bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii16<o, F, outs, ins, asm, pattern>, REX_W;
class RIi32 <bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : Ii32<o, F, outs, ins, asm, pattern>, REX_W;
class RIi32S <bits<8> o, Format F, dag outs, dag ins, string asm,
              list<dag> pattern>
      : Ii32S<o, F, outs, ins, asm, pattern>, REX_W;
class RIi64<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii64<o, F, outs, ins, asm, pattern>, REX_W;

class RS2I<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : S2I<o, F, outs, ins, asm, pattern>, REX_W;
class VRS2I<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : VS2I<o, F, outs, ins, asm, pattern>, REX_W;

// MMX Instruction templates
//

// MMXI   - MMX instructions with TB prefix.
// MMXI32 - MMX instructions with TB prefix valid only in 32 bit mode.
// MMXI64 - MMX instructions with TB prefix valid only in 64 bit mode.
// MMX2I  - MMX / SSE2 instructions with PD prefix.
// MMXIi8 - MMX instructions with ImmT == Imm8 and PS prefix.
// MMXIi8 - MMX instructions with ImmT == Imm8 and PS prefix.
// MMXID  - MMX instructions with XD prefix.
// MMXIS  - MMX instructions with XS prefix.
class MMXI<bits<8> o, Format F, dag outs, dag ins, string asm,
           list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, PS, Requires<[HasMMX]>;
class MMXI32<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, PS, Requires<[HasMMX,Not64BitMode]>;
class MMXI64<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, PS, Requires<[HasMMX,In64BitMode]>;
class MMXRI<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, PS, REX_W,
        Requires<[HasMMX,In64BitMode]>;
class MMX2I<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : I<o, F, outs, ins, asm, pattern>, PD, Requires<[HasMMX]>;
class MMXIi8<bits<8> o, Format F, dag outs, dag ins, string asm,
             list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, PS, Requires<[HasMMX]>;
class MMXID<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XD, Requires<[HasMMX]>;
class MMXIS<bits<8> o, Format F, dag outs, dag ins, string asm,
            list<dag> pattern>
      : Ii8<o, F, outs, ins, asm, pattern>, XS, Requires<[HasMMX]>;

/// ITy - This instruction base class takes the type info for the instruction.
/// Using this, it:
/// 1. Concatenates together the instruction mnemonic with the appropriate
///    suffix letter, a tab, and the arguments.
/// 2. Infers whether the instruction should have a 0x66 prefix byte.
/// 3. Infers whether the instruction should have a 0x40 REX_W prefix.
/// 4. Infers whether the low bit of the opcode should be 0 (for i8 operations)
///    or 1 (for i16,i32,i64 operations).
class ITy<bits<8> opcode, Format f, X86TypeInfo typeinfo, dag outs, dag ins,
          string mnemonic, string args, list<dag> pattern>
  : I<{opcode{7}, opcode{6}, opcode{5}, opcode{4},
       opcode{3}, opcode{2}, opcode{1}, typeinfo.HasOddOpcode },
      f, outs, ins,
      !strconcat(mnemonic, "{", typeinfo.InstrSuffix, "}\t", args), pattern> {

  // Infer instruction prefixes from type info.
  let OpSize = typeinfo.OpSize;
  let hasREX_W  = typeinfo.HasREX_W;
}
