//===-- X86InstrAVX512.td - AVX512 Instruction Set ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 AVX512 instruction set, defining the
// instructions, and properties of the instructions which are needed for code
// generation, machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

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

  // Corresponding write-mask register class.
  RegisterClass KRCWM = !cast<RegisterClass>("VK" # NumElts # "WM");

  // The mask VT.
  ValueType KVT = !cast<ValueType>(!if (!eq (NumElts, 1), "i1",
                                                          "v" # NumElts # "i1"));

  // Suffix used in the instruction mnemonic.
  string Suffix = suffix;

  // VTName is a string name for vector VT. For vector types it will be
  // v # NumElts # EltVT, so for vector of 8 elements of i32 it will be v8i32
  // It is a little bit complex for scalar types, where NumElts = 1.
  // In this case we build v4f32 or v2f64
  string VTName = "v" # !if (!eq (NumElts, 1),
                        !if (!eq (EltVT.Size, 32), 4,
                        !if (!eq (EltVT.Size, 64), 2, NumElts)), NumElts) # EltVT;

  // The vector VT.
  ValueType VT = !cast<ValueType>(VTName);

  string EltTypeName = !cast<string>(EltVT);
  // Size of the element type in bits, e.g. 32 for v16i32.
  string EltSizeName = !subst("i", "", !subst("f", "", EltTypeName));
  int EltSize = EltVT.Size;

  // "i" for integer types and "f" for floating-point types
  string TypeVariantName = !subst(EltSizeName, "", EltTypeName);

  // Size of RC in bits, e.g. 512 for VR512.
  int Size = VT.Size;

  // The corresponding memory operand, e.g. i512mem for VR512.
  X86MemOperand MemOp = !cast<X86MemOperand>(TypeVariantName # Size # "mem");
  X86MemOperand ScalarMemOp = !cast<X86MemOperand>(EltVT # "mem");
  // FP scalar memory operand for intrinsics - ssmem/sdmem.
  Operand IntScalarMemOp = !if (!eq (EltTypeName, "f32"), !cast<Operand>("ssmem"),
                           !if (!eq (EltTypeName, "f64"), !cast<Operand>("sdmem"), ?));

  // Load patterns
  // Note: For 128/256-bit integer VT we choose loadv2i64/loadv4i64
  //       due to load promotion during legalization
  PatFrag LdFrag = !cast<PatFrag>("load" #
                                  !if (!eq (TypeVariantName, "i"),
                                       !if (!eq (Size, 128), "v2i64",
                                       !if (!eq (Size, 256), "v4i64",
                                       !if (!eq (Size, 512), "v8i64",
                                            VTName))), VTName));

  PatFrag AlignedLdFrag = !cast<PatFrag>("alignedload" #
                                         !if (!eq (TypeVariantName, "i"),
                                               !if (!eq (Size, 128), "v2i64",
                                               !if (!eq (Size, 256), "v4i64",
                                               !if (!eq (Size, 512), "v8i64",
                                                   VTName))), VTName));

  PatFrag ScalarLdFrag = !cast<PatFrag>("load" # EltVT);

  ComplexPattern ScalarIntMemCPat = !if (!eq (EltTypeName, "f32"),
                                          !cast<ComplexPattern>("sse_load_f32"),
                                    !if (!eq (EltTypeName, "f64"),
                                          !cast<ComplexPattern>("sse_load_f64"),
                                    ?));

  // The corresponding float type, e.g. v16f32 for v16i32
  // Note: For EltSize < 32, FloatVT is illegal and TableGen
  //       fails to compile, so we choose FloatVT = VT
  ValueType FloatVT = !cast<ValueType>(
                        !if (!eq (!srl(EltSize,5),0),
                             VTName,
                             !if (!eq(TypeVariantName, "i"),
                                  "v" # NumElts # "f" # EltSize,
                                  VTName)));

  ValueType IntVT = !cast<ValueType>(
                        !if (!eq (!srl(EltSize,5),0),
                             VTName,
                             !if (!eq(TypeVariantName, "f"),
                                  "v" # NumElts # "i" # EltSize,
                                  VTName)));
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
                     SSEPackedInt));

  RegisterClass FRC = !if (!eq (EltTypeName, "f32"), FR32X, FR64X);

  // A vector tye of the same width with element type i64. This is used to
  // create patterns for logic ops.
  ValueType i64VT = !cast<ValueType>("v" # !srl(Size, 6) # "i64");

  // A vector type of the same width with element type i32.  This is used to
  // create the canonical constant zero node ImmAllZerosV.
  ValueType i32VT = !cast<ValueType>("v" # !srl(Size, 5) # "i32");
  dag ImmAllZerosV = (VT (bitconvert (i32VT immAllZerosV)));

  string ZSuffix = !if (!eq (Size, 128), "Z128",
                   !if (!eq (Size, 256), "Z256", "Z"));
}

def v64i8_info  : X86VectorVTInfo<64,  i8, VR512, "b">;
def v32i16_info : X86VectorVTInfo<32, i16, VR512, "w">;
def v16i32_info : X86VectorVTInfo<16, i32, VR512, "d">;
def v8i64_info  : X86VectorVTInfo<8,  i64, VR512, "q">;
def v16f32_info : X86VectorVTInfo<16, f32, VR512, "ps">;
def v8f64_info  : X86VectorVTInfo<8,  f64, VR512, "pd">;

// "x" in v32i8x_info means RC = VR256X
def v32i8x_info  : X86VectorVTInfo<32,  i8, VR256X, "b">;
def v16i16x_info : X86VectorVTInfo<16, i16, VR256X, "w">;
def v8i32x_info  : X86VectorVTInfo<8,  i32, VR256X, "d">;
def v4i64x_info  : X86VectorVTInfo<4,  i64, VR256X, "q">;
def v8f32x_info  : X86VectorVTInfo<8,  f32, VR256X, "ps">;
def v4f64x_info  : X86VectorVTInfo<4,  f64, VR256X, "pd">;

def v16i8x_info  : X86VectorVTInfo<16,  i8, VR128X, "b">;
def v8i16x_info  : X86VectorVTInfo<8,  i16, VR128X, "w">;
def v4i32x_info  : X86VectorVTInfo<4,  i32, VR128X, "d">;
def v2i64x_info  : X86VectorVTInfo<2,  i64, VR128X, "q">;
def v4f32x_info  : X86VectorVTInfo<4,  f32, VR128X, "ps">;
def v2f64x_info  : X86VectorVTInfo<2,  f64, VR128X, "pd">;

// We map scalar types to the smallest (128-bit) vector type
// with the appropriate element type. This allows to use the same masking logic.
def i32x_info    : X86VectorVTInfo<1,  i32, GR32, "si">;
def i64x_info    : X86VectorVTInfo<1,  i64, GR64, "sq">;
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
def avx512vl_f32_info : AVX512VLVectorVTInfo<v16f32_info, v8f32x_info,
                                             v4f32x_info>;
def avx512vl_f64_info : AVX512VLVectorVTInfo<v8f64_info, v4f64x_info,
                                             v2f64x_info>;

// This multiclass generates the masking variants from the non-masking
// variant.  It only provides the assembly pieces for the masking variants.
// It assumes custom ISel patterns for masking which can be provided as
// template arguments.
multiclass AVX512_maskable_custom<bits<8> O, Format F,
                                  dag Outs,
                                  dag Ins, dag MaskingIns, dag ZeroMaskingIns,
                                  string OpcodeStr,
                                  string AttSrcAsm, string IntelSrcAsm,
                                  list<dag> Pattern,
                                  list<dag> MaskingPattern,
                                  list<dag> ZeroMaskingPattern,
                                  string MaskingConstraint = "",
                                  InstrItinClass itin = NoItinerary,
                                  bit IsCommutable = 0,
                                  bit IsKCommutable = 0> {
  let isCommutable = IsCommutable in
    def NAME: AVX512<O, F, Outs, Ins,
                       OpcodeStr#"\t{"#AttSrcAsm#", $dst|"#
                                     "$dst, "#IntelSrcAsm#"}",
                       Pattern, itin>;

  // Prefer over VMOV*rrk Pat<>
  let isCommutable = IsKCommutable in
    def NAME#k: AVX512<O, F, Outs, MaskingIns,
                       OpcodeStr#"\t{"#AttSrcAsm#", $dst {${mask}}|"#
                                     "$dst {${mask}}, "#IntelSrcAsm#"}",
                       MaskingPattern, itin>,
              EVEX_K {
      // In case of the 3src subclass this is overridden with a let.
      string Constraints = MaskingConstraint;
    }

  // Zero mask does not add any restrictions to commute operands transformation.
  // So, it is Ok to use IsCommutable instead of IsKCommutable.
  let isCommutable = IsCommutable in // Prefer over VMOV*rrkz Pat<>
    def NAME#kz: AVX512<O, F, Outs, ZeroMaskingIns,
                       OpcodeStr#"\t{"#AttSrcAsm#", $dst {${mask}} {z}|"#
                                     "$dst {${mask}} {z}, "#IntelSrcAsm#"}",
                       ZeroMaskingPattern,
                       itin>,
              EVEX_KZ;
}


// Common base class of AVX512_maskable and AVX512_maskable_3src.
multiclass AVX512_maskable_common<bits<8> O, Format F, X86VectorVTInfo _,
                                  dag Outs,
                                  dag Ins, dag MaskingIns, dag ZeroMaskingIns,
                                  string OpcodeStr,
                                  string AttSrcAsm, string IntelSrcAsm,
                                  dag RHS, dag MaskingRHS,
                                  SDNode Select = vselect,
                                  string MaskingConstraint = "",
                                  InstrItinClass itin = NoItinerary,
                                  bit IsCommutable = 0,
                                  bit IsKCommutable = 0> :
  AVX512_maskable_custom<O, F, Outs, Ins, MaskingIns, ZeroMaskingIns, OpcodeStr,
                         AttSrcAsm, IntelSrcAsm,
                         [(set _.RC:$dst, RHS)],
                         [(set _.RC:$dst, MaskingRHS)],
                         [(set _.RC:$dst,
                               (Select _.KRCWM:$mask, RHS, _.ImmAllZerosV))],
                         MaskingConstraint, NoItinerary, IsCommutable,
                         IsKCommutable>;

// Similar to AVX512_maskable_common, but with scalar types.
multiclass AVX512_maskable_fp_common<bits<8> O, Format F, X86VectorVTInfo _,
                                  dag Outs,
                                  dag Ins, dag MaskingIns, dag ZeroMaskingIns,
                                  string OpcodeStr,
                                  string AttSrcAsm, string IntelSrcAsm,
                                  SDNode Select = vselect,
                                  string MaskingConstraint = "",
                                  InstrItinClass itin = NoItinerary,
                                  bit IsCommutable = 0,
                                  bit IsKCommutable = 0> :
  AVX512_maskable_custom<O, F, Outs, Ins, MaskingIns, ZeroMaskingIns, OpcodeStr,
                         AttSrcAsm, IntelSrcAsm,
                         [], [], [],
                         MaskingConstraint, NoItinerary, IsCommutable,
                         IsKCommutable>;

// This multiclass generates the unconditional/non-masking, the masking and
// the zero-masking variant of the vector instruction.  In the masking case, the
// perserved vector elements come from a new dummy input operand tied to $dst.
multiclass AVX512_maskable<bits<8> O, Format F, X86VectorVTInfo _,
                           dag Outs, dag Ins, string OpcodeStr,
                           string AttSrcAsm, string IntelSrcAsm,
                           dag RHS,
                           InstrItinClass itin = NoItinerary,
                           bit IsCommutable = 0, bit IsKCommutable = 0,
                           SDNode Select = vselect> :
   AVX512_maskable_common<O, F, _, Outs, Ins,
                          !con((ins _.RC:$src0, _.KRCWM:$mask), Ins),
                          !con((ins _.KRCWM:$mask), Ins),
                          OpcodeStr, AttSrcAsm, IntelSrcAsm, RHS,
                          (Select _.KRCWM:$mask, RHS, _.RC:$src0), Select,
                          "$src0 = $dst", itin, IsCommutable, IsKCommutable>;

// This multiclass generates the unconditional/non-masking, the masking and
// the zero-masking variant of the scalar instruction.
multiclass AVX512_maskable_scalar<bits<8> O, Format F, X86VectorVTInfo _,
                           dag Outs, dag Ins, string OpcodeStr,
                           string AttSrcAsm, string IntelSrcAsm,
                           dag RHS,
                           InstrItinClass itin = NoItinerary,
                           bit IsCommutable = 0> :
   AVX512_maskable_common<O, F, _, Outs, Ins,
                          !con((ins _.RC:$src0, _.KRCWM:$mask), Ins),
                          !con((ins _.KRCWM:$mask), Ins),
                          OpcodeStr, AttSrcAsm, IntelSrcAsm, RHS,
                          (X86selects _.KRCWM:$mask, RHS, _.RC:$src0),
                          X86selects, "$src0 = $dst", itin, IsCommutable>;

// Similar to AVX512_maskable but in this case one of the source operands
// ($src1) is already tied to $dst so we just use that for the preserved
// vector elements.  NOTE that the NonTiedIns (the ins dag) should exclude
// $src1.
multiclass AVX512_maskable_3src<bits<8> O, Format F, X86VectorVTInfo _,
                                dag Outs, dag NonTiedIns, string OpcodeStr,
                                string AttSrcAsm, string IntelSrcAsm,
                                dag RHS, bit IsCommutable = 0,
                                bit IsKCommutable = 0> :
   AVX512_maskable_common<O, F, _, Outs,
                          !con((ins _.RC:$src1), NonTiedIns),
                          !con((ins _.RC:$src1, _.KRCWM:$mask), NonTiedIns),
                          !con((ins _.RC:$src1, _.KRCWM:$mask), NonTiedIns),
                          OpcodeStr, AttSrcAsm, IntelSrcAsm, RHS,
                          (vselect _.KRCWM:$mask, RHS, _.RC:$src1),
                          vselect, "", NoItinerary, IsCommutable, IsKCommutable>;

multiclass AVX512_maskable_3src_scalar<bits<8> O, Format F, X86VectorVTInfo _,
                                     dag Outs, dag NonTiedIns, string OpcodeStr,
                                     string AttSrcAsm, string IntelSrcAsm,
                                     dag RHS, bit IsCommutable = 0,
                                     bit IsKCommutable = 0> :
   AVX512_maskable_common<O, F, _, Outs,
                          !con((ins _.RC:$src1), NonTiedIns),
                          !con((ins _.RC:$src1, _.KRCWM:$mask), NonTiedIns),
                          !con((ins _.RC:$src1, _.KRCWM:$mask), NonTiedIns),
                          OpcodeStr, AttSrcAsm, IntelSrcAsm, RHS,
                          (X86selects _.KRCWM:$mask, RHS, _.RC:$src1),
                          X86selects, "", NoItinerary, IsCommutable,
                          IsKCommutable>;

multiclass AVX512_maskable_in_asm<bits<8> O, Format F, X86VectorVTInfo _,
                                  dag Outs, dag Ins,
                                  string OpcodeStr,
                                  string AttSrcAsm, string IntelSrcAsm,
                                  list<dag> Pattern> :
   AVX512_maskable_custom<O, F, Outs, Ins,
                          !con((ins _.RC:$src0, _.KRCWM:$mask), Ins),
                          !con((ins _.KRCWM:$mask), Ins),
                          OpcodeStr, AttSrcAsm, IntelSrcAsm, Pattern, [], [],
                          "$src0 = $dst">;


// Instruction with mask that puts result in mask register,
// like "compare" and "vptest"
multiclass AVX512_maskable_custom_cmp<bits<8> O, Format F,
                                  dag Outs,
                                  dag Ins, dag MaskingIns,
                                  string OpcodeStr,
                                  string AttSrcAsm, string IntelSrcAsm,
                                  list<dag> Pattern,
                                  list<dag> MaskingPattern,
                                  bit IsCommutable = 0> {
    let isCommutable = IsCommutable in
    def NAME: AVX512<O, F, Outs, Ins,
                       OpcodeStr#"\t{"#AttSrcAsm#", $dst|"#
                                     "$dst, "#IntelSrcAsm#"}",
                       Pattern, NoItinerary>;

    def NAME#k: AVX512<O, F, Outs, MaskingIns,
                       OpcodeStr#"\t{"#AttSrcAsm#", $dst {${mask}}|"#
                                     "$dst {${mask}}, "#IntelSrcAsm#"}",
                       MaskingPattern, NoItinerary>, EVEX_K;
}

multiclass AVX512_maskable_common_cmp<bits<8> O, Format F, X86VectorVTInfo _,
                                  dag Outs,
                                  dag Ins, dag MaskingIns,
                                  string OpcodeStr,
                                  string AttSrcAsm, string IntelSrcAsm,
                                  dag RHS, dag MaskingRHS,
                                  bit IsCommutable = 0> :
  AVX512_maskable_custom_cmp<O, F, Outs, Ins, MaskingIns, OpcodeStr,
                         AttSrcAsm, IntelSrcAsm,
                         [(set _.KRC:$dst, RHS)],
                         [(set _.KRC:$dst, MaskingRHS)], IsCommutable>;

multiclass AVX512_maskable_cmp<bits<8> O, Format F, X86VectorVTInfo _,
                           dag Outs, dag Ins, string OpcodeStr,
                           string AttSrcAsm, string IntelSrcAsm,
                           dag RHS, bit IsCommutable = 0> :
   AVX512_maskable_common_cmp<O, F, _, Outs, Ins,
                          !con((ins _.KRCWM:$mask), Ins),
                          OpcodeStr, AttSrcAsm, IntelSrcAsm, RHS,
                          (and _.KRCWM:$mask, RHS), IsCommutable>;

multiclass AVX512_maskable_cmp_alt<bits<8> O, Format F, X86VectorVTInfo _,
                           dag Outs, dag Ins, string OpcodeStr,
                           string AttSrcAsm, string IntelSrcAsm> :
   AVX512_maskable_custom_cmp<O, F, Outs,
                             Ins, !con((ins _.KRCWM:$mask),Ins), OpcodeStr,
                             AttSrcAsm, IntelSrcAsm, [],[]>;

// This multiclass generates the unconditional/non-masking, the masking and
// the zero-masking variant of the vector instruction.  In the masking case, the
// perserved vector elements come from a new dummy input operand tied to $dst.
multiclass AVX512_maskable_logic<bits<8> O, Format F, X86VectorVTInfo _,
                           dag Outs, dag Ins, string OpcodeStr,
                           string AttSrcAsm, string IntelSrcAsm,
                           dag RHS, dag MaskedRHS,
                           InstrItinClass itin = NoItinerary,
                           bit IsCommutable = 0, SDNode Select = vselect> :
   AVX512_maskable_custom<O, F, Outs, Ins,
                          !con((ins _.RC:$src0, _.KRCWM:$mask), Ins),
                          !con((ins _.KRCWM:$mask), Ins),
                          OpcodeStr, AttSrcAsm, IntelSrcAsm,
                          [(set _.RC:$dst, RHS)],
                          [(set _.RC:$dst,
                                (Select _.KRCWM:$mask, MaskedRHS, _.RC:$src0))],
                          [(set _.RC:$dst,
                                (Select _.KRCWM:$mask, MaskedRHS,
                                        _.ImmAllZerosV))],
                          "$src0 = $dst", itin, IsCommutable>;

// Bitcasts between 512-bit vector types. Return the original type since
// no instruction is needed for the conversion.
def : Pat<(v8f64  (bitconvert (v8i64  VR512:$src))), (v8f64  VR512:$src)>;
def : Pat<(v8f64  (bitconvert (v16i32 VR512:$src))), (v8f64  VR512:$src)>;
def : Pat<(v8f64  (bitconvert (v32i16 VR512:$src))), (v8f64  VR512:$src)>;
def : Pat<(v8f64  (bitconvert (v64i8  VR512:$src))), (v8f64  VR512:$src)>;
def : Pat<(v8f64  (bitconvert (v16f32 VR512:$src))), (v8f64  VR512:$src)>;
def : Pat<(v16f32 (bitconvert (v8i64  VR512:$src))), (v16f32 VR512:$src)>;
def : Pat<(v16f32 (bitconvert (v16i32 VR512:$src))), (v16f32 VR512:$src)>;
def : Pat<(v16f32 (bitconvert (v32i16 VR512:$src))), (v16f32 VR512:$src)>;
def : Pat<(v16f32 (bitconvert (v64i8  VR512:$src))), (v16f32 VR512:$src)>;
def : Pat<(v16f32 (bitconvert (v8f64  VR512:$src))), (v16f32 VR512:$src)>;
def : Pat<(v8i64  (bitconvert (v16i32 VR512:$src))), (v8i64  VR512:$src)>;
def : Pat<(v8i64  (bitconvert (v32i16 VR512:$src))), (v8i64  VR512:$src)>;
def : Pat<(v8i64  (bitconvert (v64i8  VR512:$src))), (v8i64  VR512:$src)>;
def : Pat<(v8i64  (bitconvert (v8f64  VR512:$src))), (v8i64  VR512:$src)>;
def : Pat<(v8i64  (bitconvert (v16f32 VR512:$src))), (v8i64  VR512:$src)>;
def : Pat<(v16i32 (bitconvert (v8i64  VR512:$src))), (v16i32 VR512:$src)>;
def : Pat<(v16i32 (bitconvert (v16f32 VR512:$src))), (v16i32 VR512:$src)>;
def : Pat<(v16i32 (bitconvert (v32i16 VR512:$src))), (v16i32 VR512:$src)>;
def : Pat<(v16i32 (bitconvert (v64i8  VR512:$src))), (v16i32 VR512:$src)>;
def : Pat<(v16i32 (bitconvert (v8f64  VR512:$src))), (v16i32 VR512:$src)>;
def : Pat<(v32i16 (bitconvert (v8i64  VR512:$src))), (v32i16 VR512:$src)>;
def : Pat<(v32i16 (bitconvert (v16i32 VR512:$src))), (v32i16 VR512:$src)>;
def : Pat<(v32i16 (bitconvert (v64i8  VR512:$src))), (v32i16 VR512:$src)>;
def : Pat<(v32i16 (bitconvert (v8f64  VR512:$src))), (v32i16 VR512:$src)>;
def : Pat<(v32i16 (bitconvert (v16f32 VR512:$src))), (v32i16 VR512:$src)>;
def : Pat<(v32i16 (bitconvert (v16f32 VR512:$src))), (v32i16 VR512:$src)>;
def : Pat<(v64i8  (bitconvert (v8i64  VR512:$src))), (v64i8  VR512:$src)>;
def : Pat<(v64i8  (bitconvert (v16i32 VR512:$src))), (v64i8  VR512:$src)>;
def : Pat<(v64i8  (bitconvert (v32i16 VR512:$src))), (v64i8  VR512:$src)>;
def : Pat<(v64i8  (bitconvert (v8f64  VR512:$src))), (v64i8  VR512:$src)>;
def : Pat<(v64i8  (bitconvert (v16f32 VR512:$src))), (v64i8  VR512:$src)>;

// Alias instruction that maps zero vector to pxor / xorp* for AVX-512.
// This is expanded by ExpandPostRAPseudos to an xorps / vxorps, and then
// swizzled by ExecutionDepsFix to pxor.
// We set canFoldAsLoad because this can be converted to a constant-pool
// load of an all-zeros value if folding it would be beneficial.
let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1,
    isPseudo = 1, Predicates = [HasAVX512], SchedRW = [WriteZero] in {
def AVX512_512_SET0 : I<0, Pseudo, (outs VR512:$dst), (ins), "",
               [(set VR512:$dst, (v16i32 immAllZerosV))]>;
def AVX512_512_SETALLONES : I<0, Pseudo, (outs VR512:$dst), (ins), "",
               [(set VR512:$dst, (v16i32 immAllOnesV))]>;
}

// Alias instructions that allow VPTERNLOG to be used with a mask to create
// a mix of all ones and all zeros elements. This is done this way to force
// the same register to be used as input for all three sources.
let isPseudo = 1, Predicates = [HasAVX512] in {
def AVX512_512_SEXT_MASK_32 : I<0, Pseudo, (outs VR512:$dst),
                                (ins VK16WM:$mask), "",
                           [(set VR512:$dst, (vselect (v16i1 VK16WM:$mask),
                                                      (v16i32 immAllOnesV),
                                                      (v16i32 immAllZerosV)))]>;
def AVX512_512_SEXT_MASK_64 : I<0, Pseudo, (outs VR512:$dst),
                                (ins VK8WM:$mask), "",
                [(set VR512:$dst, (vselect (v8i1 VK8WM:$mask),
                                           (bc_v8i64 (v16i32 immAllOnesV)),
                                           (bc_v8i64 (v16i32 immAllZerosV))))]>;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1,
    isPseudo = 1, Predicates = [HasAVX512], SchedRW = [WriteZero] in {
def AVX512_128_SET0 : I<0, Pseudo, (outs VR128X:$dst), (ins), "",
               [(set VR128X:$dst, (v4i32 immAllZerosV))]>;
def AVX512_256_SET0 : I<0, Pseudo, (outs VR256X:$dst), (ins), "",
               [(set VR256X:$dst, (v8i32 immAllZerosV))]>;
}

// Alias instructions that map fld0 to xorps for sse or vxorps for avx.
// This is expanded by ExpandPostRAPseudos.
let isReMaterializable = 1, isAsCheapAsAMove = 1, canFoldAsLoad = 1,
    isPseudo = 1, SchedRW = [WriteZero], Predicates = [HasAVX512] in {
  def AVX512_FsFLD0SS : I<0, Pseudo, (outs FR32X:$dst), (ins), "",
                          [(set FR32X:$dst, fp32imm0)]>;
  def AVX512_FsFLD0SD : I<0, Pseudo, (outs FR64X:$dst), (ins), "",
                          [(set FR64X:$dst, fpimm0)]>;
}

//===----------------------------------------------------------------------===//
// AVX-512 - VECTOR INSERT
//
multiclass vinsert_for_size<int Opcode, X86VectorVTInfo From, X86VectorVTInfo To,
                                                       PatFrag vinsert_insert> {
  let ExeDomain = To.ExeDomain in {
    defm rr : AVX512_maskable<Opcode, MRMSrcReg, To, (outs To.RC:$dst),
                   (ins To.RC:$src1, From.RC:$src2, u8imm:$src3),
                   "vinsert" # From.EltTypeName # "x" # From.NumElts,
                   "$src3, $src2, $src1", "$src1, $src2, $src3",
                   (vinsert_insert:$src3 (To.VT To.RC:$src1),
                                         (From.VT From.RC:$src2),
                                         (iPTR imm))>, AVX512AIi8Base, EVEX_4V;

    defm rm : AVX512_maskable<Opcode, MRMSrcMem, To, (outs To.RC:$dst),
                   (ins To.RC:$src1, From.MemOp:$src2, u8imm:$src3),
                   "vinsert" # From.EltTypeName # "x" # From.NumElts,
                   "$src3, $src2, $src1", "$src1, $src2, $src3",
                   (vinsert_insert:$src3 (To.VT To.RC:$src1),
                               (From.VT (bitconvert (From.LdFrag addr:$src2))),
                               (iPTR imm))>, AVX512AIi8Base, EVEX_4V,
                   EVEX_CD8<From.EltSize, From.CD8TupleForm>;
  }
}

multiclass vinsert_for_size_lowering<string InstrStr, X86VectorVTInfo From,
                       X86VectorVTInfo To, PatFrag vinsert_insert,
                       SDNodeXForm INSERT_get_vinsert_imm , list<Predicate> p> {
  let Predicates = p in {
    def : Pat<(vinsert_insert:$ins
                     (To.VT To.RC:$src1), (From.VT From.RC:$src2), (iPTR imm)),
              (To.VT (!cast<Instruction>(InstrStr#"rr")
                     To.RC:$src1, From.RC:$src2,
                     (INSERT_get_vinsert_imm To.RC:$ins)))>;

    def : Pat<(vinsert_insert:$ins
                  (To.VT To.RC:$src1),
                  (From.VT (bitconvert (From.LdFrag addr:$src2))),
                  (iPTR imm)),
              (To.VT (!cast<Instruction>(InstrStr#"rm")
                  To.RC:$src1, addr:$src2,
                  (INSERT_get_vinsert_imm To.RC:$ins)))>;
  }
}

multiclass vinsert_for_type<ValueType EltVT32, int Opcode128,
                            ValueType EltVT64, int Opcode256> {

  let Predicates = [HasVLX] in
    defm NAME # "32x4Z256" : vinsert_for_size<Opcode128,
                                 X86VectorVTInfo< 4, EltVT32, VR128X>,
                                 X86VectorVTInfo< 8, EltVT32, VR256X>,
                                 vinsert128_insert>, EVEX_V256;

  defm NAME # "32x4Z" : vinsert_for_size<Opcode128,
                                 X86VectorVTInfo< 4, EltVT32, VR128X>,
                                 X86VectorVTInfo<16, EltVT32, VR512>,
                                 vinsert128_insert>, EVEX_V512;

  defm NAME # "64x4Z" : vinsert_for_size<Opcode256,
                                 X86VectorVTInfo< 4, EltVT64, VR256X>,
                                 X86VectorVTInfo< 8, EltVT64, VR512>,
                                 vinsert256_insert>, VEX_W, EVEX_V512;

  let Predicates = [HasVLX, HasDQI] in
    defm NAME # "64x2Z256" : vinsert_for_size<Opcode128,
                                   X86VectorVTInfo< 2, EltVT64, VR128X>,
                                   X86VectorVTInfo< 4, EltVT64, VR256X>,
                                   vinsert128_insert>, VEX_W, EVEX_V256;

  let Predicates = [HasDQI] in {
    defm NAME # "64x2Z" : vinsert_for_size<Opcode128,
                                 X86VectorVTInfo< 2, EltVT64, VR128X>,
                                 X86VectorVTInfo< 8, EltVT64, VR512>,
                                 vinsert128_insert>, VEX_W, EVEX_V512;

    defm NAME # "32x8Z" : vinsert_for_size<Opcode256,
                                   X86VectorVTInfo< 8, EltVT32, VR256X>,
                                   X86VectorVTInfo<16, EltVT32, VR512>,
                                   vinsert256_insert>, EVEX_V512;
  }
}

defm VINSERTF : vinsert_for_type<f32, 0x18, f64, 0x1a>;
defm VINSERTI : vinsert_for_type<i32, 0x38, i64, 0x3a>;

// Codegen pattern with the alternative types,
// Only add this if 64x2 and its friends are not supported natively via AVX512DQ.
defm : vinsert_for_size_lowering<"VINSERTF32x4Z256", v2f64x_info, v4f64x_info,
              vinsert128_insert, INSERT_get_vinsert128_imm, [HasVLX, NoDQI]>;
defm : vinsert_for_size_lowering<"VINSERTI32x4Z256", v2i64x_info, v4i64x_info,
              vinsert128_insert, INSERT_get_vinsert128_imm, [HasVLX, NoDQI]>;

defm : vinsert_for_size_lowering<"VINSERTF32x4Z", v2f64x_info, v8f64_info,
              vinsert128_insert, INSERT_get_vinsert128_imm, [HasAVX512, NoDQI]>;
defm : vinsert_for_size_lowering<"VINSERTI32x4Z", v2i64x_info, v8i64_info,
              vinsert128_insert, INSERT_get_vinsert128_imm, [HasAVX512, NoDQI]>;

defm : vinsert_for_size_lowering<"VINSERTF64x4Z", v8f32x_info, v16f32_info,
              vinsert256_insert, INSERT_get_vinsert256_imm, [HasAVX512, NoDQI]>;
defm : vinsert_for_size_lowering<"VINSERTI64x4Z", v8i32x_info, v16i32_info,
              vinsert256_insert, INSERT_get_vinsert256_imm, [HasAVX512, NoDQI]>;

// Codegen pattern with the alternative types insert VEC128 into VEC256
defm : vinsert_for_size_lowering<"VINSERTI32x4Z256", v8i16x_info, v16i16x_info,
              vinsert128_insert, INSERT_get_vinsert128_imm, [HasVLX]>;
defm : vinsert_for_size_lowering<"VINSERTI32x4Z256", v16i8x_info, v32i8x_info,
              vinsert128_insert, INSERT_get_vinsert128_imm, [HasVLX]>;
// Codegen pattern with the alternative types insert VEC128 into VEC512
defm : vinsert_for_size_lowering<"VINSERTI32x4Z", v8i16x_info, v32i16_info,
              vinsert128_insert, INSERT_get_vinsert128_imm, [HasAVX512]>;
defm : vinsert_for_size_lowering<"VINSERTI32x4Z", v16i8x_info, v64i8_info,
               vinsert128_insert, INSERT_get_vinsert128_imm, [HasAVX512]>;
// Codegen pattern with the alternative types insert VEC256 into VEC512
defm : vinsert_for_size_lowering<"VINSERTI64x4Z", v16i16x_info, v32i16_info,
              vinsert256_insert, INSERT_get_vinsert256_imm, [HasAVX512]>;
defm : vinsert_for_size_lowering<"VINSERTI64x4Z", v32i8x_info, v64i8_info,
              vinsert256_insert, INSERT_get_vinsert256_imm, [HasAVX512]>;

// vinsertps - insert f32 to XMM
let ExeDomain = SSEPackedSingle in {
def VINSERTPSZrr : AVX512AIi8<0x21, MRMSrcReg, (outs VR128X:$dst),
      (ins VR128X:$src1, VR128X:$src2, u8imm:$src3),
      "vinsertps\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
      [(set VR128X:$dst, (X86insertps VR128X:$src1, VR128X:$src2, imm:$src3))]>,
      EVEX_4V;
def VINSERTPSZrm: AVX512AIi8<0x21, MRMSrcMem, (outs VR128X:$dst),
      (ins VR128X:$src1, f32mem:$src2, u8imm:$src3),
      "vinsertps\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
      [(set VR128X:$dst, (X86insertps VR128X:$src1,
                          (v4f32 (scalar_to_vector (loadf32 addr:$src2))),
                          imm:$src3))]>, EVEX_4V, EVEX_CD8<32, CD8VT1>;
}

//===----------------------------------------------------------------------===//
// AVX-512 VECTOR EXTRACT
//---

multiclass vextract_for_size<int Opcode,
                             X86VectorVTInfo From, X86VectorVTInfo To,
                             PatFrag vextract_extract,
                             SDNodeXForm EXTRACT_get_vextract_imm> {

  let hasSideEffects = 0, ExeDomain = To.ExeDomain in {
    // use AVX512_maskable_in_asm (AVX512_maskable can't be used due to
    // vextract_extract), we interesting only in patterns without mask,
    // intrinsics pattern match generated bellow.
    defm rr : AVX512_maskable_in_asm<Opcode, MRMDestReg, To, (outs To.RC:$dst),
                (ins From.RC:$src1, u8imm:$idx),
                "vextract" # To.EltTypeName # "x" # To.NumElts,
                "$idx, $src1", "$src1, $idx",
                [(set To.RC:$dst, (vextract_extract:$idx (From.VT From.RC:$src1),
                                                         (iPTR imm)))]>,
              AVX512AIi8Base, EVEX;
    def mr  : AVX512AIi8<Opcode, MRMDestMem, (outs),
                    (ins To.MemOp:$dst, From.RC:$src1, u8imm:$idx),
                    "vextract" # To.EltTypeName # "x" # To.NumElts #
                        "\t{$idx, $src1, $dst|$dst, $src1, $idx}",
                    [(store (To.VT (vextract_extract:$idx
                                    (From.VT From.RC:$src1), (iPTR imm))),
                             addr:$dst)]>, EVEX;

    let mayStore = 1, hasSideEffects = 0 in
    def mrk : AVX512AIi8<Opcode, MRMDestMem, (outs),
                    (ins To.MemOp:$dst, To.KRCWM:$mask,
                                        From.RC:$src1, u8imm:$idx),
                     "vextract" # To.EltTypeName # "x" # To.NumElts #
                          "\t{$idx, $src1, $dst {${mask}}|"
                          "$dst {${mask}}, $src1, $idx}",
                    []>, EVEX_K, EVEX;
  }

  def : Pat<(To.VT (vselect To.KRCWM:$mask,
                            (vextract_extract:$ext (From.VT From.RC:$src1),
                                                   (iPTR imm)),
                            To.RC:$src0)),
            (!cast<Instruction>(NAME # To.EltSize # "x" # To.NumElts #
                                From.ZSuffix # "rrk")
                To.RC:$src0, To.KRCWM:$mask, From.RC:$src1,
                (EXTRACT_get_vextract_imm To.RC:$ext))>;

  def : Pat<(To.VT (vselect To.KRCWM:$mask,
                            (vextract_extract:$ext (From.VT From.RC:$src1),
                                                   (iPTR imm)),
                            To.ImmAllZerosV)),
            (!cast<Instruction>(NAME # To.EltSize # "x" # To.NumElts #
                                From.ZSuffix # "rrkz")
                To.KRCWM:$mask, From.RC:$src1,
                (EXTRACT_get_vextract_imm To.RC:$ext))>;
}

// Codegen pattern for the alternative types
multiclass vextract_for_size_lowering<string InstrStr, X86VectorVTInfo From,
                X86VectorVTInfo To, PatFrag vextract_extract,
                SDNodeXForm EXTRACT_get_vextract_imm, list<Predicate> p> {
  let Predicates = p in {
     def : Pat<(vextract_extract:$ext (From.VT From.RC:$src1), (iPTR imm)),
               (To.VT (!cast<Instruction>(InstrStr#"rr")
                          From.RC:$src1,
                          (EXTRACT_get_vextract_imm To.RC:$ext)))>;
     def : Pat<(store (To.VT (vextract_extract:$ext (From.VT From.RC:$src1),
                              (iPTR imm))), addr:$dst),
               (!cast<Instruction>(InstrStr#"mr") addr:$dst, From.RC:$src1,
                (EXTRACT_get_vextract_imm To.RC:$ext))>;
  }
}

multiclass vextract_for_type<ValueType EltVT32, int Opcode128,
                             ValueType EltVT64, int Opcode256> {
  defm NAME # "32x4Z" : vextract_for_size<Opcode128,
                                 X86VectorVTInfo<16, EltVT32, VR512>,
                                 X86VectorVTInfo< 4, EltVT32, VR128X>,
                                 vextract128_extract,
                                 EXTRACT_get_vextract128_imm>,
                                     EVEX_V512, EVEX_CD8<32, CD8VT4>;
  defm NAME # "64x4Z" : vextract_for_size<Opcode256,
                                 X86VectorVTInfo< 8, EltVT64, VR512>,
                                 X86VectorVTInfo< 4, EltVT64, VR256X>,
                                 vextract256_extract,
                                 EXTRACT_get_vextract256_imm>,
                                     VEX_W, EVEX_V512, EVEX_CD8<64, CD8VT4>;
  let Predicates = [HasVLX] in
    defm NAME # "32x4Z256" : vextract_for_size<Opcode128,
                                 X86VectorVTInfo< 8, EltVT32, VR256X>,
                                 X86VectorVTInfo< 4, EltVT32, VR128X>,
                                 vextract128_extract,
                                 EXTRACT_get_vextract128_imm>,
                                     EVEX_V256, EVEX_CD8<32, CD8VT4>;
  let Predicates = [HasVLX, HasDQI] in
    defm NAME # "64x2Z256" : vextract_for_size<Opcode128,
                                 X86VectorVTInfo< 4, EltVT64, VR256X>,
                                 X86VectorVTInfo< 2, EltVT64, VR128X>,
                                 vextract128_extract,
                                 EXTRACT_get_vextract128_imm>,
                                     VEX_W, EVEX_V256, EVEX_CD8<64, CD8VT2>;
  let Predicates = [HasDQI] in {
    defm NAME # "64x2Z" : vextract_for_size<Opcode128,
                                 X86VectorVTInfo< 8, EltVT64, VR512>,
                                 X86VectorVTInfo< 2, EltVT64, VR128X>,
                                 vextract128_extract,
                                 EXTRACT_get_vextract128_imm>,
                                     VEX_W, EVEX_V512, EVEX_CD8<64, CD8VT2>;
    defm NAME # "32x8Z" : vextract_for_size<Opcode256,
                                 X86VectorVTInfo<16, EltVT32, VR512>,
                                 X86VectorVTInfo< 8, EltVT32, VR256X>,
                                 vextract256_extract,
                                 EXTRACT_get_vextract256_imm>,
                                     EVEX_V512, EVEX_CD8<32, CD8VT8>;
  }
}

defm VEXTRACTF : vextract_for_type<f32, 0x19, f64, 0x1b>;
defm VEXTRACTI : vextract_for_type<i32, 0x39, i64, 0x3b>;

// extract_subvector codegen patterns with the alternative types.
// Only add this if 64x2 and its friends are not supported natively via AVX512DQ.
defm : vextract_for_size_lowering<"VEXTRACTF32x4Z", v8f64_info, v2f64x_info,
          vextract128_extract, EXTRACT_get_vextract128_imm, [HasAVX512, NoDQI]>;
defm : vextract_for_size_lowering<"VEXTRACTI32x4Z", v8i64_info, v2i64x_info,
          vextract128_extract, EXTRACT_get_vextract128_imm, [HasAVX512, NoDQI]>;

defm : vextract_for_size_lowering<"VEXTRACTF64x4Z", v16f32_info, v8f32x_info,
          vextract256_extract, EXTRACT_get_vextract256_imm, [HasAVX512, NoDQI]>;
defm : vextract_for_size_lowering<"VEXTRACTI64x4Z", v16i32_info, v8i32x_info,
          vextract256_extract, EXTRACT_get_vextract256_imm, [HasAVX512, NoDQI]>;

defm : vextract_for_size_lowering<"VEXTRACTF32x4Z256", v4f64x_info, v2f64x_info,
          vextract128_extract, EXTRACT_get_vextract128_imm, [HasVLX, NoDQI]>;
defm : vextract_for_size_lowering<"VEXTRACTI32x4Z256", v4i64x_info, v2i64x_info,
          vextract128_extract, EXTRACT_get_vextract128_imm, [HasVLX, NoDQI]>;

// Codegen pattern with the alternative types extract VEC128 from VEC256
defm : vextract_for_size_lowering<"VEXTRACTI32x4Z256", v16i16x_info, v8i16x_info,
          vextract128_extract, EXTRACT_get_vextract128_imm, [HasVLX]>;
defm : vextract_for_size_lowering<"VEXTRACTI32x4Z256", v32i8x_info, v16i8x_info,
          vextract128_extract, EXTRACT_get_vextract128_imm, [HasVLX]>;

// Codegen pattern with the alternative types extract VEC128 from VEC512
defm : vextract_for_size_lowering<"VEXTRACTI32x4Z", v32i16_info, v8i16x_info,
                 vextract128_extract, EXTRACT_get_vextract128_imm, [HasAVX512]>;
defm : vextract_for_size_lowering<"VEXTRACTI32x4Z", v64i8_info, v16i8x_info,
                 vextract128_extract, EXTRACT_get_vextract128_imm, [HasAVX512]>;
// Codegen pattern with the alternative types extract VEC256 from VEC512
defm : vextract_for_size_lowering<"VEXTRACTI64x4Z", v32i16_info, v16i16x_info,
                 vextract256_extract, EXTRACT_get_vextract256_imm, [HasAVX512]>;
defm : vextract_for_size_lowering<"VEXTRACTI64x4Z", v64i8_info, v32i8x_info,
                 vextract256_extract, EXTRACT_get_vextract256_imm, [HasAVX512]>;

// A 128-bit subvector extract from the first 256-bit vector position
// is a subregister copy that needs no instruction.
def : Pat<(v2i64 (extract_subvector (v8i64 VR512:$src), (iPTR 0))),
          (v2i64 (EXTRACT_SUBREG (v8i64 VR512:$src), sub_xmm))>;
def : Pat<(v2f64 (extract_subvector (v8f64 VR512:$src), (iPTR 0))),
          (v2f64 (EXTRACT_SUBREG (v8f64 VR512:$src), sub_xmm))>;
def : Pat<(v4i32 (extract_subvector (v16i32 VR512:$src), (iPTR 0))),
          (v4i32 (EXTRACT_SUBREG (v16i32 VR512:$src), sub_xmm))>;
def : Pat<(v4f32 (extract_subvector (v16f32 VR512:$src), (iPTR 0))),
          (v4f32 (EXTRACT_SUBREG (v16f32 VR512:$src), sub_xmm))>;
def : Pat<(v8i16 (extract_subvector (v32i16 VR512:$src), (iPTR 0))),
          (v8i16 (EXTRACT_SUBREG (v32i16 VR512:$src), sub_xmm))>;
def : Pat<(v16i8 (extract_subvector (v64i8 VR512:$src), (iPTR 0))),
          (v16i8 (EXTRACT_SUBREG (v64i8 VR512:$src), sub_xmm))>;

// A 256-bit subvector extract from the first 256-bit vector position
// is a subregister copy that needs no instruction.
def : Pat<(v4i64 (extract_subvector (v8i64 VR512:$src), (iPTR 0))),
          (v4i64 (EXTRACT_SUBREG (v8i64 VR512:$src), sub_ymm))>;
def : Pat<(v4f64 (extract_subvector (v8f64 VR512:$src), (iPTR 0))),
          (v4f64 (EXTRACT_SUBREG (v8f64 VR512:$src), sub_ymm))>;
def : Pat<(v8i32 (extract_subvector (v16i32 VR512:$src), (iPTR 0))),
          (v8i32 (EXTRACT_SUBREG (v16i32 VR512:$src), sub_ymm))>;
def : Pat<(v8f32 (extract_subvector (v16f32 VR512:$src), (iPTR 0))),
          (v8f32 (EXTRACT_SUBREG (v16f32 VR512:$src), sub_ymm))>;
def : Pat<(v16i16 (extract_subvector (v32i16 VR512:$src), (iPTR 0))),
          (v16i16 (EXTRACT_SUBREG (v32i16 VR512:$src), sub_ymm))>;
def : Pat<(v32i8 (extract_subvector (v64i8 VR512:$src), (iPTR 0))),
          (v32i8 (EXTRACT_SUBREG (v64i8 VR512:$src), sub_ymm))>;

let AddedComplexity = 25 in { // to give priority over vinsertf128rm
// A 128-bit subvector insert to the first 512-bit vector position
// is a subregister copy that needs no instruction.
def : Pat<(v8i64 (insert_subvector undef, (v2i64 VR128X:$src), (iPTR 0))),
          (INSERT_SUBREG (v8i64 (IMPLICIT_DEF)), VR128X:$src, sub_xmm)>;
def : Pat<(v8f64 (insert_subvector undef, (v2f64 VR128X:$src), (iPTR 0))),
          (INSERT_SUBREG (v8f64 (IMPLICIT_DEF)), VR128X:$src, sub_xmm)>;
def : Pat<(v16i32 (insert_subvector undef, (v4i32 VR128X:$src), (iPTR 0))),
          (INSERT_SUBREG (v16i32 (IMPLICIT_DEF)), VR128X:$src, sub_xmm)>;
def : Pat<(v16f32 (insert_subvector undef, (v4f32 VR128X:$src), (iPTR 0))),
          (INSERT_SUBREG (v16f32 (IMPLICIT_DEF)), VR128X:$src, sub_xmm)>;
def : Pat<(v32i16 (insert_subvector undef, (v8i16 VR128X:$src), (iPTR 0))),
          (INSERT_SUBREG (v32i16 (IMPLICIT_DEF)), VR128X:$src, sub_xmm)>;
def : Pat<(v64i8 (insert_subvector undef, (v16i8 VR128X:$src), (iPTR 0))),
          (INSERT_SUBREG (v64i8 (IMPLICIT_DEF)), VR128X:$src, sub_xmm)>;

// A 256-bit subvector insert to the first 512-bit vector position
// is a subregister copy that needs no instruction.
def : Pat<(v8i64 (insert_subvector undef, (v4i64 VR256X:$src), (iPTR 0))),
          (INSERT_SUBREG (v8i64 (IMPLICIT_DEF)), VR256X:$src, sub_ymm)>;
def : Pat<(v8f64 (insert_subvector undef, (v4f64 VR256X:$src), (iPTR 0))),
          (INSERT_SUBREG (v8f64 (IMPLICIT_DEF)), VR256X:$src, sub_ymm)>;
def : Pat<(v16i32 (insert_subvector undef, (v8i32 VR256X:$src), (iPTR 0))),
          (INSERT_SUBREG (v16i32 (IMPLICIT_DEF)), VR256X:$src, sub_ymm)>;
def : Pat<(v16f32 (insert_subvector undef, (v8f32 VR256X:$src), (iPTR 0))),
          (INSERT_SUBREG (v16f32 (IMPLICIT_DEF)), VR256X:$src, sub_ymm)>;
def : Pat<(v32i16 (insert_subvector undef, (v16i16 VR256X:$src), (iPTR 0))),
          (INSERT_SUBREG (v32i16 (IMPLICIT_DEF)), VR256X:$src, sub_ymm)>;
def : Pat<(v64i8 (insert_subvector undef, (v32i8 VR256X:$src), (iPTR 0))),
          (INSERT_SUBREG (v64i8 (IMPLICIT_DEF)), VR256X:$src, sub_ymm)>;
}

// vextractps - extract 32 bits from XMM
def VEXTRACTPSZrr : AVX512AIi8<0x17, MRMDestReg, (outs GR32:$dst),
      (ins VR128X:$src1, u8imm:$src2),
      "vextractps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      [(set GR32:$dst, (extractelt (bc_v4i32 (v4f32 VR128X:$src1)), imm:$src2))]>,
      EVEX;

def VEXTRACTPSZmr : AVX512AIi8<0x17, MRMDestMem, (outs),
      (ins f32mem:$dst, VR128X:$src1, u8imm:$src2),
      "vextractps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
      [(store (extractelt (bc_v4i32 (v4f32 VR128X:$src1)), imm:$src2),
                          addr:$dst)]>, EVEX, EVEX_CD8<32, CD8VT1>;

//===---------------------------------------------------------------------===//
// AVX-512 BROADCAST
//---
// broadcast with a scalar argument.
multiclass avx512_broadcast_scalar<bits<8> opc, string OpcodeStr,
                            X86VectorVTInfo DestInfo, X86VectorVTInfo SrcInfo> {
  def : Pat<(DestInfo.VT (X86VBroadcast SrcInfo.FRC:$src)),
            (!cast<Instruction>(NAME#DestInfo.ZSuffix#r)
             (COPY_TO_REGCLASS SrcInfo.FRC:$src, SrcInfo.RC))>;
  def : Pat<(DestInfo.VT (vselect DestInfo.KRCWM:$mask,
                                  (X86VBroadcast SrcInfo.FRC:$src),
                                  DestInfo.RC:$src0)),
            (!cast<Instruction>(NAME#DestInfo.ZSuffix#rk)
             DestInfo.RC:$src0, DestInfo.KRCWM:$mask,
             (COPY_TO_REGCLASS SrcInfo.FRC:$src, SrcInfo.RC))>;
  def : Pat<(DestInfo.VT (vselect DestInfo.KRCWM:$mask,
                                  (X86VBroadcast SrcInfo.FRC:$src),
                                  DestInfo.ImmAllZerosV)),
            (!cast<Instruction>(NAME#DestInfo.ZSuffix#rkz)
             DestInfo.KRCWM:$mask, (COPY_TO_REGCLASS SrcInfo.FRC:$src, SrcInfo.RC))>;
}

multiclass avx512_broadcast_rm<bits<8> opc, string OpcodeStr,
                            X86VectorVTInfo DestInfo, X86VectorVTInfo SrcInfo> {
  let ExeDomain = DestInfo.ExeDomain in {
  defm r : AVX512_maskable<opc, MRMSrcReg, DestInfo, (outs DestInfo.RC:$dst),
                   (ins SrcInfo.RC:$src), OpcodeStr, "$src", "$src",
                   (DestInfo.VT (X86VBroadcast (SrcInfo.VT SrcInfo.RC:$src)))>,
                   T8PD, EVEX;
  defm m : AVX512_maskable<opc, MRMSrcMem, DestInfo, (outs DestInfo.RC:$dst),
                   (ins SrcInfo.ScalarMemOp:$src), OpcodeStr, "$src", "$src",
                   (DestInfo.VT (X86VBroadcast
                                   (SrcInfo.ScalarLdFrag addr:$src)))>,
                   T8PD, EVEX, EVEX_CD8<SrcInfo.EltSize, CD8VT1>;
  }

  def : Pat<(DestInfo.VT (X86VBroadcast
                          (SrcInfo.VT (scalar_to_vector
                                       (SrcInfo.ScalarLdFrag addr:$src))))),
            (!cast<Instruction>(NAME#DestInfo.ZSuffix#m) addr:$src)>;
  def : Pat<(DestInfo.VT (vselect DestInfo.KRCWM:$mask,
                          (X86VBroadcast
                           (SrcInfo.VT (scalar_to_vector
                                        (SrcInfo.ScalarLdFrag addr:$src)))),
                          DestInfo.RC:$src0)),
            (!cast<Instruction>(NAME#DestInfo.ZSuffix#mk)
             DestInfo.RC:$src0, DestInfo.KRCWM:$mask, addr:$src)>;
  def : Pat<(DestInfo.VT (vselect DestInfo.KRCWM:$mask,
                          (X86VBroadcast
                           (SrcInfo.VT (scalar_to_vector
                                        (SrcInfo.ScalarLdFrag addr:$src)))),
                          DestInfo.ImmAllZerosV)),
            (!cast<Instruction>(NAME#DestInfo.ZSuffix#mkz)
             DestInfo.KRCWM:$mask, addr:$src)>;
}

multiclass avx512_fp_broadcast_sd<bits<8> opc, string OpcodeStr,
                                                       AVX512VLVectorVTInfo _> {
  let Predicates = [HasAVX512] in
    defm Z  : avx512_broadcast_rm<opc, OpcodeStr, _.info512, _.info128>,
              avx512_broadcast_scalar<opc, OpcodeStr, _.info512, _.info128>,
                               EVEX_V512;

  let Predicates = [HasVLX] in {
    defm Z256  : avx512_broadcast_rm<opc, OpcodeStr, _.info256, _.info128>,
                 avx512_broadcast_scalar<opc, OpcodeStr, _.info256, _.info128>,
                             EVEX_V256;
  }
}

multiclass avx512_fp_broadcast_ss<bits<8> opc, string OpcodeStr,
                                                       AVX512VLVectorVTInfo _> {
  let Predicates = [HasAVX512] in
    defm Z  : avx512_broadcast_rm<opc, OpcodeStr, _.info512, _.info128>,
              avx512_broadcast_scalar<opc, OpcodeStr, _.info512, _.info128>,
                               EVEX_V512;

  let Predicates = [HasVLX] in {
    defm Z256  : avx512_broadcast_rm<opc, OpcodeStr, _.info256, _.info128>,
                 avx512_broadcast_scalar<opc, OpcodeStr, _.info256, _.info128>,
                             EVEX_V256;
    defm Z128  : avx512_broadcast_rm<opc, OpcodeStr, _.info128, _.info128>,
                 avx512_broadcast_scalar<opc, OpcodeStr, _.info128, _.info128>,
                             EVEX_V128;
  }
}
defm VBROADCASTSS  : avx512_fp_broadcast_ss<0x18, "vbroadcastss",
                                       avx512vl_f32_info>;
defm VBROADCASTSD  : avx512_fp_broadcast_sd<0x19, "vbroadcastsd",
                                       avx512vl_f64_info>, VEX_W;

def : Pat<(int_x86_avx512_vbroadcast_ss_512 addr:$src),
          (VBROADCASTSSZm addr:$src)>;
def : Pat<(int_x86_avx512_vbroadcast_sd_512 addr:$src),
          (VBROADCASTSDZm addr:$src)>;

multiclass avx512_int_broadcast_reg<bits<8> opc, X86VectorVTInfo _,
                                    SDPatternOperator OpNode,
                                    RegisterClass SrcRC> {
  let ExeDomain = _.ExeDomain in
  defm r : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins SrcRC:$src),
                         "vpbroadcast"##_.Suffix, "$src", "$src",
                         (_.VT (OpNode SrcRC:$src))>, T8PD, EVEX;
}

multiclass avx512_int_broadcast_reg_vl<bits<8> opc, AVX512VLVectorVTInfo _,
                                       SDPatternOperator OpNode,
                                       RegisterClass SrcRC, Predicate prd> {
  let Predicates = [prd] in
    defm Z : avx512_int_broadcast_reg<opc, _.info512, OpNode, SrcRC>, EVEX_V512;
  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_int_broadcast_reg<opc, _.info256, OpNode, SrcRC>, EVEX_V256;
    defm Z128 : avx512_int_broadcast_reg<opc, _.info128, OpNode, SrcRC>, EVEX_V128;
  }
}

let isCodeGenOnly = 1 in {
defm VPBROADCASTBr : avx512_int_broadcast_reg_vl<0x7A, avx512vl_i8_info,
                                                 X86VBroadcast, GR8, HasBWI>;
defm VPBROADCASTWr : avx512_int_broadcast_reg_vl<0x7B, avx512vl_i16_info,
                                                 X86VBroadcast, GR16, HasBWI>;
}
let isAsmParserOnly = 1 in {
  defm VPBROADCASTBr_Alt : avx512_int_broadcast_reg_vl<0x7A, avx512vl_i8_info,
                                                       null_frag, GR32, HasBWI>;
  defm VPBROADCASTWr_Alt : avx512_int_broadcast_reg_vl<0x7B, avx512vl_i16_info,
                                                       null_frag, GR32, HasBWI>;
}
defm VPBROADCASTDr : avx512_int_broadcast_reg_vl<0x7C, avx512vl_i32_info,
                                                 X86VBroadcast, GR32, HasAVX512>;
defm VPBROADCASTQr : avx512_int_broadcast_reg_vl<0x7C, avx512vl_i64_info,
                                                 X86VBroadcast, GR64, HasAVX512>, VEX_W;

def : Pat <(v16i32 (X86vzext VK16WM:$mask)),
           (VPBROADCASTDrZrkz VK16WM:$mask, (i32 (MOV32ri 0x1)))>;
def : Pat <(v8i64 (X86vzext VK8WM:$mask)),
           (VPBROADCASTQrZrkz VK8WM:$mask, (i64 (MOV64ri 0x1)))>;

// Provide aliases for broadcast from the same register class that
// automatically does the extract.
multiclass avx512_int_broadcast_rm_lowering<X86VectorVTInfo DestInfo,
                                            X86VectorVTInfo SrcInfo> {
  def : Pat<(DestInfo.VT (X86VBroadcast (SrcInfo.VT SrcInfo.RC:$src))),
            (!cast<Instruction>(NAME#DestInfo.ZSuffix#"r")
                (EXTRACT_SUBREG (SrcInfo.VT SrcInfo.RC:$src), sub_xmm))>;
}

multiclass avx512_int_broadcast_rm_vl<bits<8> opc, string OpcodeStr,
                                        AVX512VLVectorVTInfo _, Predicate prd> {
  let Predicates = [prd] in {
    defm Z :   avx512_broadcast_rm<opc, OpcodeStr, _.info512, _.info128>,
               avx512_int_broadcast_rm_lowering<_.info512, _.info256>,
                                  EVEX_V512;
    // Defined separately to avoid redefinition.
    defm Z_Alt : avx512_int_broadcast_rm_lowering<_.info512, _.info512>;
  }
  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_broadcast_rm<opc, OpcodeStr, _.info256, _.info128>,
                avx512_int_broadcast_rm_lowering<_.info256, _.info256>,
                                 EVEX_V256;
    defm Z128 : avx512_broadcast_rm<opc, OpcodeStr, _.info128, _.info128>,
                                 EVEX_V128;
  }
}

defm VPBROADCASTB  : avx512_int_broadcast_rm_vl<0x78, "vpbroadcastb",
                                           avx512vl_i8_info, HasBWI>;
defm VPBROADCASTW  : avx512_int_broadcast_rm_vl<0x79, "vpbroadcastw",
                                           avx512vl_i16_info, HasBWI>;
defm VPBROADCASTD  : avx512_int_broadcast_rm_vl<0x58, "vpbroadcastd",
                                           avx512vl_i32_info, HasAVX512>;
defm VPBROADCASTQ  : avx512_int_broadcast_rm_vl<0x59, "vpbroadcastq",
                                           avx512vl_i64_info, HasAVX512>, VEX_W;

multiclass avx512_subvec_broadcast_rm<bits<8> opc, string OpcodeStr,
                          X86VectorVTInfo _Dst, X86VectorVTInfo _Src> {
  defm rm : AVX512_maskable<opc, MRMSrcMem, _Dst, (outs _Dst.RC:$dst),
                           (ins _Src.MemOp:$src), OpcodeStr, "$src", "$src",
                           (_Dst.VT (X86SubVBroadcast
                             (_Src.VT (bitconvert (_Src.LdFrag addr:$src)))))>,
                            AVX5128IBase, EVEX;
}

let Predicates = [HasAVX512] in {
  // 32-bit targets will fail to load a i64 directly but can use ZEXT_LOAD.
  def : Pat<(v8i64 (X86VBroadcast (v8i64 (X86vzload addr:$src)))),
            (VPBROADCASTQZm addr:$src)>;
}

let Predicates = [HasVLX, HasBWI] in {
  // 32-bit targets will fail to load a i64 directly but can use ZEXT_LOAD.
  def : Pat<(v2i64 (X86VBroadcast (v2i64 (X86vzload addr:$src)))),
            (VPBROADCASTQZ128m addr:$src)>;
  def : Pat<(v4i64 (X86VBroadcast (v4i64 (X86vzload addr:$src)))),
            (VPBROADCASTQZ256m addr:$src)>;
  // loadi16 is tricky to fold, because !isTypeDesirableForOp, justifiably.
  // This means we'll encounter truncated i32 loads; match that here.
  def : Pat<(v8i16 (X86VBroadcast (i16 (trunc (i32 (load addr:$src)))))),
            (VPBROADCASTWZ128m addr:$src)>;
  def : Pat<(v16i16 (X86VBroadcast (i16 (trunc (i32 (load addr:$src)))))),
            (VPBROADCASTWZ256m addr:$src)>;
  def : Pat<(v8i16 (X86VBroadcast
              (i16 (trunc (i32 (zextloadi16 addr:$src)))))),
            (VPBROADCASTWZ128m addr:$src)>;
  def : Pat<(v16i16 (X86VBroadcast
              (i16 (trunc (i32 (zextloadi16 addr:$src)))))),
            (VPBROADCASTWZ256m addr:$src)>;
}

//===----------------------------------------------------------------------===//
// AVX-512 BROADCAST SUBVECTORS
//

defm VBROADCASTI32X4 : avx512_subvec_broadcast_rm<0x5a, "vbroadcasti32x4",
                       v16i32_info, v4i32x_info>,
                       EVEX_V512, EVEX_CD8<32, CD8VT4>;
defm VBROADCASTF32X4 : avx512_subvec_broadcast_rm<0x1a, "vbroadcastf32x4",
                       v16f32_info, v4f32x_info>,
                       EVEX_V512, EVEX_CD8<32, CD8VT4>;
defm VBROADCASTI64X4 : avx512_subvec_broadcast_rm<0x5b, "vbroadcasti64x4",
                       v8i64_info, v4i64x_info>, VEX_W,
                       EVEX_V512, EVEX_CD8<64, CD8VT4>;
defm VBROADCASTF64X4 : avx512_subvec_broadcast_rm<0x1b, "vbroadcastf64x4",
                       v8f64_info, v4f64x_info>, VEX_W,
                       EVEX_V512, EVEX_CD8<64, CD8VT4>;

let Predicates = [HasAVX512] in {
def : Pat<(v32i16 (X86SubVBroadcast (bc_v16i16 (loadv4i64 addr:$src)))),
          (VBROADCASTI64X4rm addr:$src)>;
def : Pat<(v64i8 (X86SubVBroadcast (bc_v32i8 (loadv4i64 addr:$src)))),
          (VBROADCASTI64X4rm addr:$src)>;

// Provide fallback in case the load node that is used in the patterns above
// is used by additional users, which prevents the pattern selection.
def : Pat<(v8f64 (X86SubVBroadcast (v4f64 VR256X:$src))),
          (VINSERTF64x4Zrr (INSERT_SUBREG (v8f64 (IMPLICIT_DEF)), VR256X:$src, sub_ymm),
                           (v4f64 VR256X:$src), 1)>;
def : Pat<(v8i64 (X86SubVBroadcast (v4i64 VR256X:$src))),
          (VINSERTI64x4Zrr (INSERT_SUBREG (v8i64 (IMPLICIT_DEF)), VR256X:$src, sub_ymm),
                           (v4i64 VR256X:$src), 1)>;
def : Pat<(v32i16 (X86SubVBroadcast (v16i16 VR256X:$src))),
          (VINSERTI64x4Zrr (INSERT_SUBREG (v32i16 (IMPLICIT_DEF)), VR256X:$src, sub_ymm),
                           (v16i16 VR256X:$src), 1)>;
def : Pat<(v64i8 (X86SubVBroadcast (v32i8 VR256X:$src))),
          (VINSERTI64x4Zrr (INSERT_SUBREG (v64i8 (IMPLICIT_DEF)), VR256X:$src, sub_ymm),
                           (v32i8 VR256X:$src), 1)>;

def : Pat<(v32i16 (X86SubVBroadcast (bc_v8i16 (loadv2i64 addr:$src)))),
          (VBROADCASTI32X4rm addr:$src)>;
def : Pat<(v64i8 (X86SubVBroadcast (bc_v16i8 (loadv2i64 addr:$src)))),
          (VBROADCASTI32X4rm addr:$src)>;
}

let Predicates = [HasVLX] in {
defm VBROADCASTI32X4Z256 : avx512_subvec_broadcast_rm<0x5a, "vbroadcasti32x4",
                           v8i32x_info, v4i32x_info>,
                           EVEX_V256, EVEX_CD8<32, CD8VT4>;
defm VBROADCASTF32X4Z256 : avx512_subvec_broadcast_rm<0x1a, "vbroadcastf32x4",
                           v8f32x_info, v4f32x_info>,
                           EVEX_V256, EVEX_CD8<32, CD8VT4>;

def : Pat<(v16i16 (X86SubVBroadcast (bc_v8i16 (loadv2i64 addr:$src)))),
          (VBROADCASTI32X4Z256rm addr:$src)>;
def : Pat<(v32i8 (X86SubVBroadcast (bc_v16i8 (loadv2i64 addr:$src)))),
          (VBROADCASTI32X4Z256rm addr:$src)>;

// Provide fallback in case the load node that is used in the patterns above
// is used by additional users, which prevents the pattern selection.
def : Pat<(v8f32 (X86SubVBroadcast (v4f32 VR128X:$src))),
          (VINSERTF32x4Z256rr (INSERT_SUBREG (v8f32 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                              (v4f32 VR128X:$src), 1)>;
def : Pat<(v8i32 (X86SubVBroadcast (v4i32 VR128X:$src))),
          (VINSERTI32x4Z256rr (INSERT_SUBREG (v8i32 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                              (v4i32 VR128X:$src), 1)>;
def : Pat<(v16i16 (X86SubVBroadcast (v8i16 VR128X:$src))),
          (VINSERTI32x4Z256rr (INSERT_SUBREG (v16i16 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                              (v8i16 VR128X:$src), 1)>;
def : Pat<(v32i8 (X86SubVBroadcast (v16i8 VR128X:$src))),
          (VINSERTI32x4Z256rr (INSERT_SUBREG (v32i8 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                              (v16i8 VR128X:$src), 1)>;
}

let Predicates = [HasVLX, HasDQI] in {
defm VBROADCASTI64X2Z128 : avx512_subvec_broadcast_rm<0x5a, "vbroadcasti64x2",
                           v4i64x_info, v2i64x_info>, VEX_W,
                           EVEX_V256, EVEX_CD8<64, CD8VT2>;
defm VBROADCASTF64X2Z128 : avx512_subvec_broadcast_rm<0x1a, "vbroadcastf64x2",
                           v4f64x_info, v2f64x_info>, VEX_W,
                           EVEX_V256, EVEX_CD8<64, CD8VT2>;

// Provide fallback in case the load node that is used in the patterns above
// is used by additional users, which prevents the pattern selection.
def : Pat<(v4f64 (X86SubVBroadcast (v2f64 VR128X:$src))),
          (VINSERTF64x2Z256rr (INSERT_SUBREG (v4f64 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                              (v2f64 VR128X:$src), 1)>;
def : Pat<(v4i64 (X86SubVBroadcast (v2i64 VR128X:$src))),
          (VINSERTI64x2Z256rr (INSERT_SUBREG (v4i64 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                              (v2i64 VR128X:$src), 1)>;
}

let Predicates = [HasVLX, NoDQI] in {
def : Pat<(v4f64 (X86SubVBroadcast (loadv2f64 addr:$src))),
          (VBROADCASTF32X4Z256rm addr:$src)>;
def : Pat<(v4i64 (X86SubVBroadcast (loadv2i64 addr:$src))),
          (VBROADCASTI32X4Z256rm addr:$src)>;

// Provide fallback in case the load node that is used in the patterns above
// is used by additional users, which prevents the pattern selection.
def : Pat<(v4f64 (X86SubVBroadcast (v2f64 VR128X:$src))),
          (VINSERTF32x4Z256rr (INSERT_SUBREG (v4f64 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                              (v2f64 VR128X:$src), 1)>;
def : Pat<(v4i64 (X86SubVBroadcast (v2i64 VR128X:$src))),
          (VINSERTI32x4Z256rr (INSERT_SUBREG (v4i64 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                              (v2i64 VR128X:$src), 1)>;
}

let Predicates = [HasAVX512, NoDQI] in {
def : Pat<(v8f64 (X86SubVBroadcast (loadv2f64 addr:$src))),
          (VBROADCASTF32X4rm addr:$src)>;
def : Pat<(v8i64 (X86SubVBroadcast (loadv2i64 addr:$src))),
          (VBROADCASTI32X4rm addr:$src)>;

def : Pat<(v16f32 (X86SubVBroadcast (loadv8f32 addr:$src))),
          (VBROADCASTF64X4rm addr:$src)>;
def : Pat<(v16i32 (X86SubVBroadcast (bc_v8i32 (loadv4i64 addr:$src)))),
          (VBROADCASTI64X4rm addr:$src)>;

// Provide fallback in case the load node that is used in the patterns above
// is used by additional users, which prevents the pattern selection.
def : Pat<(v16f32 (X86SubVBroadcast (v8f32 VR256X:$src))),
          (VINSERTF64x4Zrr (INSERT_SUBREG (v16f32 (IMPLICIT_DEF)), VR256X:$src, sub_ymm),
                           (v8f32 VR256X:$src), 1)>;
def : Pat<(v16i32 (X86SubVBroadcast (v8i32 VR256X:$src))),
          (VINSERTI64x4Zrr (INSERT_SUBREG (v16i32 (IMPLICIT_DEF)), VR256X:$src, sub_ymm),
                           (v8i32 VR256X:$src), 1)>;
}

let Predicates = [HasDQI] in {
defm VBROADCASTI64X2 : avx512_subvec_broadcast_rm<0x5a, "vbroadcasti64x2",
                       v8i64_info, v2i64x_info>, VEX_W,
                       EVEX_V512, EVEX_CD8<64, CD8VT2>;
defm VBROADCASTI32X8 : avx512_subvec_broadcast_rm<0x5b, "vbroadcasti32x8",
                       v16i32_info, v8i32x_info>,
                       EVEX_V512, EVEX_CD8<32, CD8VT8>;
defm VBROADCASTF64X2 : avx512_subvec_broadcast_rm<0x1a, "vbroadcastf64x2",
                       v8f64_info, v2f64x_info>, VEX_W,
                       EVEX_V512, EVEX_CD8<64, CD8VT2>;
defm VBROADCASTF32X8 : avx512_subvec_broadcast_rm<0x1b, "vbroadcastf32x8",
                       v16f32_info, v8f32x_info>,
                       EVEX_V512, EVEX_CD8<32, CD8VT8>;

// Provide fallback in case the load node that is used in the patterns above
// is used by additional users, which prevents the pattern selection.
def : Pat<(v16f32 (X86SubVBroadcast (v8f32 VR256X:$src))),
          (VINSERTF32x8Zrr (INSERT_SUBREG (v16f32 (IMPLICIT_DEF)), VR256X:$src, sub_ymm),
                           (v8f32 VR256X:$src), 1)>;
def : Pat<(v16i32 (X86SubVBroadcast (v8i32 VR256X:$src))),
          (VINSERTI32x8Zrr (INSERT_SUBREG (v16i32 (IMPLICIT_DEF)), VR256X:$src, sub_ymm),
                           (v8i32 VR256X:$src), 1)>;
}

multiclass avx512_common_broadcast_32x2<bits<8> opc, string OpcodeStr,
                         AVX512VLVectorVTInfo _Dst, AVX512VLVectorVTInfo _Src> {
  let Predicates = [HasDQI] in
    defm Z :    avx512_broadcast_rm<opc, OpcodeStr, _Dst.info512, _Src.info128>,
                                  EVEX_V512;
  let Predicates = [HasDQI, HasVLX] in
    defm Z256 : avx512_broadcast_rm<opc, OpcodeStr, _Dst.info256, _Src.info128>,
                                  EVEX_V256;
}

multiclass avx512_common_broadcast_i32x2<bits<8> opc, string OpcodeStr,
                         AVX512VLVectorVTInfo _Dst, AVX512VLVectorVTInfo _Src> :
  avx512_common_broadcast_32x2<opc, OpcodeStr, _Dst, _Src> {

  let Predicates = [HasDQI, HasVLX] in
    defm Z128 : avx512_broadcast_rm<opc, OpcodeStr, _Dst.info128, _Src.info128>,
                                      EVEX_V128;
}

defm VBROADCASTI32X2  : avx512_common_broadcast_i32x2<0x59, "vbroadcasti32x2",
                                          avx512vl_i32_info, avx512vl_i64_info>;
defm VBROADCASTF32X2  : avx512_common_broadcast_32x2<0x19, "vbroadcastf32x2",
                                          avx512vl_f32_info, avx512vl_f64_info>;

let Predicates = [HasVLX] in {
def : Pat<(v8f32 (X86VBroadcast (v8f32 VR256X:$src))),
          (VBROADCASTSSZ256r (EXTRACT_SUBREG (v8f32 VR256X:$src), sub_xmm))>;
def : Pat<(v4f64 (X86VBroadcast (v4f64 VR256X:$src))),
          (VBROADCASTSDZ256r (EXTRACT_SUBREG (v4f64 VR256X:$src), sub_xmm))>;
}

def : Pat<(v16f32 (X86VBroadcast (v16f32 VR512:$src))),
          (VBROADCASTSSZr (EXTRACT_SUBREG (v16f32 VR512:$src), sub_xmm))>;
def : Pat<(v16f32 (X86VBroadcast (v8f32 VR256X:$src))),
          (VBROADCASTSSZr (EXTRACT_SUBREG (v8f32 VR256X:$src), sub_xmm))>;

def : Pat<(v8f64 (X86VBroadcast (v8f64 VR512:$src))),
          (VBROADCASTSDZr (EXTRACT_SUBREG (v8f64 VR512:$src), sub_xmm))>;
def : Pat<(v8f64 (X86VBroadcast (v4f64 VR256X:$src))),
          (VBROADCASTSDZr (EXTRACT_SUBREG (v4f64 VR256X:$src), sub_xmm))>;

//===----------------------------------------------------------------------===//
// AVX-512 BROADCAST MASK TO VECTOR REGISTER
//---
multiclass avx512_mask_broadcastm<bits<8> opc, string OpcodeStr,
                                  X86VectorVTInfo _, RegisterClass KRC> {
  def rr : AVX512XS8I<opc, MRMSrcReg, (outs _.RC:$dst), (ins KRC:$src),
                  !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                  [(set _.RC:$dst, (_.VT (X86VBroadcastm KRC:$src)))]>, EVEX;
}

multiclass avx512_mask_broadcast<bits<8> opc, string OpcodeStr,
                                 AVX512VLVectorVTInfo VTInfo, RegisterClass KRC> {
  let Predicates = [HasCDI] in
    defm Z : avx512_mask_broadcastm<opc, OpcodeStr, VTInfo.info512, KRC>, EVEX_V512;
  let Predicates = [HasCDI, HasVLX] in {
    defm Z256 : avx512_mask_broadcastm<opc, OpcodeStr, VTInfo.info256, KRC>, EVEX_V256;
    defm Z128 : avx512_mask_broadcastm<opc, OpcodeStr, VTInfo.info128, KRC>, EVEX_V128;
  }
}

defm VPBROADCASTMW2D : avx512_mask_broadcast<0x3A, "vpbroadcastmw2d",
                                               avx512vl_i32_info, VK16>;
defm VPBROADCASTMB2Q : avx512_mask_broadcast<0x2A, "vpbroadcastmb2q",
                                               avx512vl_i64_info, VK8>, VEX_W;

//===----------------------------------------------------------------------===//
// -- VPERMI2 - 3 source operands form --
multiclass avx512_perm_i<bits<8> opc, string OpcodeStr, X86VectorVTInfo _> {
let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in {
  // The index operand in the pattern should really be an integer type. However,
  // if we do that and it happens to come from a bitcast, then it becomes
  // difficult to find the bitcast needed to convert the index to the
  // destination type for the passthru since it will be folded with the bitcast
  // of the index operand.
  defm rr: AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.RC:$src3),
          OpcodeStr, "$src3, $src2", "$src2, $src3",
          (_.VT (X86VPermi2X _.RC:$src1, _.RC:$src2, _.RC:$src3)), 1>, EVEX_4V,
         AVX5128IBase;

  defm rm: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
            (ins _.RC:$src2, _.MemOp:$src3),
            OpcodeStr, "$src3, $src2", "$src2, $src3",
            (_.VT (X86VPermi2X _.RC:$src1, _.RC:$src2,
                   (_.VT (bitconvert (_.LdFrag addr:$src3))))), 1>,
            EVEX_4V, AVX5128IBase;
  }
}
multiclass avx512_perm_i_mb<bits<8> opc, string OpcodeStr,
                            X86VectorVTInfo _> {
  let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in
  defm rmb: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
              (ins _.RC:$src2, _.ScalarMemOp:$src3),
              OpcodeStr,   !strconcat("${src3}", _.BroadcastStr,", $src2"),
              !strconcat("$src2, ${src3}", _.BroadcastStr ),
              (_.VT (X86VPermi2X _.RC:$src1,
               _.RC:$src2,(_.VT (X86VBroadcast (_.ScalarLdFrag addr:$src3))))),
              1>, AVX5128IBase, EVEX_4V, EVEX_B;
}

multiclass avx512_perm_i_sizes<bits<8> opc, string OpcodeStr,
                               AVX512VLVectorVTInfo VTInfo> {
  defm NAME: avx512_perm_i<opc, OpcodeStr, VTInfo.info512>,
            avx512_perm_i_mb<opc, OpcodeStr, VTInfo.info512>, EVEX_V512;
  let Predicates = [HasVLX] in {
  defm NAME#128: avx512_perm_i<opc, OpcodeStr, VTInfo.info128>,
                 avx512_perm_i_mb<opc, OpcodeStr, VTInfo.info128>, EVEX_V128;
  defm NAME#256: avx512_perm_i<opc, OpcodeStr, VTInfo.info256>,
                 avx512_perm_i_mb<opc, OpcodeStr, VTInfo.info256>, EVEX_V256;
  }
}

multiclass avx512_perm_i_sizes_bw<bits<8> opc, string OpcodeStr,
                                 AVX512VLVectorVTInfo VTInfo,
                                 Predicate Prd> {
  let Predicates = [Prd] in
  defm NAME: avx512_perm_i<opc, OpcodeStr, VTInfo.info512>, EVEX_V512;
  let Predicates = [Prd, HasVLX] in {
  defm NAME#128: avx512_perm_i<opc, OpcodeStr, VTInfo.info128>, EVEX_V128;
  defm NAME#256: avx512_perm_i<opc, OpcodeStr, VTInfo.info256>,  EVEX_V256;
  }
}

defm VPERMI2D  : avx512_perm_i_sizes<0x76, "vpermi2d",
                  avx512vl_i32_info>, EVEX_CD8<32, CD8VF>;
defm VPERMI2Q  : avx512_perm_i_sizes<0x76, "vpermi2q",
                  avx512vl_i64_info>, VEX_W, EVEX_CD8<64, CD8VF>;
defm VPERMI2W  : avx512_perm_i_sizes_bw<0x75, "vpermi2w",
                  avx512vl_i16_info, HasBWI>,
                  VEX_W, EVEX_CD8<16, CD8VF>;
defm VPERMI2B  : avx512_perm_i_sizes_bw<0x75, "vpermi2b",
                  avx512vl_i8_info, HasVBMI>,
                  EVEX_CD8<8, CD8VF>;
defm VPERMI2PS : avx512_perm_i_sizes<0x77, "vpermi2ps",
                  avx512vl_f32_info>, EVEX_CD8<32, CD8VF>;
defm VPERMI2PD : avx512_perm_i_sizes<0x77, "vpermi2pd",
                  avx512vl_f64_info>, VEX_W, EVEX_CD8<64, CD8VF>;

// VPERMT2
multiclass avx512_perm_t<bits<8> opc, string OpcodeStr,
                         X86VectorVTInfo _, X86VectorVTInfo IdxVT> {
let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in {
  defm rr: AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
          (ins IdxVT.RC:$src2, _.RC:$src3),
          OpcodeStr, "$src3, $src2", "$src2, $src3",
          (_.VT (X86VPermt2 _.RC:$src1, IdxVT.RC:$src2, _.RC:$src3)), 1>,
          EVEX_4V, AVX5128IBase;

  defm rm: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
            (ins IdxVT.RC:$src2, _.MemOp:$src3),
            OpcodeStr, "$src3, $src2", "$src2, $src3",
            (_.VT (X86VPermt2 _.RC:$src1, IdxVT.RC:$src2,
                   (bitconvert (_.LdFrag addr:$src3)))), 1>,
            EVEX_4V, AVX5128IBase;
  }
}
multiclass avx512_perm_t_mb<bits<8> opc, string OpcodeStr,
                            X86VectorVTInfo _, X86VectorVTInfo IdxVT> {
  let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in
  defm rmb: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
              (ins IdxVT.RC:$src2, _.ScalarMemOp:$src3),
              OpcodeStr,   !strconcat("${src3}", _.BroadcastStr,", $src2"),
              !strconcat("$src2, ${src3}", _.BroadcastStr ),
              (_.VT (X86VPermt2 _.RC:$src1,
               IdxVT.RC:$src2,(_.VT (X86VBroadcast (_.ScalarLdFrag addr:$src3))))),
              1>, AVX5128IBase, EVEX_4V, EVEX_B;
}

multiclass avx512_perm_t_sizes<bits<8> opc, string OpcodeStr,
                               AVX512VLVectorVTInfo VTInfo,
                               AVX512VLVectorVTInfo ShuffleMask> {
  defm NAME: avx512_perm_t<opc, OpcodeStr, VTInfo.info512,
                              ShuffleMask.info512>,
            avx512_perm_t_mb<opc, OpcodeStr, VTInfo.info512,
                              ShuffleMask.info512>, EVEX_V512;
  let Predicates = [HasVLX] in {
  defm NAME#128: avx512_perm_t<opc, OpcodeStr, VTInfo.info128,
                              ShuffleMask.info128>,
                 avx512_perm_t_mb<opc, OpcodeStr, VTInfo.info128,
                              ShuffleMask.info128>, EVEX_V128;
  defm NAME#256: avx512_perm_t<opc, OpcodeStr, VTInfo.info256,
                              ShuffleMask.info256>,
                 avx512_perm_t_mb<opc, OpcodeStr, VTInfo.info256,
                              ShuffleMask.info256>, EVEX_V256;
  }
}

multiclass avx512_perm_t_sizes_bw<bits<8> opc, string OpcodeStr,
                                 AVX512VLVectorVTInfo VTInfo,
                                 AVX512VLVectorVTInfo Idx,
                                 Predicate Prd> {
  let Predicates = [Prd] in
  defm NAME: avx512_perm_t<opc, OpcodeStr, VTInfo.info512,
                           Idx.info512>, EVEX_V512;
  let Predicates = [Prd, HasVLX] in {
  defm NAME#128: avx512_perm_t<opc, OpcodeStr, VTInfo.info128,
                               Idx.info128>, EVEX_V128;
  defm NAME#256: avx512_perm_t<opc, OpcodeStr, VTInfo.info256,
                               Idx.info256>, EVEX_V256;
  }
}

defm VPERMT2D  : avx512_perm_t_sizes<0x7E, "vpermt2d",
                  avx512vl_i32_info, avx512vl_i32_info>, EVEX_CD8<32, CD8VF>;
defm VPERMT2Q  : avx512_perm_t_sizes<0x7E, "vpermt2q",
                  avx512vl_i64_info, avx512vl_i64_info>, VEX_W, EVEX_CD8<64, CD8VF>;
defm VPERMT2W  : avx512_perm_t_sizes_bw<0x7D, "vpermt2w",
                  avx512vl_i16_info, avx512vl_i16_info, HasBWI>,
                  VEX_W, EVEX_CD8<16, CD8VF>;
defm VPERMT2B  : avx512_perm_t_sizes_bw<0x7D, "vpermt2b",
                  avx512vl_i8_info, avx512vl_i8_info, HasVBMI>,
                  EVEX_CD8<8, CD8VF>;
defm VPERMT2PS : avx512_perm_t_sizes<0x7F, "vpermt2ps",
                  avx512vl_f32_info, avx512vl_i32_info>, EVEX_CD8<32, CD8VF>;
defm VPERMT2PD : avx512_perm_t_sizes<0x7F, "vpermt2pd",
                  avx512vl_f64_info, avx512vl_i64_info>, VEX_W, EVEX_CD8<64, CD8VF>;

//===----------------------------------------------------------------------===//
// AVX-512 - BLEND using mask
//
multiclass avx512_blendmask<bits<8> opc, string OpcodeStr, X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain, hasSideEffects = 0 in {
  def rr : AVX5128I<opc, MRMSrcReg, (outs _.RC:$dst),
             (ins _.RC:$src1, _.RC:$src2),
             !strconcat(OpcodeStr,
             "\t{$src2, $src1, ${dst}|${dst}, $src1, $src2}"),
             []>, EVEX_4V;
  def rrk : AVX5128I<opc, MRMSrcReg, (outs _.RC:$dst),
             (ins _.KRCWM:$mask, _.RC:$src1, _.RC:$src2),
             !strconcat(OpcodeStr,
             "\t{$src2, $src1, ${dst} {${mask}}|${dst} {${mask}}, $src1, $src2}"),
             []>, EVEX_4V, EVEX_K;
  def rrkz : AVX5128I<opc, MRMSrcReg, (outs _.RC:$dst),
             (ins _.KRCWM:$mask, _.RC:$src1, _.RC:$src2),
             !strconcat(OpcodeStr,
             "\t{$src2, $src1, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src1, $src2}"),
             []>, EVEX_4V, EVEX_KZ;
  let mayLoad = 1 in {
  def rm  : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst),
             (ins _.RC:$src1, _.MemOp:$src2),
             !strconcat(OpcodeStr,
             "\t{$src2, $src1, ${dst}|${dst}, $src1, $src2}"),
             []>, EVEX_4V, EVEX_CD8<_.EltSize, CD8VF>;
  def rmk : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst),
             (ins _.KRCWM:$mask, _.RC:$src1, _.MemOp:$src2),
             !strconcat(OpcodeStr,
             "\t{$src2, $src1, ${dst} {${mask}}|${dst} {${mask}}, $src1, $src2}"),
             []>, EVEX_4V, EVEX_K, EVEX_CD8<_.EltSize, CD8VF>;
  def rmkz : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst),
             (ins _.KRCWM:$mask, _.RC:$src1, _.MemOp:$src2),
             !strconcat(OpcodeStr,
             "\t{$src2, $src1, ${dst} {${mask}} {z}|${dst} {${mask}} {z}, $src1, $src2}"),
             []>, EVEX_4V, EVEX_KZ, EVEX_CD8<_.EltSize, CD8VF>;
  }
  }
}
multiclass avx512_blendmask_rmb<bits<8> opc, string OpcodeStr, X86VectorVTInfo _> {

  let mayLoad = 1, hasSideEffects = 0 in {
  def rmbk : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst),
      (ins _.KRCWM:$mask, _.RC:$src1, _.ScalarMemOp:$src2),
       !strconcat(OpcodeStr,
            "\t{${src2}", _.BroadcastStr, ", $src1, $dst {${mask}}|",
            "$dst {${mask}}, $src1, ${src2}", _.BroadcastStr, "}"),
      []>, EVEX_4V, EVEX_K, EVEX_B, EVEX_CD8<_.EltSize, CD8VF>;

  def rmb : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst),
      (ins _.RC:$src1, _.ScalarMemOp:$src2),
       !strconcat(OpcodeStr,
            "\t{${src2}", _.BroadcastStr, ", $src1, $dst|",
            "$dst, $src1, ${src2}", _.BroadcastStr, "}"),
      []>,  EVEX_4V, EVEX_B, EVEX_CD8<_.EltSize, CD8VF>;
  }
}

multiclass blendmask_dq <bits<8> opc, string OpcodeStr,
                                 AVX512VLVectorVTInfo VTInfo> {
  defm Z : avx512_blendmask      <opc, OpcodeStr, VTInfo.info512>,
           avx512_blendmask_rmb  <opc, OpcodeStr, VTInfo.info512>, EVEX_V512;

  let Predicates = [HasVLX] in {
    defm Z256 : avx512_blendmask<opc, OpcodeStr, VTInfo.info256>,
                avx512_blendmask_rmb  <opc, OpcodeStr, VTInfo.info256>, EVEX_V256;
    defm Z128 : avx512_blendmask<opc, OpcodeStr, VTInfo.info128>,
                avx512_blendmask_rmb  <opc, OpcodeStr, VTInfo.info128>, EVEX_V128;
  }
}

multiclass blendmask_bw <bits<8> opc, string OpcodeStr,
                         AVX512VLVectorVTInfo VTInfo> {
  let Predicates = [HasBWI] in
    defm Z : avx512_blendmask    <opc, OpcodeStr, VTInfo.info512>, EVEX_V512;

  let Predicates = [HasBWI, HasVLX] in {
    defm Z256 : avx512_blendmask <opc, OpcodeStr, VTInfo.info256>, EVEX_V256;
    defm Z128 : avx512_blendmask <opc, OpcodeStr, VTInfo.info128>, EVEX_V128;
  }
}


defm VBLENDMPS : blendmask_dq <0x65, "vblendmps", avx512vl_f32_info>;
defm VBLENDMPD : blendmask_dq <0x65, "vblendmpd", avx512vl_f64_info>, VEX_W;
defm VPBLENDMD : blendmask_dq <0x64, "vpblendmd", avx512vl_i32_info>;
defm VPBLENDMQ : blendmask_dq <0x64, "vpblendmq", avx512vl_i64_info>, VEX_W;
defm VPBLENDMB : blendmask_bw <0x66, "vpblendmb", avx512vl_i8_info>;
defm VPBLENDMW : blendmask_bw <0x66, "vpblendmw", avx512vl_i16_info>, VEX_W;


//===----------------------------------------------------------------------===//
// Compare Instructions
//===----------------------------------------------------------------------===//

// avx512_cmp_scalar - AVX512 CMPSS and CMPSD

multiclass avx512_cmp_scalar<X86VectorVTInfo _, SDNode OpNode, SDNode OpNodeRnd>{

  defm  rr_Int  : AVX512_maskable_cmp<0xC2, MRMSrcReg, _,
                      (outs _.KRC:$dst),
                      (ins _.RC:$src1, _.RC:$src2, AVXCC:$cc),
                      "vcmp${cc}"#_.Suffix,
                      "$src2, $src1", "$src1, $src2",
                      (OpNode (_.VT _.RC:$src1),
                              (_.VT _.RC:$src2),
                              imm:$cc)>, EVEX_4V;
  defm  rm_Int  : AVX512_maskable_cmp<0xC2, MRMSrcMem, _,
                    (outs _.KRC:$dst),
                    (ins _.RC:$src1, _.IntScalarMemOp:$src2, AVXCC:$cc),
                    "vcmp${cc}"#_.Suffix,
                    "$src2, $src1", "$src1, $src2",
                    (OpNode (_.VT _.RC:$src1), _.ScalarIntMemCPat:$src2,
                        imm:$cc)>, EVEX_4V, EVEX_CD8<_.EltSize, CD8VT1>;

  defm  rrb_Int  : AVX512_maskable_cmp<0xC2, MRMSrcReg, _,
                     (outs _.KRC:$dst),
                     (ins _.RC:$src1, _.RC:$src2, AVXCC:$cc),
                     "vcmp${cc}"#_.Suffix,
                     "{sae}, $src2, $src1", "$src1, $src2, {sae}",
                     (OpNodeRnd (_.VT _.RC:$src1),
                                (_.VT _.RC:$src2),
                                imm:$cc,
                                (i32 FROUND_NO_EXC))>, EVEX_4V, EVEX_B;
  // Accept explicit immediate argument form instead of comparison code.
  let isAsmParserOnly = 1, hasSideEffects = 0 in {
    defm  rri_alt  : AVX512_maskable_cmp_alt<0xC2, MRMSrcReg, _,
                        (outs VK1:$dst),
                        (ins _.RC:$src1, _.RC:$src2, u8imm:$cc),
                        "vcmp"#_.Suffix,
                        "$cc, $src2, $src1", "$src1, $src2, $cc">, EVEX_4V;
    defm  rmi_alt  : AVX512_maskable_cmp_alt<0xC2, MRMSrcMem, _,
                        (outs _.KRC:$dst),
                        (ins _.RC:$src1, _.ScalarMemOp:$src2, u8imm:$cc),
                        "vcmp"#_.Suffix,
                        "$cc, $src2, $src1", "$src1, $src2, $cc">,
                        EVEX_4V, EVEX_CD8<_.EltSize, CD8VT1>;

    defm  rrb_alt  : AVX512_maskable_cmp_alt<0xC2, MRMSrcReg, _,
                       (outs _.KRC:$dst),
                       (ins _.RC:$src1, _.RC:$src2, u8imm:$cc),
                       "vcmp"#_.Suffix,
                       "$cc, {sae}, $src2, $src1","$src1, $src2, {sae}, $cc">,
                       EVEX_4V, EVEX_B;
  }// let isAsmParserOnly = 1, hasSideEffects = 0

  let isCodeGenOnly = 1 in {
    let isCommutable = 1 in
    def rr : AVX512Ii8<0xC2, MRMSrcReg,
                (outs _.KRC:$dst), (ins _.FRC:$src1, _.FRC:$src2, AVXCC:$cc),
                !strconcat("vcmp${cc}", _.Suffix,
                           "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                [(set _.KRC:$dst, (OpNode _.FRC:$src1,
                                          _.FRC:$src2,
                                          imm:$cc))],
                IIC_SSE_ALU_F32S_RR>, EVEX_4V;
    def rm : AVX512Ii8<0xC2, MRMSrcMem,
              (outs _.KRC:$dst),
              (ins _.FRC:$src1, _.ScalarMemOp:$src2, AVXCC:$cc),
              !strconcat("vcmp${cc}", _.Suffix,
                         "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
              [(set _.KRC:$dst, (OpNode _.FRC:$src1,
                                        (_.ScalarLdFrag addr:$src2),
                                        imm:$cc))],
              IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_CD8<_.EltSize, CD8VT1>;
  }
}

let Predicates = [HasAVX512] in {
  let ExeDomain = SSEPackedSingle in
  defm VCMPSSZ : avx512_cmp_scalar<f32x_info, X86cmpms, X86cmpmsRnd>,
                                   AVX512XSIi8Base;
  let ExeDomain = SSEPackedDouble in
  defm VCMPSDZ : avx512_cmp_scalar<f64x_info, X86cmpms, X86cmpmsRnd>,
                                   AVX512XDIi8Base, VEX_W;
}

multiclass avx512_icmp_packed<bits<8> opc, string OpcodeStr, SDNode OpNode,
              X86VectorVTInfo _, bit IsCommutable> {
  let isCommutable = IsCommutable in
  def rr : AVX512BI<opc, MRMSrcReg,
             (outs _.KRC:$dst), (ins _.RC:$src1, _.RC:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set _.KRC:$dst, (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2)))],
             IIC_SSE_ALU_F32P_RR>, EVEX_4V;
  def rm : AVX512BI<opc, MRMSrcMem,
             (outs _.KRC:$dst), (ins _.RC:$src1, _.MemOp:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set _.KRC:$dst, (OpNode (_.VT _.RC:$src1),
                                     (_.VT (bitconvert (_.LdFrag addr:$src2)))))],
             IIC_SSE_ALU_F32P_RM>, EVEX_4V;
  def rrk : AVX512BI<opc, MRMSrcReg,
              (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1, _.RC:$src2),
              !strconcat(OpcodeStr, "\t{$src2, $src1, $dst {${mask}}|",
                          "$dst {${mask}}, $src1, $src2}"),
              [(set _.KRC:$dst, (and _.KRCWM:$mask,
                                   (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2))))],
              IIC_SSE_ALU_F32P_RR>, EVEX_4V, EVEX_K;
  def rmk : AVX512BI<opc, MRMSrcMem,
              (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1, _.MemOp:$src2),
              !strconcat(OpcodeStr, "\t{$src2, $src1, $dst {${mask}}|",
                          "$dst {${mask}}, $src1, $src2}"),
              [(set _.KRC:$dst, (and _.KRCWM:$mask,
                                   (OpNode (_.VT _.RC:$src1),
                                       (_.VT (bitconvert
                                              (_.LdFrag addr:$src2))))))],
              IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_K;
}

multiclass avx512_icmp_packed_rmb<bits<8> opc, string OpcodeStr, SDNode OpNode,
              X86VectorVTInfo _, bit IsCommutable> :
           avx512_icmp_packed<opc, OpcodeStr, OpNode, _, IsCommutable> {
  def rmb : AVX512BI<opc, MRMSrcMem,
              (outs _.KRC:$dst), (ins _.RC:$src1, _.ScalarMemOp:$src2),
              !strconcat(OpcodeStr, "\t{${src2}", _.BroadcastStr, ", $src1, $dst",
                                    "|$dst, $src1, ${src2}", _.BroadcastStr, "}"),
              [(set _.KRC:$dst, (OpNode (_.VT _.RC:$src1),
                              (X86VBroadcast (_.ScalarLdFrag addr:$src2))))],
              IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_B;
  def rmbk : AVX512BI<opc, MRMSrcMem,
               (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1,
                                       _.ScalarMemOp:$src2),
               !strconcat(OpcodeStr,
                          "\t{${src2}", _.BroadcastStr, ", $src1, $dst {${mask}}|",
                          "$dst {${mask}}, $src1, ${src2}", _.BroadcastStr, "}"),
               [(set _.KRC:$dst, (and _.KRCWM:$mask,
                                      (OpNode (_.VT _.RC:$src1),
                                        (X86VBroadcast
                                          (_.ScalarLdFrag addr:$src2)))))],
               IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_K, EVEX_B;
}

multiclass avx512_icmp_packed_vl<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                 AVX512VLVectorVTInfo VTInfo, Predicate prd,
                                 bit IsCommutable = 0> {
  let Predicates = [prd] in
  defm Z : avx512_icmp_packed<opc, OpcodeStr, OpNode, VTInfo.info512,
                              IsCommutable>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_icmp_packed<opc, OpcodeStr, OpNode, VTInfo.info256,
                                   IsCommutable>, EVEX_V256;
    defm Z128 : avx512_icmp_packed<opc, OpcodeStr, OpNode, VTInfo.info128,
                                   IsCommutable>, EVEX_V128;
  }
}

multiclass avx512_icmp_packed_rmb_vl<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode, AVX512VLVectorVTInfo VTInfo,
                                  Predicate prd, bit IsCommutable = 0> {
  let Predicates = [prd] in
  defm Z : avx512_icmp_packed_rmb<opc, OpcodeStr, OpNode, VTInfo.info512,
                                  IsCommutable>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_icmp_packed_rmb<opc, OpcodeStr, OpNode, VTInfo.info256,
                                       IsCommutable>, EVEX_V256;
    defm Z128 : avx512_icmp_packed_rmb<opc, OpcodeStr, OpNode, VTInfo.info128,
                                       IsCommutable>, EVEX_V128;
  }
}

defm VPCMPEQB : avx512_icmp_packed_vl<0x74, "vpcmpeqb", X86pcmpeqm,
                      avx512vl_i8_info, HasBWI, 1>,
                EVEX_CD8<8, CD8VF>;

defm VPCMPEQW : avx512_icmp_packed_vl<0x75, "vpcmpeqw", X86pcmpeqm,
                      avx512vl_i16_info, HasBWI, 1>,
                EVEX_CD8<16, CD8VF>;

defm VPCMPEQD : avx512_icmp_packed_rmb_vl<0x76, "vpcmpeqd", X86pcmpeqm,
                      avx512vl_i32_info, HasAVX512, 1>,
                EVEX_CD8<32, CD8VF>;

defm VPCMPEQQ : avx512_icmp_packed_rmb_vl<0x29, "vpcmpeqq", X86pcmpeqm,
                      avx512vl_i64_info, HasAVX512, 1>,
                T8PD, VEX_W, EVEX_CD8<64, CD8VF>;

defm VPCMPGTB : avx512_icmp_packed_vl<0x64, "vpcmpgtb", X86pcmpgtm,
                      avx512vl_i8_info, HasBWI>,
                EVEX_CD8<8, CD8VF>;

defm VPCMPGTW : avx512_icmp_packed_vl<0x65, "vpcmpgtw", X86pcmpgtm,
                      avx512vl_i16_info, HasBWI>,
                EVEX_CD8<16, CD8VF>;

defm VPCMPGTD : avx512_icmp_packed_rmb_vl<0x66, "vpcmpgtd", X86pcmpgtm,
                      avx512vl_i32_info, HasAVX512>,
                EVEX_CD8<32, CD8VF>;

defm VPCMPGTQ : avx512_icmp_packed_rmb_vl<0x37, "vpcmpgtq", X86pcmpgtm,
                      avx512vl_i64_info, HasAVX512>,
                T8PD, VEX_W, EVEX_CD8<64, CD8VF>;

let Predicates = [HasAVX512, NoVLX] in {
def : Pat<(v8i1 (X86pcmpgtm (v8i32 VR256X:$src1), (v8i32 VR256X:$src2))),
            (COPY_TO_REGCLASS (VPCMPGTDZrr
            (v16i32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src1, sub_ymm)),
            (v16i32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src2, sub_ymm))), VK8)>;

def : Pat<(v8i1 (X86pcmpeqm (v8i32 VR256X:$src1), (v8i32 VR256X:$src2))),
            (COPY_TO_REGCLASS (VPCMPEQDZrr
            (v16i32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src1, sub_ymm)),
            (v16i32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src2, sub_ymm))), VK8)>;
}

multiclass avx512_icmp_cc<bits<8> opc, string Suffix, SDNode OpNode,
                          X86VectorVTInfo _> {
  let isCommutable = 1 in
  def rri : AVX512AIi8<opc, MRMSrcReg,
             (outs _.KRC:$dst), (ins _.RC:$src1, _.RC:$src2, AVX512ICC:$cc),
             !strconcat("vpcmp${cc}", Suffix,
                        "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set _.KRC:$dst, (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                                       imm:$cc))],
             IIC_SSE_ALU_F32P_RR>, EVEX_4V;
  def rmi : AVX512AIi8<opc, MRMSrcMem,
             (outs _.KRC:$dst), (ins _.RC:$src1, _.MemOp:$src2, AVX512ICC:$cc),
             !strconcat("vpcmp${cc}", Suffix,
                        "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set _.KRC:$dst, (OpNode (_.VT _.RC:$src1),
                              (_.VT (bitconvert (_.LdFrag addr:$src2))),
                              imm:$cc))],
             IIC_SSE_ALU_F32P_RM>, EVEX_4V;
  def rrik : AVX512AIi8<opc, MRMSrcReg,
              (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1, _.RC:$src2,
                                      AVX512ICC:$cc),
              !strconcat("vpcmp${cc}", Suffix,
                         "\t{$src2, $src1, $dst {${mask}}|",
                         "$dst {${mask}}, $src1, $src2}"),
              [(set _.KRC:$dst, (and _.KRCWM:$mask,
                                  (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                                          imm:$cc)))],
              IIC_SSE_ALU_F32P_RR>, EVEX_4V, EVEX_K;
  def rmik : AVX512AIi8<opc, MRMSrcMem,
              (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1, _.MemOp:$src2,
                                    AVX512ICC:$cc),
              !strconcat("vpcmp${cc}", Suffix,
                         "\t{$src2, $src1, $dst {${mask}}|",
                         "$dst {${mask}}, $src1, $src2}"),
              [(set _.KRC:$dst, (and _.KRCWM:$mask,
                                   (OpNode (_.VT _.RC:$src1),
                                      (_.VT (bitconvert (_.LdFrag addr:$src2))),
                                      imm:$cc)))],
              IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_K;

  // Accept explicit immediate argument form instead of comparison code.
  let isAsmParserOnly = 1, hasSideEffects = 0 in {
    def rri_alt : AVX512AIi8<opc, MRMSrcReg,
               (outs _.KRC:$dst), (ins _.RC:$src1, _.RC:$src2, u8imm:$cc),
               !strconcat("vpcmp", Suffix, "\t{$cc, $src2, $src1, $dst|",
                          "$dst, $src1, $src2, $cc}"),
               [], IIC_SSE_ALU_F32P_RR>, EVEX_4V;
    let mayLoad = 1 in
    def rmi_alt : AVX512AIi8<opc, MRMSrcMem,
               (outs _.KRC:$dst), (ins _.RC:$src1, _.MemOp:$src2, u8imm:$cc),
               !strconcat("vpcmp", Suffix, "\t{$cc, $src2, $src1, $dst|",
                          "$dst, $src1, $src2, $cc}"),
               [], IIC_SSE_ALU_F32P_RM>, EVEX_4V;
    def rrik_alt : AVX512AIi8<opc, MRMSrcReg,
               (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1, _.RC:$src2,
                                       u8imm:$cc),
               !strconcat("vpcmp", Suffix,
                          "\t{$cc, $src2, $src1, $dst {${mask}}|",
                          "$dst {${mask}}, $src1, $src2, $cc}"),
               [], IIC_SSE_ALU_F32P_RR>, EVEX_4V, EVEX_K;
    let mayLoad = 1 in
    def rmik_alt : AVX512AIi8<opc, MRMSrcMem,
               (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1, _.MemOp:$src2,
                                       u8imm:$cc),
               !strconcat("vpcmp", Suffix,
                          "\t{$cc, $src2, $src1, $dst {${mask}}|",
                          "$dst {${mask}}, $src1, $src2, $cc}"),
               [], IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_K;
  }
}

multiclass avx512_icmp_cc_rmb<bits<8> opc, string Suffix, SDNode OpNode,
                              X86VectorVTInfo _> :
           avx512_icmp_cc<opc, Suffix, OpNode, _> {
  def rmib : AVX512AIi8<opc, MRMSrcMem,
             (outs _.KRC:$dst), (ins _.RC:$src1, _.ScalarMemOp:$src2,
                                     AVX512ICC:$cc),
             !strconcat("vpcmp${cc}", Suffix,
                        "\t{${src2}", _.BroadcastStr, ", $src1, $dst|",
                        "$dst, $src1, ${src2}", _.BroadcastStr, "}"),
             [(set _.KRC:$dst, (OpNode (_.VT _.RC:$src1),
                               (X86VBroadcast (_.ScalarLdFrag addr:$src2)),
                               imm:$cc))],
             IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_B;
  def rmibk : AVX512AIi8<opc, MRMSrcMem,
              (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1,
                                       _.ScalarMemOp:$src2, AVX512ICC:$cc),
              !strconcat("vpcmp${cc}", Suffix,
                       "\t{${src2}", _.BroadcastStr, ", $src1, $dst {${mask}}|",
                       "$dst {${mask}}, $src1, ${src2}", _.BroadcastStr, "}"),
              [(set _.KRC:$dst, (and _.KRCWM:$mask,
                                  (OpNode (_.VT _.RC:$src1),
                                    (X86VBroadcast (_.ScalarLdFrag addr:$src2)),
                                    imm:$cc)))],
              IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_K, EVEX_B;

  // Accept explicit immediate argument form instead of comparison code.
  let isAsmParserOnly = 1, hasSideEffects = 0, mayLoad = 1 in {
    def rmib_alt : AVX512AIi8<opc, MRMSrcMem,
               (outs _.KRC:$dst), (ins _.RC:$src1, _.ScalarMemOp:$src2,
                                       u8imm:$cc),
               !strconcat("vpcmp", Suffix,
                   "\t{$cc, ${src2}", _.BroadcastStr, ", $src1, $dst|",
                   "$dst, $src1, ${src2}", _.BroadcastStr, ", $cc}"),
               [], IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_B;
    def rmibk_alt : AVX512AIi8<opc, MRMSrcMem,
               (outs _.KRC:$dst), (ins _.KRCWM:$mask, _.RC:$src1,
                                       _.ScalarMemOp:$src2, u8imm:$cc),
               !strconcat("vpcmp", Suffix,
                  "\t{$cc, ${src2}", _.BroadcastStr, ", $src1, $dst {${mask}}|",
                  "$dst {${mask}}, $src1, ${src2}", _.BroadcastStr, ", $cc}"),
               [], IIC_SSE_ALU_F32P_RM>, EVEX_4V, EVEX_K, EVEX_B;
  }
}

multiclass avx512_icmp_cc_vl<bits<8> opc, string Suffix, SDNode OpNode,
                             AVX512VLVectorVTInfo VTInfo, Predicate prd> {
  let Predicates = [prd] in
  defm Z : avx512_icmp_cc<opc, Suffix, OpNode, VTInfo.info512>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_icmp_cc<opc, Suffix, OpNode, VTInfo.info256>, EVEX_V256;
    defm Z128 : avx512_icmp_cc<opc, Suffix, OpNode, VTInfo.info128>, EVEX_V128;
  }
}

multiclass avx512_icmp_cc_rmb_vl<bits<8> opc, string Suffix, SDNode OpNode,
                                AVX512VLVectorVTInfo VTInfo, Predicate prd> {
  let Predicates = [prd] in
  defm Z : avx512_icmp_cc_rmb<opc, Suffix, OpNode, VTInfo.info512>,
           EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_icmp_cc_rmb<opc, Suffix, OpNode, VTInfo.info256>,
                EVEX_V256;
    defm Z128 : avx512_icmp_cc_rmb<opc, Suffix, OpNode, VTInfo.info128>,
                EVEX_V128;
  }
}

defm VPCMPB : avx512_icmp_cc_vl<0x3F, "b", X86cmpm, avx512vl_i8_info,
                                HasBWI>, EVEX_CD8<8, CD8VF>;
defm VPCMPUB : avx512_icmp_cc_vl<0x3E, "ub", X86cmpmu, avx512vl_i8_info,
                                 HasBWI>, EVEX_CD8<8, CD8VF>;

defm VPCMPW : avx512_icmp_cc_vl<0x3F, "w", X86cmpm, avx512vl_i16_info,
                                HasBWI>, VEX_W, EVEX_CD8<16, CD8VF>;
defm VPCMPUW : avx512_icmp_cc_vl<0x3E, "uw", X86cmpmu, avx512vl_i16_info,
                                 HasBWI>, VEX_W, EVEX_CD8<16, CD8VF>;

defm VPCMPD : avx512_icmp_cc_rmb_vl<0x1F, "d", X86cmpm, avx512vl_i32_info,
                                    HasAVX512>, EVEX_CD8<32, CD8VF>;
defm VPCMPUD : avx512_icmp_cc_rmb_vl<0x1E, "ud", X86cmpmu, avx512vl_i32_info,
                                     HasAVX512>, EVEX_CD8<32, CD8VF>;

defm VPCMPQ : avx512_icmp_cc_rmb_vl<0x1F, "q", X86cmpm, avx512vl_i64_info,
                                    HasAVX512>, VEX_W, EVEX_CD8<64, CD8VF>;
defm VPCMPUQ : avx512_icmp_cc_rmb_vl<0x1E, "uq", X86cmpmu, avx512vl_i64_info,
                                     HasAVX512>, VEX_W, EVEX_CD8<64, CD8VF>;

multiclass avx512_vcmp_common<X86VectorVTInfo _> {

  defm  rri  : AVX512_maskable_cmp<0xC2, MRMSrcReg, _,
                   (outs _.KRC:$dst), (ins _.RC:$src1, _.RC:$src2,AVXCC:$cc),
                   "vcmp${cc}"#_.Suffix,
                   "$src2, $src1", "$src1, $src2",
                   (X86cmpm (_.VT _.RC:$src1),
                         (_.VT _.RC:$src2),
                           imm:$cc), 1>;

  defm  rmi  : AVX512_maskable_cmp<0xC2, MRMSrcMem, _,
                (outs _.KRC:$dst),(ins _.RC:$src1, _.MemOp:$src2, AVXCC:$cc),
                "vcmp${cc}"#_.Suffix,
                "$src2, $src1", "$src1, $src2",
                (X86cmpm (_.VT _.RC:$src1),
                        (_.VT (bitconvert (_.LdFrag addr:$src2))),
                        imm:$cc)>;

  defm  rmbi : AVX512_maskable_cmp<0xC2, MRMSrcMem, _,
                (outs _.KRC:$dst),
                (ins _.RC:$src1, _.ScalarMemOp:$src2, AVXCC:$cc),
                "vcmp${cc}"#_.Suffix,
                "${src2}"##_.BroadcastStr##", $src1",
                "$src1, ${src2}"##_.BroadcastStr,
                (X86cmpm (_.VT _.RC:$src1),
                        (_.VT (X86VBroadcast(_.ScalarLdFrag addr:$src2))),
                        imm:$cc)>,EVEX_B;
  // Accept explicit immediate argument form instead of comparison code.
  let isAsmParserOnly = 1, hasSideEffects = 0 in {
    defm  rri_alt : AVX512_maskable_cmp_alt<0xC2, MRMSrcReg, _,
                         (outs _.KRC:$dst),
                         (ins _.RC:$src1, _.RC:$src2, u8imm:$cc),
                         "vcmp"#_.Suffix,
                         "$cc, $src2, $src1", "$src1, $src2, $cc">;

    let mayLoad = 1 in {
      defm rmi_alt : AVX512_maskable_cmp_alt<0xC2, MRMSrcMem, _,
                             (outs _.KRC:$dst),
                             (ins _.RC:$src1, _.MemOp:$src2, u8imm:$cc),
                             "vcmp"#_.Suffix,
                             "$cc, $src2, $src1", "$src1, $src2, $cc">;

      defm  rmbi_alt : AVX512_maskable_cmp_alt<0xC2, MRMSrcMem, _,
                         (outs _.KRC:$dst),
                         (ins _.RC:$src1, _.ScalarMemOp:$src2, u8imm:$cc),
                         "vcmp"#_.Suffix,
                         "$cc, ${src2}"##_.BroadcastStr##", $src1",
                         "$src1, ${src2}"##_.BroadcastStr##", $cc">,EVEX_B;
    }
 }
}

multiclass avx512_vcmp_sae<X86VectorVTInfo _> {
  // comparison code form (VCMP[EQ/LT/LE/...]
  defm  rrib  : AVX512_maskable_cmp<0xC2, MRMSrcReg, _,
                     (outs _.KRC:$dst),(ins _.RC:$src1, _.RC:$src2, AVXCC:$cc),
                     "vcmp${cc}"#_.Suffix,
                     "{sae}, $src2, $src1", "$src1, $src2, {sae}",
                     (X86cmpmRnd (_.VT _.RC:$src1),
                                    (_.VT _.RC:$src2),
                                    imm:$cc,
                                (i32 FROUND_NO_EXC))>, EVEX_B;

  let isAsmParserOnly = 1, hasSideEffects = 0 in {
    defm  rrib_alt  : AVX512_maskable_cmp_alt<0xC2, MRMSrcReg, _,
                         (outs _.KRC:$dst),
                         (ins _.RC:$src1, _.RC:$src2, u8imm:$cc),
                         "vcmp"#_.Suffix,
                         "$cc, {sae}, $src2, $src1",
                         "$src1, $src2, {sae}, $cc">, EVEX_B;
   }
}

multiclass avx512_vcmp<AVX512VLVectorVTInfo _> {
  let Predicates = [HasAVX512] in {
    defm Z    : avx512_vcmp_common<_.info512>,
                avx512_vcmp_sae<_.info512>, EVEX_V512;

  }
  let Predicates = [HasAVX512,HasVLX] in {
   defm Z128 : avx512_vcmp_common<_.info128>, EVEX_V128;
   defm Z256 : avx512_vcmp_common<_.info256>, EVEX_V256;
  }
}

defm VCMPPD : avx512_vcmp<avx512vl_f64_info>,
                          AVX512PDIi8Base, EVEX_4V, EVEX_CD8<64, CD8VF>, VEX_W;
defm VCMPPS : avx512_vcmp<avx512vl_f32_info>,
                          AVX512PSIi8Base, EVEX_4V, EVEX_CD8<32, CD8VF>;

def : Pat<(v8i1 (X86cmpm (v8f32 VR256X:$src1), (v8f32 VR256X:$src2), imm:$cc)),
          (COPY_TO_REGCLASS (VCMPPSZrri
            (v16f32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src1, sub_ymm)),
            (v16f32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src2, sub_ymm)),
            imm:$cc), VK8)>;
def : Pat<(v8i1 (X86cmpm (v8i32 VR256X:$src1), (v8i32 VR256X:$src2), imm:$cc)),
          (COPY_TO_REGCLASS (VPCMPDZrri
            (v16i32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src1, sub_ymm)),
            (v16i32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src2, sub_ymm)),
            imm:$cc), VK8)>;
def : Pat<(v8i1 (X86cmpmu (v8i32 VR256X:$src1), (v8i32 VR256X:$src2), imm:$cc)),
          (COPY_TO_REGCLASS (VPCMPUDZrri
            (v16i32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src1, sub_ymm)),
            (v16i32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src2, sub_ymm)),
            imm:$cc), VK8)>;

// ----------------------------------------------------------------
// FPClass
//handle fpclass instruction  mask =  op(reg_scalar,imm)
//                                    op(mem_scalar,imm)
multiclass avx512_scalar_fpclass<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                 X86VectorVTInfo _, Predicate prd> {
  let Predicates = [prd] in {
      def rr : AVX512<opc, MRMSrcReg, (outs _.KRC:$dst),//_.KRC:$dst),
                      (ins _.RC:$src1, i32u8imm:$src2),
                      OpcodeStr##_.Suffix#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set _.KRC:$dst,(OpNode (_.VT _.RC:$src1),
                              (i32 imm:$src2)))], NoItinerary>;
      def rrk : AVX512<opc, MRMSrcReg, (outs _.KRC:$dst),
                      (ins _.KRCWM:$mask, _.RC:$src1, i32u8imm:$src2),
                      OpcodeStr##_.Suffix#
                      "\t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}",
                      [(set _.KRC:$dst,(or _.KRCWM:$mask,
                                      (OpNode (_.VT _.RC:$src1),
                                      (i32 imm:$src2))))], NoItinerary>, EVEX_K;
    def rm : AVX512<opc, MRMSrcMem, (outs _.KRC:$dst),
                    (ins _.MemOp:$src1, i32u8imm:$src2),
                    OpcodeStr##_.Suffix##
                              "\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set _.KRC:$dst,
                          (OpNode (_.VT (bitconvert (_.LdFrag addr:$src1))),
                                  (i32 imm:$src2)))], NoItinerary>;
    def rmk : AVX512<opc, MRMSrcMem, (outs _.KRC:$dst),
                    (ins _.KRCWM:$mask, _.MemOp:$src1, i32u8imm:$src2),
                    OpcodeStr##_.Suffix##
                    "\t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}",
                    [(set _.KRC:$dst,(or _.KRCWM:$mask,
                        (OpNode (_.VT (bitconvert (_.LdFrag addr:$src1))),
                            (i32 imm:$src2))))], NoItinerary>, EVEX_K;
  }
}

//handle fpclass instruction mask = fpclass(reg_vec, reg_vec, imm)
//                                  fpclass(reg_vec, mem_vec, imm)
//                                  fpclass(reg_vec, broadcast(eltVt), imm)
multiclass avx512_vector_fpclass<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                 X86VectorVTInfo _, string mem, string broadcast>{
  def rr : AVX512<opc, MRMSrcReg, (outs _.KRC:$dst),
                      (ins _.RC:$src1, i32u8imm:$src2),
                      OpcodeStr##_.Suffix#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set _.KRC:$dst,(OpNode (_.VT _.RC:$src1),
                                       (i32 imm:$src2)))], NoItinerary>;
  def rrk : AVX512<opc, MRMSrcReg, (outs _.KRC:$dst),
                      (ins _.KRCWM:$mask, _.RC:$src1, i32u8imm:$src2),
                      OpcodeStr##_.Suffix#
                      "\t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}",
                      [(set _.KRC:$dst,(or _.KRCWM:$mask,
                                       (OpNode (_.VT _.RC:$src1),
                                       (i32 imm:$src2))))], NoItinerary>, EVEX_K;
  def rm : AVX512<opc, MRMSrcMem, (outs _.KRC:$dst),
                    (ins _.MemOp:$src1, i32u8imm:$src2),
                    OpcodeStr##_.Suffix##mem#
                    "\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                    [(set _.KRC:$dst,(OpNode
                                     (_.VT (bitconvert (_.LdFrag addr:$src1))),
                                     (i32 imm:$src2)))], NoItinerary>;
  def rmk : AVX512<opc, MRMSrcMem, (outs _.KRC:$dst),
                    (ins _.KRCWM:$mask, _.MemOp:$src1, i32u8imm:$src2),
                    OpcodeStr##_.Suffix##mem#
                    "\t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}",
                    [(set _.KRC:$dst, (or _.KRCWM:$mask, (OpNode
                                  (_.VT (bitconvert (_.LdFrag addr:$src1))),
                                  (i32 imm:$src2))))], NoItinerary>, EVEX_K;
  def rmb : AVX512<opc, MRMSrcMem, (outs _.KRC:$dst),
                    (ins _.ScalarMemOp:$src1, i32u8imm:$src2),
                    OpcodeStr##_.Suffix##broadcast##"\t{$src2, ${src1}"##
                                      _.BroadcastStr##", $dst|$dst, ${src1}"
                                                  ##_.BroadcastStr##", $src2}",
                    [(set _.KRC:$dst,(OpNode
                                     (_.VT (X86VBroadcast
                                           (_.ScalarLdFrag addr:$src1))),
                                     (i32 imm:$src2)))], NoItinerary>,EVEX_B;
  def rmbk : AVX512<opc, MRMSrcMem, (outs _.KRC:$dst),
                    (ins _.KRCWM:$mask, _.ScalarMemOp:$src1, i32u8imm:$src2),
                    OpcodeStr##_.Suffix##broadcast##"\t{$src2, ${src1}"##
                          _.BroadcastStr##", $dst {${mask}}|$dst {${mask}}, ${src1}"##
                                                   _.BroadcastStr##", $src2}",
                    [(set _.KRC:$dst,(or _.KRCWM:$mask, (OpNode
                                     (_.VT (X86VBroadcast
                                           (_.ScalarLdFrag addr:$src1))),
                                     (i32 imm:$src2))))], NoItinerary>,
                                                          EVEX_B, EVEX_K;
}

multiclass avx512_vector_fpclass_all<string OpcodeStr,
            AVX512VLVectorVTInfo _, bits<8> opc, SDNode OpNode, Predicate prd,
                                                              string broadcast>{
  let Predicates = [prd] in {
    defm Z    : avx512_vector_fpclass<opc, OpcodeStr, OpNode, _.info512, "{z}",
                                      broadcast>, EVEX_V512;
  }
  let Predicates = [prd, HasVLX] in {
    defm Z128 : avx512_vector_fpclass<opc, OpcodeStr, OpNode, _.info128, "{x}",
                                      broadcast>, EVEX_V128;
    defm Z256 : avx512_vector_fpclass<opc, OpcodeStr, OpNode, _.info256, "{y}",
                                      broadcast>, EVEX_V256;
  }
}

multiclass avx512_fp_fpclass_all<string OpcodeStr, bits<8> opcVec,
             bits<8> opcScalar, SDNode VecOpNode, SDNode ScalarOpNode, Predicate prd>{
  defm PS : avx512_vector_fpclass_all<OpcodeStr,  avx512vl_f32_info, opcVec,
                                      VecOpNode, prd, "{l}">, EVEX_CD8<32, CD8VF>;
  defm PD : avx512_vector_fpclass_all<OpcodeStr,  avx512vl_f64_info, opcVec,
                                      VecOpNode, prd, "{q}">,EVEX_CD8<64, CD8VF> , VEX_W;
  defm SS : avx512_scalar_fpclass<opcScalar, OpcodeStr, ScalarOpNode,
                                      f32x_info, prd>, EVEX_CD8<32, CD8VT1>;
  defm SD : avx512_scalar_fpclass<opcScalar, OpcodeStr, ScalarOpNode,
                                      f64x_info, prd>, EVEX_CD8<64, CD8VT1>, VEX_W;
}

defm VFPCLASS : avx512_fp_fpclass_all<"vfpclass", 0x66, 0x67, X86Vfpclass,
                                      X86Vfpclasss, HasDQI>, AVX512AIi8Base,EVEX;

//-----------------------------------------------------------------
// Mask register copy, including
// - copy between mask registers
// - load/store mask registers
// - copy from GPR to mask register and vice versa
//
multiclass avx512_mask_mov<bits<8> opc_kk, bits<8> opc_km, bits<8> opc_mk,
                         string OpcodeStr, RegisterClass KRC,
                         ValueType vvt, X86MemOperand x86memop> {
  let hasSideEffects = 0 in
  def kk : I<opc_kk, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src),
             !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), []>;
  def km : I<opc_km, MRMSrcMem, (outs KRC:$dst), (ins x86memop:$src),
             !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
             [(set KRC:$dst, (vvt (load addr:$src)))]>;
  def mk : I<opc_mk, MRMDestMem, (outs), (ins x86memop:$dst, KRC:$src),
             !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
             [(store KRC:$src, addr:$dst)]>;
}

multiclass avx512_mask_mov_gpr<bits<8> opc_kr, bits<8> opc_rk,
                             string OpcodeStr,
                             RegisterClass KRC, RegisterClass GRC> {
  let hasSideEffects = 0 in {
    def kr : I<opc_kr, MRMSrcReg, (outs KRC:$dst), (ins GRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), []>;
    def rk : I<opc_rk, MRMSrcReg, (outs GRC:$dst), (ins KRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), []>;
  }
}

let Predicates = [HasDQI] in
  defm KMOVB : avx512_mask_mov<0x90, 0x90, 0x91, "kmovb", VK8, v8i1, i8mem>,
               avx512_mask_mov_gpr<0x92, 0x93, "kmovb", VK8, GR32>,
               VEX, PD;

let Predicates = [HasAVX512] in
  defm KMOVW : avx512_mask_mov<0x90, 0x90, 0x91, "kmovw", VK16, v16i1, i16mem>,
               avx512_mask_mov_gpr<0x92, 0x93, "kmovw", VK16, GR32>,
               VEX, PS;

let Predicates = [HasBWI] in {
  defm KMOVD : avx512_mask_mov<0x90, 0x90, 0x91, "kmovd", VK32, v32i1,i32mem>,
               VEX, PD, VEX_W;
  defm KMOVD : avx512_mask_mov_gpr<0x92, 0x93, "kmovd", VK32, GR32>,
               VEX, XD;
  defm KMOVQ : avx512_mask_mov<0x90, 0x90, 0x91, "kmovq", VK64, v64i1, i64mem>,
               VEX, PS, VEX_W;
  defm KMOVQ : avx512_mask_mov_gpr<0x92, 0x93, "kmovq", VK64, GR64>,
               VEX, XD, VEX_W;
}

// GR from/to mask register
def : Pat<(v16i1 (bitconvert (i16 GR16:$src))),
          (COPY_TO_REGCLASS GR16:$src, VK16)>;
def : Pat<(i16 (bitconvert (v16i1 VK16:$src))),
          (COPY_TO_REGCLASS VK16:$src, GR16)>;

def : Pat<(v8i1 (bitconvert (i8 GR8:$src))),
          (COPY_TO_REGCLASS GR8:$src, VK8)>;
def : Pat<(i8 (bitconvert (v8i1 VK8:$src))),
          (COPY_TO_REGCLASS VK8:$src, GR8)>;

def : Pat<(i32 (zext (i16 (bitconvert (v16i1 VK16:$src))))),
          (KMOVWrk VK16:$src)>;
def : Pat<(i32 (anyext (i16 (bitconvert (v16i1 VK16:$src))))),
          (i32 (INSERT_SUBREG (IMPLICIT_DEF),
                (i16 (COPY_TO_REGCLASS VK16:$src, GR16)), sub_16bit))>;

def : Pat<(i32 (zext (i8 (bitconvert (v8i1 VK8:$src))))),
          (MOVZX32rr8 (COPY_TO_REGCLASS VK8:$src, GR8))>, Requires<[NoDQI]>;
def : Pat<(i32 (zext (i8 (bitconvert (v8i1 VK8:$src))))),
          (KMOVBrk VK8:$src)>, Requires<[HasDQI]>;
def : Pat<(i32 (anyext (i8 (bitconvert (v8i1 VK8:$src))))),
          (i32 (INSERT_SUBREG (IMPLICIT_DEF),
                (i8 (COPY_TO_REGCLASS VK8:$src, GR8)), sub_8bit))>;

def : Pat<(v32i1 (bitconvert (i32 GR32:$src))),
          (COPY_TO_REGCLASS GR32:$src, VK32)>;
def : Pat<(i32 (bitconvert (v32i1 VK32:$src))),
          (COPY_TO_REGCLASS VK32:$src, GR32)>;
def : Pat<(v64i1 (bitconvert (i64 GR64:$src))),
          (COPY_TO_REGCLASS GR64:$src, VK64)>;
def : Pat<(i64 (bitconvert (v64i1 VK64:$src))),
          (COPY_TO_REGCLASS VK64:$src, GR64)>;

// Load/store kreg
let Predicates = [HasDQI] in {
  def : Pat<(store (i8 (bitconvert (v8i1 VK8:$src))), addr:$dst),
            (KMOVBmk addr:$dst, VK8:$src)>;
  def : Pat<(v8i1 (bitconvert (i8 (load addr:$src)))),
            (KMOVBkm addr:$src)>;

  def : Pat<(store VK4:$src, addr:$dst),
            (KMOVBmk addr:$dst, (COPY_TO_REGCLASS VK4:$src, VK8))>;
  def : Pat<(store VK2:$src, addr:$dst),
            (KMOVBmk addr:$dst, (COPY_TO_REGCLASS VK2:$src, VK8))>;
  def : Pat<(store VK1:$src, addr:$dst),
            (KMOVBmk addr:$dst, (COPY_TO_REGCLASS VK1:$src, VK8))>;

  def : Pat<(v2i1 (load addr:$src)),
            (COPY_TO_REGCLASS (KMOVBkm addr:$src), VK2)>;
  def : Pat<(v4i1 (load addr:$src)),
            (COPY_TO_REGCLASS (KMOVBkm addr:$src), VK4)>;
}
let Predicates = [HasAVX512, NoDQI] in {
  def : Pat<(store VK1:$src, addr:$dst),
            (MOV8mr addr:$dst,
             (EXTRACT_SUBREG (KMOVWrk (COPY_TO_REGCLASS VK1:$src, VK16)),
              sub_8bit))>;
  def : Pat<(store VK2:$src, addr:$dst),
            (MOV8mr addr:$dst,
             (EXTRACT_SUBREG (KMOVWrk (COPY_TO_REGCLASS VK2:$src, VK16)),
              sub_8bit))>;
  def : Pat<(store VK4:$src, addr:$dst),
            (MOV8mr addr:$dst,
             (EXTRACT_SUBREG (KMOVWrk (COPY_TO_REGCLASS VK4:$src, VK16)),
              sub_8bit))>;
  def : Pat<(store VK8:$src, addr:$dst),
            (MOV8mr addr:$dst,
             (EXTRACT_SUBREG (KMOVWrk (COPY_TO_REGCLASS VK8:$src, VK16)),
              sub_8bit))>;

  def : Pat<(v8i1 (load addr:$src)),
            (COPY_TO_REGCLASS (MOVZX32rm8 addr:$src), VK8)>;
  def : Pat<(v2i1 (load addr:$src)),
            (COPY_TO_REGCLASS (MOVZX32rm8 addr:$src), VK2)>;
  def : Pat<(v4i1 (load addr:$src)),
            (COPY_TO_REGCLASS (MOVZX32rm8 addr:$src), VK4)>;
}

let Predicates = [HasAVX512] in {
  def : Pat<(store (i16 (bitconvert (v16i1 VK16:$src))), addr:$dst),
            (KMOVWmk addr:$dst, VK16:$src)>;
  def : Pat<(i1 (load addr:$src)),
            (COPY_TO_REGCLASS (AND32ri8 (MOVZX32rm8 addr:$src), (i32 1)), VK1)>;
  def : Pat<(v16i1 (bitconvert (i16 (load addr:$src)))),
            (KMOVWkm addr:$src)>;
}
let Predicates = [HasBWI] in {
  def : Pat<(store (i32 (bitconvert (v32i1 VK32:$src))), addr:$dst),
            (KMOVDmk addr:$dst, VK32:$src)>;
  def : Pat<(v32i1 (bitconvert (i32 (load addr:$src)))),
            (KMOVDkm addr:$src)>;
  def : Pat<(store (i64 (bitconvert (v64i1 VK64:$src))), addr:$dst),
            (KMOVQmk addr:$dst, VK64:$src)>;
  def : Pat<(v64i1 (bitconvert (i64 (load addr:$src)))),
            (KMOVQkm addr:$src)>;
}

let Predicates = [HasAVX512] in {
  def : Pat<(i1 (trunc (i64 GR64:$src))),
            (COPY_TO_REGCLASS (KMOVWkr (AND32ri8 (EXTRACT_SUBREG $src, sub_32bit),
                                        (i32 1))), VK1)>;

  def : Pat<(i1 (trunc (i32 GR32:$src))),
            (COPY_TO_REGCLASS (KMOVWkr (AND32ri8 $src, (i32 1))), VK1)>;

  def : Pat<(i1 (trunc (i32 (assertzext_i1 GR32:$src)))),
            (COPY_TO_REGCLASS GR32:$src, VK1)>;

  def : Pat<(i1 (trunc (i8 GR8:$src))),
       (COPY_TO_REGCLASS
        (KMOVWkr (AND32ri8 (INSERT_SUBREG (i32 (IMPLICIT_DEF)),
                                          GR8:$src, sub_8bit), (i32 1))),
       VK1)>;

  def : Pat<(i1 (trunc (i16 GR16:$src))),
       (COPY_TO_REGCLASS
        (KMOVWkr (AND32ri8 (INSERT_SUBREG (i32 (IMPLICIT_DEF)),
                                          GR16:$src, sub_16bit), (i32 1))),
       VK1)>;

  def : Pat<(i32 (zext VK1:$src)),
            (AND32ri8 (KMOVWrk (COPY_TO_REGCLASS VK1:$src, VK16)), (i32 1))>;

  def : Pat<(i32 (anyext VK1:$src)),
            (COPY_TO_REGCLASS VK1:$src, GR32)>;

  def : Pat<(i8 (zext VK1:$src)),
            (EXTRACT_SUBREG
             (AND32ri8 (KMOVWrk
                        (COPY_TO_REGCLASS VK1:$src, VK16)), (i32 1)), sub_8bit)>;

  def : Pat<(i8 (anyext VK1:$src)),
            (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS VK1:$src, GR32)), sub_8bit)>;

  def : Pat<(i64 (zext VK1:$src)),
            (AND64ri8 (SUBREG_TO_REG (i64 0),
             (KMOVWrk (COPY_TO_REGCLASS VK1:$src, VK16)), sub_32bit), (i64 1))>;

  def : Pat<(i64 (anyext VK1:$src)),
            (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
             (i32 (COPY_TO_REGCLASS VK1:$src, GR32)), sub_32bit)>;

  def : Pat<(i16 (zext VK1:$src)),
            (EXTRACT_SUBREG
             (AND32ri8 (KMOVWrk (COPY_TO_REGCLASS VK1:$src, VK16)), (i32 1)),
              sub_16bit)>;

  def : Pat<(i16 (anyext VK1:$src)),
            (EXTRACT_SUBREG (i32 (COPY_TO_REGCLASS VK1:$src, GR32)), sub_16bit)>;
}
def : Pat<(v16i1 (scalar_to_vector VK1:$src)),
          (COPY_TO_REGCLASS VK1:$src, VK16)>;
def : Pat<(v8i1 (scalar_to_vector VK1:$src)),
          (COPY_TO_REGCLASS VK1:$src, VK8)>;
def : Pat<(v4i1 (scalar_to_vector VK1:$src)),
          (COPY_TO_REGCLASS VK1:$src, VK4)>;
def : Pat<(v2i1 (scalar_to_vector VK1:$src)),
          (COPY_TO_REGCLASS VK1:$src, VK2)>;
def : Pat<(v32i1 (scalar_to_vector VK1:$src)),
          (COPY_TO_REGCLASS VK1:$src, VK32)>;
def : Pat<(v64i1 (scalar_to_vector VK1:$src)),
          (COPY_TO_REGCLASS VK1:$src, VK64)>;

def : Pat<(store (i1 -1), addr:$dst), (MOV8mi addr:$dst, (i8 1))>;
def : Pat<(store (i1  1), addr:$dst), (MOV8mi addr:$dst, (i8 1))>;
def : Pat<(store (i1  0), addr:$dst), (MOV8mi addr:$dst, (i8 0))>;

def : Pat<(i1 (X86Vextract VK64:$src, (iPTR 0))), (COPY_TO_REGCLASS VK64:$src, VK1)>;
def : Pat<(i1 (X86Vextract VK32:$src, (iPTR 0))), (COPY_TO_REGCLASS VK32:$src, VK1)>;
def : Pat<(i1 (X86Vextract VK16:$src, (iPTR 0))), (COPY_TO_REGCLASS VK16:$src, VK1)>;
def : Pat<(i1 (X86Vextract VK8:$src,  (iPTR 0))), (COPY_TO_REGCLASS VK8:$src,  VK1)>;
def : Pat<(i1 (X86Vextract VK4:$src,  (iPTR 0))), (COPY_TO_REGCLASS VK4:$src,  VK1)>;
def : Pat<(i1 (X86Vextract VK2:$src,  (iPTR 0))), (COPY_TO_REGCLASS VK2:$src,  VK1)>;

// Mask unary operation
// - KNOT
multiclass avx512_mask_unop<bits<8> opc, string OpcodeStr,
                            RegisterClass KRC, SDPatternOperator OpNode,
                            Predicate prd> {
  let Predicates = [prd] in
    def rr : I<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src),
               !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
               [(set KRC:$dst, (OpNode KRC:$src))]>;
}

multiclass avx512_mask_unop_all<bits<8> opc, string OpcodeStr,
                                SDPatternOperator OpNode> {
  defm B : avx512_mask_unop<opc, !strconcat(OpcodeStr, "b"), VK8, OpNode,
                            HasDQI>, VEX, PD;
  defm W : avx512_mask_unop<opc, !strconcat(OpcodeStr, "w"), VK16, OpNode,
                            HasAVX512>, VEX, PS;
  defm D : avx512_mask_unop<opc, !strconcat(OpcodeStr, "d"), VK32, OpNode,
                            HasBWI>, VEX, PD, VEX_W;
  defm Q : avx512_mask_unop<opc, !strconcat(OpcodeStr, "q"), VK64, OpNode,
                            HasBWI>, VEX, PS, VEX_W;
}

defm KNOT : avx512_mask_unop_all<0x44, "knot", vnot>;

// KNL does not support KMOVB, 8-bit mask is promoted to 16-bit
let Predicates = [HasAVX512, NoDQI] in
def : Pat<(vnot VK8:$src),
          (COPY_TO_REGCLASS (KNOTWrr (COPY_TO_REGCLASS VK8:$src, VK16)), VK8)>;

def : Pat<(vnot VK4:$src),
          (COPY_TO_REGCLASS (KNOTWrr (COPY_TO_REGCLASS VK4:$src, VK16)), VK4)>;
def : Pat<(vnot VK2:$src),
          (COPY_TO_REGCLASS (KNOTWrr (COPY_TO_REGCLASS VK2:$src, VK16)), VK2)>;

// Mask binary operation
// - KAND, KANDN, KOR, KXNOR, KXOR
multiclass avx512_mask_binop<bits<8> opc, string OpcodeStr,
                           RegisterClass KRC, SDPatternOperator OpNode,
                           Predicate prd, bit IsCommutable> {
  let Predicates = [prd], isCommutable = IsCommutable in
    def rr : I<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src1, KRC:$src2),
               !strconcat(OpcodeStr,
                          "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
               [(set KRC:$dst, (OpNode KRC:$src1, KRC:$src2))]>;
}

multiclass avx512_mask_binop_all<bits<8> opc, string OpcodeStr,
                               SDPatternOperator OpNode, bit IsCommutable,
                               Predicate prdW = HasAVX512> {
  defm B : avx512_mask_binop<opc, !strconcat(OpcodeStr, "b"), VK8, OpNode,
                             HasDQI, IsCommutable>, VEX_4V, VEX_L, PD;
  defm W : avx512_mask_binop<opc, !strconcat(OpcodeStr, "w"), VK16, OpNode,
                             prdW, IsCommutable>, VEX_4V, VEX_L, PS;
  defm D : avx512_mask_binop<opc, !strconcat(OpcodeStr, "d"), VK32, OpNode,
                             HasBWI, IsCommutable>, VEX_4V, VEX_L, VEX_W, PD;
  defm Q : avx512_mask_binop<opc, !strconcat(OpcodeStr, "q"), VK64, OpNode,
                             HasBWI, IsCommutable>, VEX_4V, VEX_L, VEX_W, PS;
}

def andn : PatFrag<(ops node:$i0, node:$i1), (and (not node:$i0), node:$i1)>;
def xnor : PatFrag<(ops node:$i0, node:$i1), (not (xor node:$i0, node:$i1))>;
// These nodes use 'vnot' instead of 'not' to support vectors.
def vandn : PatFrag<(ops node:$i0, node:$i1), (and (vnot node:$i0), node:$i1)>;
def vxnor : PatFrag<(ops node:$i0, node:$i1), (vnot (xor node:$i0, node:$i1))>;

defm KAND  : avx512_mask_binop_all<0x41, "kand",  and,   1>;
defm KOR   : avx512_mask_binop_all<0x45, "kor",   or,    1>;
defm KXNOR : avx512_mask_binop_all<0x46, "kxnor", vxnor, 1>;
defm KXOR  : avx512_mask_binop_all<0x47, "kxor",  xor,   1>;
defm KANDN : avx512_mask_binop_all<0x42, "kandn", vandn, 0>;
defm KADD  : avx512_mask_binop_all<0x4A, "kadd",  add,   1, HasDQI>;

multiclass avx512_binop_pat<SDPatternOperator VOpNode, SDPatternOperator OpNode,
                            Instruction Inst> {
  // With AVX512F, 8-bit mask is promoted to 16-bit mask,
  // for the DQI set, this type is legal and KxxxB instruction is used
  let Predicates = [NoDQI] in
  def : Pat<(VOpNode VK8:$src1, VK8:$src2),
            (COPY_TO_REGCLASS
              (Inst (COPY_TO_REGCLASS VK8:$src1, VK16),
                    (COPY_TO_REGCLASS VK8:$src2, VK16)), VK8)>;

  // All types smaller than 8 bits require conversion anyway
  def : Pat<(OpNode VK1:$src1, VK1:$src2),
        (COPY_TO_REGCLASS (Inst
                           (COPY_TO_REGCLASS VK1:$src1, VK16),
                           (COPY_TO_REGCLASS VK1:$src2, VK16)), VK1)>;
  def : Pat<(VOpNode VK2:$src1, VK2:$src2),
        (COPY_TO_REGCLASS (Inst
                           (COPY_TO_REGCLASS VK2:$src1, VK16),
                           (COPY_TO_REGCLASS VK2:$src2, VK16)), VK1)>;
  def : Pat<(VOpNode VK4:$src1, VK4:$src2),
        (COPY_TO_REGCLASS (Inst
                           (COPY_TO_REGCLASS VK4:$src1, VK16),
                           (COPY_TO_REGCLASS VK4:$src2, VK16)), VK1)>;
}

defm : avx512_binop_pat<and,   and,  KANDWrr>;
defm : avx512_binop_pat<vandn, andn, KANDNWrr>;
defm : avx512_binop_pat<or,    or,   KORWrr>;
defm : avx512_binop_pat<vxnor, xnor, KXNORWrr>;
defm : avx512_binop_pat<xor,   xor,  KXORWrr>;

// Mask unpacking
multiclass avx512_mask_unpck<string Suffix,RegisterClass KRC, ValueType VT,
                             RegisterClass KRCSrc, Predicate prd> {
  let Predicates = [prd] in {
    let hasSideEffects = 0 in
    def rr : I<0x4b, MRMSrcReg, (outs KRC:$dst),
               (ins KRC:$src1, KRC:$src2),
               "kunpck"#Suffix#"\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>,
               VEX_4V, VEX_L;

    def : Pat<(VT (concat_vectors KRCSrc:$src1, KRCSrc:$src2)),
              (!cast<Instruction>(NAME##rr)
                        (COPY_TO_REGCLASS KRCSrc:$src2, KRC),
                        (COPY_TO_REGCLASS KRCSrc:$src1, KRC))>;
  }
}

defm KUNPCKBW : avx512_mask_unpck<"bw", VK16, v16i1, VK8, HasAVX512>, PD;
defm KUNPCKWD : avx512_mask_unpck<"wd", VK32, v32i1, VK16, HasBWI>, PS;
defm KUNPCKDQ : avx512_mask_unpck<"dq", VK64, v64i1, VK32, HasBWI>, PS, VEX_W;

// Mask bit testing
multiclass avx512_mask_testop<bits<8> opc, string OpcodeStr, RegisterClass KRC,
                              SDNode OpNode, Predicate prd> {
  let Predicates = [prd], Defs = [EFLAGS] in
    def rr : I<opc, MRMSrcReg, (outs), (ins KRC:$src1, KRC:$src2),
               !strconcat(OpcodeStr, "\t{$src2, $src1|$src1, $src2}"),
               [(set EFLAGS, (OpNode KRC:$src1, KRC:$src2))]>;
}

multiclass avx512_mask_testop_w<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                Predicate prdW = HasAVX512> {
  defm B : avx512_mask_testop<opc, OpcodeStr#"b", VK8, OpNode, HasDQI>,
                                                                VEX, PD;
  defm W : avx512_mask_testop<opc, OpcodeStr#"w", VK16, OpNode, prdW>,
                                                                VEX, PS;
  defm Q : avx512_mask_testop<opc, OpcodeStr#"q", VK64, OpNode, HasBWI>,
                                                                VEX, PS, VEX_W;
  defm D : avx512_mask_testop<opc, OpcodeStr#"d", VK32, OpNode, HasBWI>,
                                                                VEX, PD, VEX_W;
}

defm KORTEST : avx512_mask_testop_w<0x98, "kortest", X86kortest>;
defm KTEST   : avx512_mask_testop_w<0x99, "ktest", X86ktest, HasDQI>;

// Mask shift
multiclass avx512_mask_shiftop<bits<8> opc, string OpcodeStr, RegisterClass KRC,
                             SDNode OpNode> {
  let Predicates = [HasAVX512] in
    def ri : Ii8<opc, MRMSrcReg, (outs KRC:$dst), (ins KRC:$src, u8imm:$imm),
                 !strconcat(OpcodeStr,
                            "\t{$imm, $src, $dst|$dst, $src, $imm}"),
                            [(set KRC:$dst, (OpNode KRC:$src, (i8 imm:$imm)))]>;
}

multiclass avx512_mask_shiftop_w<bits<8> opc1, bits<8> opc2, string OpcodeStr,
                               SDNode OpNode> {
  defm W : avx512_mask_shiftop<opc1, !strconcat(OpcodeStr, "w"), VK16, OpNode>,
                               VEX, TAPD, VEX_W;
  let Predicates = [HasDQI] in
  defm B : avx512_mask_shiftop<opc1, !strconcat(OpcodeStr, "b"), VK8, OpNode>,
                               VEX, TAPD;
  let Predicates = [HasBWI] in {
  defm Q : avx512_mask_shiftop<opc2, !strconcat(OpcodeStr, "q"), VK64, OpNode>,
                               VEX, TAPD, VEX_W;
  defm D : avx512_mask_shiftop<opc2, !strconcat(OpcodeStr, "d"), VK32, OpNode>,
                               VEX, TAPD;
  }
}

defm KSHIFTL : avx512_mask_shiftop_w<0x32, 0x33, "kshiftl", X86kshiftl>;
defm KSHIFTR : avx512_mask_shiftop_w<0x30, 0x31, "kshiftr", X86kshiftr>;

// Mask setting all 0s or 1s
multiclass avx512_mask_setop<RegisterClass KRC, ValueType VT, PatFrag Val> {
  let Predicates = [HasAVX512] in
    let isReMaterializable = 1, isAsCheapAsAMove = 1, isPseudo = 1 in
      def #NAME# : I<0, Pseudo, (outs KRC:$dst), (ins), "",
                     [(set KRC:$dst, (VT Val))]>;
}

multiclass avx512_mask_setop_w<PatFrag Val> {
  defm W : avx512_mask_setop<VK16, v16i1, Val>;
  defm D : avx512_mask_setop<VK32,  v32i1, Val>;
  defm Q : avx512_mask_setop<VK64, v64i1, Val>;
}

defm KSET0 : avx512_mask_setop_w<immAllZerosV>;
defm KSET1 : avx512_mask_setop_w<immAllOnesV>;

// With AVX-512 only, 8-bit mask is promoted to 16-bit mask.
let Predicates = [HasAVX512] in {
  def : Pat<(v8i1 immAllZerosV), (COPY_TO_REGCLASS (KSET0W), VK8)>;
  def : Pat<(v4i1 immAllZerosV), (COPY_TO_REGCLASS (KSET0W), VK4)>;
  def : Pat<(v2i1 immAllZerosV), (COPY_TO_REGCLASS (KSET0W), VK2)>;
  def : Pat<(v8i1 immAllOnesV),  (COPY_TO_REGCLASS (KSET1W), VK8)>;
  def : Pat<(v4i1 immAllOnesV),  (COPY_TO_REGCLASS (KSET1W), VK4)>;
  def : Pat<(v2i1 immAllOnesV),  (COPY_TO_REGCLASS (KSET1W), VK2)>;
  let AddedComplexity = 10 in { // To optimize isel table.
    def : Pat<(i1 0), (COPY_TO_REGCLASS (KSET0W), VK1)>;
    def : Pat<(i1 1), (COPY_TO_REGCLASS (KSHIFTRWri (KSET1W), (i8 15)), VK1)>;
    def : Pat<(i1 -1), (COPY_TO_REGCLASS (KSHIFTRWri (KSET1W), (i8 15)), VK1)>;
  }
}

// Patterns for kmask insert_subvector/extract_subvector to/from index=0
multiclass operation_subvector_mask_lowering<RegisterClass subRC, ValueType subVT,
                                             RegisterClass RC, ValueType VT> {
  def : Pat<(subVT (extract_subvector (VT RC:$src), (iPTR 0))),
            (subVT (COPY_TO_REGCLASS RC:$src, subRC))>;

  def : Pat<(VT (insert_subvector undef, subRC:$src, (iPTR 0))),
            (VT (COPY_TO_REGCLASS subRC:$src, RC))>;
}

defm : operation_subvector_mask_lowering<VK2,  v2i1,  VK4,  v4i1>;
defm : operation_subvector_mask_lowering<VK2,  v2i1,  VK8,  v8i1>;
defm : operation_subvector_mask_lowering<VK2,  v2i1,  VK16, v16i1>;
defm : operation_subvector_mask_lowering<VK2,  v2i1,  VK32, v32i1>;
defm : operation_subvector_mask_lowering<VK2,  v2i1,  VK64, v64i1>;

defm : operation_subvector_mask_lowering<VK4,  v4i1,  VK8,  v8i1>;
defm : operation_subvector_mask_lowering<VK4,  v4i1,  VK16, v16i1>;
defm : operation_subvector_mask_lowering<VK4,  v4i1,  VK32, v32i1>;
defm : operation_subvector_mask_lowering<VK4,  v4i1,  VK64, v64i1>;

defm : operation_subvector_mask_lowering<VK8,  v8i1,  VK16, v16i1>;
defm : operation_subvector_mask_lowering<VK8,  v8i1,  VK32, v32i1>;
defm : operation_subvector_mask_lowering<VK8,  v8i1,  VK64, v64i1>;

defm : operation_subvector_mask_lowering<VK16, v16i1, VK32, v32i1>;
defm : operation_subvector_mask_lowering<VK16, v16i1, VK64, v64i1>;

defm : operation_subvector_mask_lowering<VK32, v32i1, VK64, v64i1>;

def : Pat<(v2i1 (extract_subvector (v4i1 VK4:$src), (iPTR 2))),
          (v2i1 (COPY_TO_REGCLASS
                  (KSHIFTRWri (COPY_TO_REGCLASS VK4:$src, VK16), (i8 2)),
                   VK2))>;
def : Pat<(v4i1 (extract_subvector (v8i1 VK8:$src), (iPTR 4))),
          (v4i1 (COPY_TO_REGCLASS
                  (KSHIFTRWri (COPY_TO_REGCLASS VK8:$src, VK16), (i8 4)),
                   VK4))>;
def : Pat<(v8i1 (extract_subvector (v16i1 VK16:$src), (iPTR 8))),
          (v8i1 (COPY_TO_REGCLASS (KSHIFTRWri VK16:$src, (i8 8)), VK8))>;
def : Pat<(v16i1 (extract_subvector (v32i1 VK32:$src), (iPTR 16))),
          (v16i1 (COPY_TO_REGCLASS (KSHIFTRDri VK32:$src, (i8 16)), VK16))>;
def : Pat<(v32i1 (extract_subvector (v64i1 VK64:$src), (iPTR 32))),
          (v32i1 (COPY_TO_REGCLASS (KSHIFTRQri VK64:$src, (i8 32)), VK32))>;


// Patterns for kmask shift
multiclass mask_shift_lowering<RegisterClass RC, ValueType VT> {
  def : Pat<(VT (X86kshiftl RC:$src, (i8 imm:$imm))),
            (VT (COPY_TO_REGCLASS
                   (KSHIFTLWri (COPY_TO_REGCLASS RC:$src, VK16),
                               (I8Imm $imm)),
                   RC))>;
  def : Pat<(VT (X86kshiftr RC:$src, (i8 imm:$imm))),
            (VT (COPY_TO_REGCLASS
                   (KSHIFTRWri (COPY_TO_REGCLASS RC:$src, VK16),
                               (I8Imm $imm)),
                   RC))>;
}

defm : mask_shift_lowering<VK8, v8i1>, Requires<[HasAVX512, NoDQI]>;
defm : mask_shift_lowering<VK4, v4i1>, Requires<[HasAVX512]>;
defm : mask_shift_lowering<VK2, v2i1>, Requires<[HasAVX512]>;
//===----------------------------------------------------------------------===//
// AVX-512 - Aligned and unaligned load and store
//


multiclass avx512_load<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                         PatFrag ld_frag, PatFrag mload,
                         SDPatternOperator SelectOprr = vselect> {
  let hasSideEffects = 0 in {
  def rr : AVX512PI<opc, MRMSrcReg, (outs _.RC:$dst), (ins _.RC:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"), [],
                    _.ExeDomain>, EVEX;
  def rrkz : AVX512PI<opc, MRMSrcReg, (outs _.RC:$dst),
                      (ins _.KRCWM:$mask,  _.RC:$src),
                      !strconcat(OpcodeStr, "\t{$src, ${dst} {${mask}} {z}|",
                       "${dst} {${mask}} {z}, $src}"),
                       [(set _.RC:$dst, (_.VT (SelectOprr _.KRCWM:$mask,
                                           (_.VT _.RC:$src),
                                           _.ImmAllZerosV)))], _.ExeDomain>,
                       EVEX, EVEX_KZ;

  let canFoldAsLoad = 1, isReMaterializable = 1,
      SchedRW = [WriteLoad] in
  def rm : AVX512PI<opc, MRMSrcMem, (outs _.RC:$dst), (ins _.MemOp:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(set _.RC:$dst, (_.VT (bitconvert (ld_frag addr:$src))))],
                    _.ExeDomain>, EVEX;

  let Constraints = "$src0 = $dst", isConvertibleToThreeAddress = 1 in {
  def rrk : AVX512PI<opc, MRMSrcReg, (outs _.RC:$dst),
                    (ins _.RC:$src0, _.KRCWM:$mask, _.RC:$src1),
                    !strconcat(OpcodeStr, "\t{$src1, ${dst} {${mask}}|",
                    "${dst} {${mask}}, $src1}"),
                    [(set _.RC:$dst, (_.VT (SelectOprr _.KRCWM:$mask,
                                        (_.VT _.RC:$src1),
                                        (_.VT _.RC:$src0))))], _.ExeDomain>,
                     EVEX, EVEX_K;
    let SchedRW = [WriteLoad] in
    def rmk : AVX512PI<opc, MRMSrcMem, (outs _.RC:$dst),
                     (ins _.RC:$src0, _.KRCWM:$mask, _.MemOp:$src1),
                     !strconcat(OpcodeStr, "\t{$src1, ${dst} {${mask}}|",
                      "${dst} {${mask}}, $src1}"),
                     [(set _.RC:$dst, (_.VT
                         (vselect _.KRCWM:$mask,
                          (_.VT (bitconvert (ld_frag addr:$src1))),
                           (_.VT _.RC:$src0))))], _.ExeDomain>, EVEX, EVEX_K;
  }
  let SchedRW = [WriteLoad] in
  def rmkz : AVX512PI<opc, MRMSrcMem, (outs _.RC:$dst),
                  (ins _.KRCWM:$mask, _.MemOp:$src),
                  OpcodeStr #"\t{$src, ${dst} {${mask}} {z}|"#
                                "${dst} {${mask}} {z}, $src}",
                  [(set _.RC:$dst, (_.VT (vselect _.KRCWM:$mask,
                    (_.VT (bitconvert (ld_frag addr:$src))), _.ImmAllZerosV)))],
                  _.ExeDomain>, EVEX, EVEX_KZ;
  }
  def : Pat<(_.VT (mload addr:$ptr, _.KRCWM:$mask, undef)),
            (!cast<Instruction>(NAME#_.ZSuffix##rmkz) _.KRCWM:$mask, addr:$ptr)>;

  def : Pat<(_.VT (mload addr:$ptr, _.KRCWM:$mask, _.ImmAllZerosV)),
            (!cast<Instruction>(NAME#_.ZSuffix##rmkz) _.KRCWM:$mask, addr:$ptr)>;

  def : Pat<(_.VT (mload addr:$ptr, _.KRCWM:$mask, (_.VT _.RC:$src0))),
            (!cast<Instruction>(NAME#_.ZSuffix##rmk) _.RC:$src0,
             _.KRCWM:$mask, addr:$ptr)>;
}

multiclass avx512_alignedload_vl<bits<8> opc, string OpcodeStr,
                                  AVX512VLVectorVTInfo _,
                                  Predicate prd> {
  let Predicates = [prd] in
  defm Z : avx512_load<opc, OpcodeStr, _.info512, _.info512.AlignedLdFrag,
                       masked_load_aligned512>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
  defm Z256 : avx512_load<opc, OpcodeStr, _.info256, _.info256.AlignedLdFrag,
                          masked_load_aligned256>, EVEX_V256;
  defm Z128 : avx512_load<opc, OpcodeStr, _.info128, _.info128.AlignedLdFrag,
                          masked_load_aligned128>, EVEX_V128;
  }
}

multiclass avx512_load_vl<bits<8> opc, string OpcodeStr,
                                  AVX512VLVectorVTInfo _,
                                  Predicate prd,
                                  SDPatternOperator SelectOprr = vselect> {
  let Predicates = [prd] in
  defm Z : avx512_load<opc, OpcodeStr, _.info512, _.info512.LdFrag,
                       masked_load_unaligned, SelectOprr>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
  defm Z256 : avx512_load<opc, OpcodeStr, _.info256, _.info256.LdFrag,
                         masked_load_unaligned, SelectOprr>, EVEX_V256;
  defm Z128 : avx512_load<opc, OpcodeStr, _.info128, _.info128.LdFrag,
                         masked_load_unaligned, SelectOprr>, EVEX_V128;
  }
}

multiclass avx512_store<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                        PatFrag st_frag, PatFrag mstore> {

  let hasSideEffects = 0 in {
  def rr_REV  : AVX512PI<opc, MRMDestReg, (outs _.RC:$dst), (ins _.RC:$src),
                         OpcodeStr # ".s\t{$src, $dst|$dst, $src}",
                         [], _.ExeDomain>, EVEX;
  def rrk_REV : AVX512PI<opc, MRMDestReg, (outs  _.RC:$dst),
                         (ins _.KRCWM:$mask, _.RC:$src),
                         OpcodeStr # ".s\t{$src, ${dst} {${mask}}|"#
                         "${dst} {${mask}}, $src}",
                         [], _.ExeDomain>,  EVEX, EVEX_K;
  def rrkz_REV : AVX512PI<opc, MRMDestReg, (outs  _.RC:$dst),
                          (ins _.KRCWM:$mask, _.RC:$src),
                          OpcodeStr # ".s\t{$src, ${dst} {${mask}} {z}|" #
                          "${dst} {${mask}} {z}, $src}",
                          [], _.ExeDomain>, EVEX, EVEX_KZ;
  }

  def mr : AVX512PI<opc, MRMDestMem, (outs), (ins _.MemOp:$dst, _.RC:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(st_frag (_.VT _.RC:$src), addr:$dst)], _.ExeDomain>, EVEX;
  def mrk : AVX512PI<opc, MRMDestMem, (outs),
                     (ins _.MemOp:$dst, _.KRCWM:$mask, _.RC:$src),
              OpcodeStr # "\t{$src, ${dst} {${mask}}|${dst} {${mask}}, $src}",
               [], _.ExeDomain>, EVEX, EVEX_K;

  def: Pat<(mstore addr:$ptr, _.KRCWM:$mask, (_.VT _.RC:$src)),
           (!cast<Instruction>(NAME#_.ZSuffix##mrk) addr:$ptr,
                                                    _.KRCWM:$mask, _.RC:$src)>;
}


multiclass avx512_store_vl< bits<8> opc, string OpcodeStr,
                            AVX512VLVectorVTInfo _, Predicate prd> {
  let Predicates = [prd] in
  defm Z : avx512_store<opc, OpcodeStr, _.info512, store,
                        masked_store_unaligned>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_store<opc, OpcodeStr, _.info256, store,
                             masked_store_unaligned>, EVEX_V256;
    defm Z128 : avx512_store<opc, OpcodeStr, _.info128, store,
                             masked_store_unaligned>, EVEX_V128;
  }
}

multiclass avx512_alignedstore_vl<bits<8> opc, string OpcodeStr,
                                  AVX512VLVectorVTInfo _,  Predicate prd> {
  let Predicates = [prd] in
  defm Z : avx512_store<opc, OpcodeStr, _.info512, alignedstore512,
                        masked_store_aligned512>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_store<opc, OpcodeStr, _.info256, alignedstore256,
                             masked_store_aligned256>, EVEX_V256;
    defm Z128 : avx512_store<opc, OpcodeStr, _.info128, alignedstore,
                             masked_store_aligned128>, EVEX_V128;
  }
}

defm VMOVAPS : avx512_alignedload_vl<0x28, "vmovaps", avx512vl_f32_info,
                                     HasAVX512>,
               avx512_alignedstore_vl<0x29, "vmovaps", avx512vl_f32_info,
                                      HasAVX512>,  PS, EVEX_CD8<32, CD8VF>;

defm VMOVAPD : avx512_alignedload_vl<0x28, "vmovapd", avx512vl_f64_info,
                                     HasAVX512>,
               avx512_alignedstore_vl<0x29, "vmovapd", avx512vl_f64_info,
                                     HasAVX512>, PD, VEX_W, EVEX_CD8<64, CD8VF>;

defm VMOVUPS : avx512_load_vl<0x10, "vmovups", avx512vl_f32_info, HasAVX512,
                              null_frag>,
               avx512_store_vl<0x11, "vmovups", avx512vl_f32_info, HasAVX512>,
                              PS, EVEX_CD8<32, CD8VF>;

defm VMOVUPD : avx512_load_vl<0x10, "vmovupd", avx512vl_f64_info, HasAVX512,
                              null_frag>,
               avx512_store_vl<0x11, "vmovupd", avx512vl_f64_info, HasAVX512>,
               PD, VEX_W, EVEX_CD8<64, CD8VF>;

defm VMOVDQA32 : avx512_alignedload_vl<0x6F, "vmovdqa32", avx512vl_i32_info,
                                       HasAVX512>,
                 avx512_alignedstore_vl<0x7F, "vmovdqa32", avx512vl_i32_info,
                                       HasAVX512>, PD, EVEX_CD8<32, CD8VF>;

defm VMOVDQA64 : avx512_alignedload_vl<0x6F, "vmovdqa64", avx512vl_i64_info,
                                       HasAVX512>,
                 avx512_alignedstore_vl<0x7F, "vmovdqa64", avx512vl_i64_info,
                                    HasAVX512>, PD, VEX_W, EVEX_CD8<64, CD8VF>;

defm VMOVDQU8 : avx512_load_vl<0x6F, "vmovdqu8", avx512vl_i8_info, HasBWI>,
                 avx512_store_vl<0x7F, "vmovdqu8", avx512vl_i8_info,
                                 HasBWI>, XD, EVEX_CD8<8, CD8VF>;

defm VMOVDQU16 : avx512_load_vl<0x6F, "vmovdqu16", avx512vl_i16_info, HasBWI>,
                 avx512_store_vl<0x7F, "vmovdqu16", avx512vl_i16_info,
                                 HasBWI>, XD, VEX_W, EVEX_CD8<16, CD8VF>;

defm VMOVDQU32 : avx512_load_vl<0x6F, "vmovdqu32", avx512vl_i32_info, HasAVX512,
                                null_frag>,
                 avx512_store_vl<0x7F, "vmovdqu32", avx512vl_i32_info,
                                 HasAVX512>, XS, EVEX_CD8<32, CD8VF>;

defm VMOVDQU64 : avx512_load_vl<0x6F, "vmovdqu64", avx512vl_i64_info, HasAVX512,
                                null_frag>,
                 avx512_store_vl<0x7F, "vmovdqu64", avx512vl_i64_info,
                                 HasAVX512>, XS, VEX_W, EVEX_CD8<64, CD8VF>;

// Special instructions to help with spilling when we don't have VLX. We need
// to load or store from a ZMM register instead. These are converted in
// expandPostRAPseudos.
let isReMaterializable = 1, canFoldAsLoad = 1,
    isPseudo = 1, SchedRW = [WriteLoad], mayLoad = 1, hasSideEffects = 0 in {
def VMOVAPSZ128rm_NOVLX : I<0, Pseudo, (outs VR128X:$dst), (ins f128mem:$src),
                            "", []>;
def VMOVAPSZ256rm_NOVLX : I<0, Pseudo, (outs VR256X:$dst), (ins f256mem:$src),
                            "", []>;
def VMOVUPSZ128rm_NOVLX : I<0, Pseudo, (outs VR128X:$dst), (ins f128mem:$src),
                            "", []>;
def VMOVUPSZ256rm_NOVLX : I<0, Pseudo, (outs VR256X:$dst), (ins f256mem:$src),
                            "", []>;
}

let isPseudo = 1, mayStore = 1, hasSideEffects = 0 in {
def VMOVAPSZ128mr_NOVLX : I<0, Pseudo, (outs), (ins f128mem:$dst, VR128X:$src),
                            "", []>;
def VMOVAPSZ256mr_NOVLX : I<0, Pseudo, (outs), (ins f256mem:$dst, VR256X:$src),
                            "", []>;
def VMOVUPSZ128mr_NOVLX : I<0, Pseudo, (outs), (ins f128mem:$dst, VR128X:$src),
                            "", []>;
def VMOVUPSZ256mr_NOVLX : I<0, Pseudo, (outs), (ins f256mem:$dst, VR256X:$src),
                            "", []>;
}

def : Pat<(v8i64 (vselect VK8WM:$mask, (bc_v8i64 (v16i32 immAllZerosV)),
                          (v8i64 VR512:$src))),
   (VMOVDQA64Zrrkz (COPY_TO_REGCLASS (KNOTWrr (COPY_TO_REGCLASS VK8:$mask, VK16)),
                                              VK8), VR512:$src)>;

def : Pat<(v16i32 (vselect VK16WM:$mask, (v16i32 immAllZerosV),
                           (v16i32 VR512:$src))),
                  (VMOVDQA32Zrrkz (KNOTWrr VK16WM:$mask), VR512:$src)>;

// These patterns exist to prevent the above patterns from introducing a second
// mask inversion when one already exists.
def : Pat<(v8i64 (vselect (xor VK8:$mask, (v8i1 immAllOnesV)),
                          (bc_v8i64 (v16i32 immAllZerosV)),
                          (v8i64 VR512:$src))),
                 (VMOVDQA64Zrrkz VK8:$mask, VR512:$src)>;
def : Pat<(v16i32 (vselect (xor VK16:$mask, (v16i1 immAllOnesV)),
                           (v16i32 immAllZerosV),
                           (v16i32 VR512:$src))),
                  (VMOVDQA32Zrrkz VK16WM:$mask, VR512:$src)>;

// Patterns for handling v8i1 selects of 256-bit vectors when VLX isn't
// available. Use a 512-bit operation and extract.
let Predicates = [HasAVX512, NoVLX] in {
def : Pat<(v8f32 (vselect (v8i1 VK8WM:$mask), (v8f32 VR256X:$src1),
                          (v8f32 VR256X:$src0))),
          (EXTRACT_SUBREG
           (v16f32
            (VMOVAPSZrrk
             (v16f32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src0, sub_ymm)),
             (COPY_TO_REGCLASS VK8WM:$mask, VK16WM),
             (v16f32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src1, sub_ymm)))),
           sub_ymm)>;

def : Pat<(v8i32 (vselect (v8i1 VK8WM:$mask), (v8i32 VR256X:$src1),
                          (v8i32 VR256X:$src0))),
          (EXTRACT_SUBREG
           (v16i32
            (VMOVDQA32Zrrk
             (v16i32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src0, sub_ymm)),
             (COPY_TO_REGCLASS VK8WM:$mask, VK16WM),
             (v16i32 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src1, sub_ymm)))),
           sub_ymm)>;
}

let Predicates = [HasVLX, NoBWI] in {
  // 128-bit load/store without BWI.
  def : Pat<(alignedstore (v8i16 VR128X:$src), addr:$dst),
            (VMOVDQA32Z128mr addr:$dst, VR128X:$src)>;
  def : Pat<(alignedstore (v16i8 VR128X:$src), addr:$dst),
            (VMOVDQA32Z128mr addr:$dst, VR128X:$src)>;
  def : Pat<(store (v8i16 VR128X:$src), addr:$dst),
            (VMOVDQU32Z128mr addr:$dst, VR128X:$src)>;
  def : Pat<(store (v16i8 VR128X:$src), addr:$dst),
            (VMOVDQU32Z128mr addr:$dst, VR128X:$src)>;

  // 256-bit load/store without BWI.
  def : Pat<(alignedstore256 (v16i16 VR256X:$src), addr:$dst),
            (VMOVDQA32Z256mr addr:$dst, VR256X:$src)>;
  def : Pat<(alignedstore256 (v32i8 VR256X:$src), addr:$dst),
            (VMOVDQA32Z256mr addr:$dst, VR256X:$src)>;
  def : Pat<(store (v16i16 VR256X:$src), addr:$dst),
            (VMOVDQU32Z256mr addr:$dst, VR256X:$src)>;
  def : Pat<(store (v32i8 VR256X:$src), addr:$dst),
            (VMOVDQU32Z256mr addr:$dst, VR256X:$src)>;
}

let Predicates = [HasVLX] in {
  // Special patterns for storing subvector extracts of lower 128-bits of 256.
  // Its cheaper to just use VMOVAPS/VMOVUPS instead of VEXTRACTF128mr
  def : Pat<(alignedstore (v2f64 (extract_subvector
                                  (v4f64 VR256X:$src), (iPTR 0))), addr:$dst),
     (VMOVAPDZ128mr addr:$dst, (v2f64 (EXTRACT_SUBREG VR256X:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v4f32 (extract_subvector
                                  (v8f32 VR256X:$src), (iPTR 0))), addr:$dst),
     (VMOVAPSZ128mr addr:$dst, (v4f32 (EXTRACT_SUBREG VR256X:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v2i64 (extract_subvector
                                  (v4i64 VR256X:$src), (iPTR 0))), addr:$dst),
     (VMOVDQA64Z128mr addr:$dst, (v2i64 (EXTRACT_SUBREG VR256X:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v4i32 (extract_subvector
                                  (v8i32 VR256X:$src), (iPTR 0))), addr:$dst),
     (VMOVDQA32Z128mr addr:$dst, (v4i32 (EXTRACT_SUBREG VR256X:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v8i16 (extract_subvector
                                  (v16i16 VR256X:$src), (iPTR 0))), addr:$dst),
     (VMOVDQA32Z128mr addr:$dst, (v8i16 (EXTRACT_SUBREG VR256X:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v16i8 (extract_subvector
                                  (v32i8 VR256X:$src), (iPTR 0))), addr:$dst),
     (VMOVDQA32Z128mr addr:$dst, (v16i8 (EXTRACT_SUBREG VR256X:$src,sub_xmm)))>;

  def : Pat<(store (v2f64 (extract_subvector
                           (v4f64 VR256X:$src), (iPTR 0))), addr:$dst),
     (VMOVUPDZ128mr addr:$dst, (v2f64 (EXTRACT_SUBREG VR256X:$src,sub_xmm)))>;
  def : Pat<(store (v4f32 (extract_subvector
                           (v8f32 VR256X:$src), (iPTR 0))), addr:$dst),
     (VMOVUPSZ128mr addr:$dst, (v4f32 (EXTRACT_SUBREG VR256X:$src,sub_xmm)))>;
  def : Pat<(store (v2i64 (extract_subvector
                           (v4i64 VR256X:$src), (iPTR 0))), addr:$dst),
     (VMOVDQU64Z128mr addr:$dst, (v2i64 (EXTRACT_SUBREG VR256X:$src,sub_xmm)))>;
  def : Pat<(store (v4i32 (extract_subvector
                           (v8i32 VR256X:$src), (iPTR 0))), addr:$dst),
     (VMOVDQU32Z128mr addr:$dst, (v4i32 (EXTRACT_SUBREG VR256X:$src,sub_xmm)))>;
  def : Pat<(store (v8i16 (extract_subvector
                           (v16i16 VR256X:$src), (iPTR 0))), addr:$dst),
     (VMOVDQU32Z128mr addr:$dst, (v8i16 (EXTRACT_SUBREG VR256X:$src,sub_xmm)))>;
  def : Pat<(store (v16i8 (extract_subvector
                           (v32i8 VR256X:$src), (iPTR 0))), addr:$dst),
     (VMOVDQU32Z128mr addr:$dst, (v16i8 (EXTRACT_SUBREG VR256X:$src,sub_xmm)))>;

  // Special patterns for storing subvector extracts of lower 128-bits of 512.
  // Its cheaper to just use VMOVAPS/VMOVUPS instead of VEXTRACTF128mr
  def : Pat<(alignedstore (v2f64 (extract_subvector
                                  (v8f64 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVAPDZ128mr addr:$dst, (v2f64 (EXTRACT_SUBREG VR512:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v4f32 (extract_subvector
                                  (v16f32 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVAPSZ128mr addr:$dst, (v4f32 (EXTRACT_SUBREG VR512:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v2i64 (extract_subvector
                                  (v8i64 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQA64Z128mr addr:$dst, (v2i64 (EXTRACT_SUBREG VR512:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v4i32 (extract_subvector
                                  (v16i32 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQA32Z128mr addr:$dst, (v4i32 (EXTRACT_SUBREG VR512:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v8i16 (extract_subvector
                                  (v32i16 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQA32Z128mr addr:$dst, (v8i16 (EXTRACT_SUBREG VR512:$src,sub_xmm)))>;
  def : Pat<(alignedstore (v16i8 (extract_subvector
                                  (v64i8 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQA32Z128mr addr:$dst, (v16i8 (EXTRACT_SUBREG VR512:$src,sub_xmm)))>;

  def : Pat<(store (v2f64 (extract_subvector
                           (v8f64 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVUPDZ128mr addr:$dst, (v2f64 (EXTRACT_SUBREG VR512:$src,sub_xmm)))>;
  def : Pat<(store (v4f32 (extract_subvector
                           (v16f32 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVUPSZ128mr addr:$dst, (v4f32 (EXTRACT_SUBREG VR512:$src,sub_xmm)))>;
  def : Pat<(store (v2i64 (extract_subvector
                           (v8i64 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQU64Z128mr addr:$dst, (v2i64 (EXTRACT_SUBREG VR512:$src,sub_xmm)))>;
  def : Pat<(store (v4i32 (extract_subvector
                           (v16i32 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQU32Z128mr addr:$dst, (v4i32 (EXTRACT_SUBREG VR512:$src,sub_xmm)))>;
  def : Pat<(store (v8i16 (extract_subvector
                           (v32i16 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQU32Z128mr addr:$dst, (v8i16 (EXTRACT_SUBREG VR512:$src,sub_xmm)))>;
  def : Pat<(store (v16i8 (extract_subvector
                           (v64i8 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQU32Z128mr addr:$dst, (v16i8 (EXTRACT_SUBREG VR512:$src,sub_xmm)))>;

  // Special patterns for storing subvector extracts of lower 256-bits of 512.
  // Its cheaper to just use VMOVAPS/VMOVUPS instead of VEXTRACTF128mr
  def : Pat<(alignedstore256 (v4f64 (extract_subvector
                                     (v8f64 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVAPDZ256mr addr:$dst, (v4f64 (EXTRACT_SUBREG VR512:$src,sub_ymm)))>;
  def : Pat<(alignedstore (v8f32 (extract_subvector
                                  (v16f32 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVAPSZ256mr addr:$dst, (v8f32 (EXTRACT_SUBREG VR512:$src,sub_ymm)))>;
  def : Pat<(alignedstore256 (v4i64 (extract_subvector
                                     (v8i64 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQA64Z256mr addr:$dst, (v4i64 (EXTRACT_SUBREG VR512:$src,sub_ymm)))>;
  def : Pat<(alignedstore256 (v8i32 (extract_subvector
                                     (v16i32 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQA32Z256mr addr:$dst, (v8i32 (EXTRACT_SUBREG VR512:$src,sub_ymm)))>;
  def : Pat<(alignedstore256 (v16i16 (extract_subvector
                                      (v32i16 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQA32Z256mr addr:$dst, (v16i16 (EXTRACT_SUBREG VR512:$src,sub_ymm)))>;
  def : Pat<(alignedstore256 (v32i8 (extract_subvector
                                     (v64i8 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQA32Z256mr addr:$dst, (v32i8 (EXTRACT_SUBREG VR512:$src,sub_ymm)))>;

  def : Pat<(store (v4f64 (extract_subvector
                           (v8f64 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVUPDZ256mr addr:$dst, (v4f64 (EXTRACT_SUBREG VR512:$src,sub_ymm)))>;
  def : Pat<(store (v8f32 (extract_subvector
                           (v16f32 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVUPSZ256mr addr:$dst, (v8f32 (EXTRACT_SUBREG VR512:$src,sub_ymm)))>;
  def : Pat<(store (v4i64 (extract_subvector
                           (v8i64 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQU64Z256mr addr:$dst, (v4i64 (EXTRACT_SUBREG VR512:$src,sub_ymm)))>;
  def : Pat<(store (v8i32 (extract_subvector
                           (v16i32 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQU32Z256mr addr:$dst, (v8i32 (EXTRACT_SUBREG VR512:$src,sub_ymm)))>;
  def : Pat<(store (v16i16 (extract_subvector
                            (v32i16 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQU32Z256mr addr:$dst, (v16i16 (EXTRACT_SUBREG VR512:$src,sub_ymm)))>;
  def : Pat<(store (v32i8 (extract_subvector
                           (v64i8 VR512:$src), (iPTR 0))), addr:$dst),
     (VMOVDQU32Z256mr addr:$dst, (v32i8 (EXTRACT_SUBREG VR512:$src,sub_ymm)))>;
}


// Move Int Doubleword to Packed Double Int
//
let ExeDomain = SSEPackedInt in {
def VMOVDI2PDIZrr : AVX512BI<0x6E, MRMSrcReg, (outs VR128X:$dst), (ins GR32:$src),
                      "vmovd\t{$src, $dst|$dst, $src}",
                      [(set VR128X:$dst,
                        (v4i32 (scalar_to_vector GR32:$src)))], IIC_SSE_MOVDQ>,
                        EVEX;
def VMOVDI2PDIZrm : AVX512BI<0x6E, MRMSrcMem, (outs VR128X:$dst), (ins i32mem:$src),
                      "vmovd\t{$src, $dst|$dst, $src}",
                      [(set VR128X:$dst,
                        (v4i32 (scalar_to_vector (loadi32 addr:$src))))],
                        IIC_SSE_MOVDQ>, EVEX, EVEX_CD8<32, CD8VT1>;
def VMOV64toPQIZrr : AVX512BI<0x6E, MRMSrcReg, (outs VR128X:$dst), (ins GR64:$src),
                      "vmovq\t{$src, $dst|$dst, $src}",
                        [(set VR128X:$dst,
                          (v2i64 (scalar_to_vector GR64:$src)))],
                          IIC_SSE_MOVDQ>, EVEX, VEX_W;
let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0, mayLoad = 1 in
def VMOV64toPQIZrm : AVX512BI<0x6E, MRMSrcMem, (outs VR128X:$dst),
                      (ins i64mem:$src),
                      "vmovq\t{$src, $dst|$dst, $src}", []>,
                      EVEX, VEX_W, EVEX_CD8<64, CD8VT1>;
let isCodeGenOnly = 1 in {
def VMOV64toSDZrr : AVX512BI<0x6E, MRMSrcReg, (outs FR64X:$dst), (ins GR64:$src),
                       "vmovq\t{$src, $dst|$dst, $src}",
                       [(set FR64X:$dst, (bitconvert GR64:$src))],
                       IIC_SSE_MOVDQ>, EVEX, VEX_W, Sched<[WriteMove]>;
def VMOV64toSDZrm : AVX512XSI<0x7E, MRMSrcMem, (outs FR64X:$dst), (ins i64mem:$src),
                      "vmovq\t{$src, $dst|$dst, $src}",
                      [(set FR64X:$dst, (bitconvert (loadi64 addr:$src)))]>,
                      EVEX, VEX_W, EVEX_CD8<8, CD8VT8>;
def VMOVSDto64Zrr : AVX512BI<0x7E, MRMDestReg, (outs GR64:$dst), (ins FR64X:$src),
                         "vmovq\t{$src, $dst|$dst, $src}",
                         [(set GR64:$dst, (bitconvert FR64X:$src))],
                         IIC_SSE_MOVDQ>, EVEX, VEX_W, Sched<[WriteMove]>;
def VMOVSDto64Zmr : AVX512BI<0x7E, MRMDestMem, (outs), (ins i64mem:$dst, FR64X:$src),
                         "vmovq\t{$src, $dst|$dst, $src}",
                         [(store (i64 (bitconvert FR64X:$src)), addr:$dst)],
                         IIC_SSE_MOVDQ>, EVEX, VEX_W, Sched<[WriteStore]>,
                         EVEX_CD8<64, CD8VT1>;
}
} // ExeDomain = SSEPackedInt

// Move Int Doubleword to Single Scalar
//
let ExeDomain = SSEPackedInt, isCodeGenOnly = 1 in {
def VMOVDI2SSZrr  : AVX512BI<0x6E, MRMSrcReg, (outs FR32X:$dst), (ins GR32:$src),
                      "vmovd\t{$src, $dst|$dst, $src}",
                      [(set FR32X:$dst, (bitconvert GR32:$src))],
                      IIC_SSE_MOVDQ>, EVEX;

def VMOVDI2SSZrm  : AVX512BI<0x6E, MRMSrcMem, (outs FR32X:$dst), (ins i32mem:$src),
                      "vmovd\t{$src, $dst|$dst, $src}",
                      [(set FR32X:$dst, (bitconvert (loadi32 addr:$src)))],
                      IIC_SSE_MOVDQ>, EVEX, EVEX_CD8<32, CD8VT1>;
} // ExeDomain = SSEPackedInt, isCodeGenOnly = 1

// Move doubleword from xmm register to r/m32
//
let ExeDomain = SSEPackedInt in {
def VMOVPDI2DIZrr  : AVX512BI<0x7E, MRMDestReg, (outs GR32:$dst), (ins VR128X:$src),
                       "vmovd\t{$src, $dst|$dst, $src}",
                       [(set GR32:$dst, (extractelt (v4i32 VR128X:$src),
                                        (iPTR 0)))], IIC_SSE_MOVD_ToGP>,
                       EVEX;
def VMOVPDI2DIZmr  : AVX512BI<0x7E, MRMDestMem, (outs),
                       (ins i32mem:$dst, VR128X:$src),
                       "vmovd\t{$src, $dst|$dst, $src}",
                       [(store (i32 (extractelt (v4i32 VR128X:$src),
                                     (iPTR 0))), addr:$dst)], IIC_SSE_MOVDQ>,
                       EVEX, EVEX_CD8<32, CD8VT1>;
} // ExeDomain = SSEPackedInt

// Move quadword from xmm1 register to r/m64
//
let ExeDomain = SSEPackedInt in {
def VMOVPQIto64Zrr : I<0x7E, MRMDestReg, (outs GR64:$dst), (ins VR128X:$src),
                      "vmovq\t{$src, $dst|$dst, $src}",
                      [(set GR64:$dst, (extractelt (v2i64 VR128X:$src),
                                                   (iPTR 0)))],
                      IIC_SSE_MOVD_ToGP>, PD, EVEX, VEX_W,
                      Requires<[HasAVX512, In64BitMode]>;

let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0, mayStore = 1 in
def VMOVPQIto64Zmr : I<0x7E, MRMDestMem, (outs), (ins i64mem:$dst, VR128X:$src),
                      "vmovq\t{$src, $dst|$dst, $src}",
                      [], IIC_SSE_MOVD_ToGP>, PD, EVEX, VEX_W,
                      Requires<[HasAVX512, In64BitMode]>;

def VMOVPQI2QIZmr : I<0xD6, MRMDestMem, (outs),
                      (ins i64mem:$dst, VR128X:$src),
                      "vmovq\t{$src, $dst|$dst, $src}",
                      [(store (extractelt (v2i64 VR128X:$src), (iPTR 0)),
                              addr:$dst)], IIC_SSE_MOVDQ>,
                      EVEX, PD, VEX_W, EVEX_CD8<64, CD8VT1>,
                      Sched<[WriteStore]>, Requires<[HasAVX512, In64BitMode]>;

let hasSideEffects = 0 in
def VMOVPQI2QIZrr : AVX512BI<0xD6, MRMDestReg, (outs VR128X:$dst),
                             (ins VR128X:$src),
                             "vmovq.s\t{$src, $dst|$dst, $src}",[]>,
                             EVEX, VEX_W;
} // ExeDomain = SSEPackedInt

// Move Scalar Single to Double Int
//
let ExeDomain = SSEPackedInt, isCodeGenOnly = 1 in {
def VMOVSS2DIZrr  : AVX512BI<0x7E, MRMDestReg, (outs GR32:$dst),
                      (ins FR32X:$src),
                      "vmovd\t{$src, $dst|$dst, $src}",
                      [(set GR32:$dst, (bitconvert FR32X:$src))],
                      IIC_SSE_MOVD_ToGP>, EVEX;
def VMOVSS2DIZmr  : AVX512BI<0x7E, MRMDestMem, (outs),
                      (ins i32mem:$dst, FR32X:$src),
                      "vmovd\t{$src, $dst|$dst, $src}",
                      [(store (i32 (bitconvert FR32X:$src)), addr:$dst)],
                      IIC_SSE_MOVDQ>, EVEX, EVEX_CD8<32, CD8VT1>;
} // ExeDomain = SSEPackedInt, isCodeGenOnly = 1

// Move Quadword Int to Packed Quadword Int
//
let ExeDomain = SSEPackedInt in {
def VMOVQI2PQIZrm : AVX512XSI<0x7E, MRMSrcMem, (outs VR128X:$dst),
                      (ins i64mem:$src),
                      "vmovq\t{$src, $dst|$dst, $src}",
                      [(set VR128X:$dst,
                        (v2i64 (scalar_to_vector (loadi64 addr:$src))))]>,
                      EVEX, VEX_W, EVEX_CD8<8, CD8VT8>;
} // ExeDomain = SSEPackedInt

//===----------------------------------------------------------------------===//
// AVX-512  MOVSS, MOVSD
//===----------------------------------------------------------------------===//

multiclass avx512_move_scalar<string asm, SDNode OpNode,
                              X86VectorVTInfo _> {
  def rr : AVX512PI<0x10, MRMSrcReg, (outs _.RC:$dst),
             (ins _.RC:$src1, _.FRC:$src2),
             !strconcat(asm, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set _.RC:$dst, (_.VT (OpNode _.RC:$src1,
                                    (scalar_to_vector _.FRC:$src2))))],
             _.ExeDomain,IIC_SSE_MOV_S_RR>, EVEX_4V;
  def rrkz : AVX512PI<0x10, MRMSrcReg, (outs _.RC:$dst),
              (ins _.KRCWM:$mask, _.RC:$src1, _.RC:$src2),
              !strconcat(asm, "\t{$src2, $src1, $dst {${mask}} {z}|",
              "$dst {${mask}} {z}, $src1, $src2}"),
              [(set _.RC:$dst, (_.VT (X86selects _.KRCWM:$mask,
                                      (_.VT (OpNode _.RC:$src1, _.RC:$src2)),
                                      _.ImmAllZerosV)))],
              _.ExeDomain,IIC_SSE_MOV_S_RR>, EVEX_4V, EVEX_KZ;
  let Constraints = "$src0 = $dst"  in
  def rrk : AVX512PI<0x10, MRMSrcReg, (outs _.RC:$dst),
             (ins _.RC:$src0, _.KRCWM:$mask, _.RC:$src1, _.RC:$src2),
             !strconcat(asm, "\t{$src2, $src1, $dst {${mask}}|",
             "$dst {${mask}}, $src1, $src2}"),
             [(set _.RC:$dst, (_.VT (X86selects _.KRCWM:$mask,
                                     (_.VT (OpNode _.RC:$src1, _.RC:$src2)),
                                     (_.VT _.RC:$src0))))],
             _.ExeDomain,IIC_SSE_MOV_S_RR>, EVEX_4V, EVEX_K;
  let canFoldAsLoad = 1, isReMaterializable = 1 in
  def rm : AVX512PI<0x10, MRMSrcMem, (outs _.FRC:$dst), (ins _.ScalarMemOp:$src),
             !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
             [(set _.FRC:$dst, (_.ScalarLdFrag addr:$src))],
             _.ExeDomain, IIC_SSE_MOV_S_RM>, EVEX;
  let mayLoad = 1, hasSideEffects = 0 in {
    let Constraints = "$src0 = $dst" in
    def rmk : AVX512PI<0x10, MRMSrcMem, (outs _.RC:$dst),
               (ins _.RC:$src0, _.KRCWM:$mask, _.ScalarMemOp:$src),
               !strconcat(asm, "\t{$src, $dst {${mask}}|",
               "$dst {${mask}}, $src}"),
               [], _.ExeDomain, IIC_SSE_MOV_S_RM>, EVEX, EVEX_K;
    def rmkz : AVX512PI<0x10, MRMSrcMem, (outs _.RC:$dst),
               (ins _.KRCWM:$mask, _.ScalarMemOp:$src),
               !strconcat(asm, "\t{$src, $dst {${mask}} {z}|",
               "$dst {${mask}} {z}, $src}"),
               [], _.ExeDomain, IIC_SSE_MOV_S_RM>, EVEX, EVEX_KZ;
  }
  def mr: AVX512PI<0x11, MRMDestMem, (outs), (ins _.ScalarMemOp:$dst, _.FRC:$src),
             !strconcat(asm, "\t{$src, $dst|$dst, $src}"),
             [(store _.FRC:$src, addr:$dst)],  _.ExeDomain, IIC_SSE_MOV_S_MR>,
             EVEX;
  let mayStore = 1, hasSideEffects = 0 in
  def mrk: AVX512PI<0x11, MRMDestMem, (outs),
              (ins _.ScalarMemOp:$dst, VK1WM:$mask, _.FRC:$src),
              !strconcat(asm, "\t{$src, $dst {${mask}}|$dst {${mask}}, $src}"),
              [], _.ExeDomain, IIC_SSE_MOV_S_MR>, EVEX, EVEX_K;
}

defm VMOVSSZ : avx512_move_scalar<"vmovss", X86Movss, f32x_info>,
                                  VEX_LIG, XS, EVEX_CD8<32, CD8VT1>;

defm VMOVSDZ : avx512_move_scalar<"vmovsd", X86Movsd, f64x_info>,
                                  VEX_LIG, XD, VEX_W, EVEX_CD8<64, CD8VT1>;


multiclass avx512_move_scalar_lowering<string InstrStr, SDNode OpNode,
                                       PatLeaf ZeroFP, X86VectorVTInfo _> {

def : Pat<(_.VT (OpNode _.RC:$src0,
                        (_.VT (scalar_to_vector
                                  (_.EltVT (X86selects (i1 (trunc GR32:$mask)),
                                                       (_.EltVT _.FRC:$src1),
                                                       (_.EltVT _.FRC:$src2))))))),
          (COPY_TO_REGCLASS (!cast<Instruction>(InstrStr#rrk)
                                          (COPY_TO_REGCLASS _.FRC:$src2, _.RC),
                                          (COPY_TO_REGCLASS GR32:$mask, VK1WM),
                                          (_.VT _.RC:$src0),
                                          (COPY_TO_REGCLASS _.FRC:$src1, _.RC)),
                            _.RC)>;

def : Pat<(_.VT (OpNode _.RC:$src0,
                        (_.VT (scalar_to_vector
                                  (_.EltVT (X86selects (i1 (trunc GR32:$mask)),
                                                       (_.EltVT _.FRC:$src1),
                                                       (_.EltVT ZeroFP))))))),
          (COPY_TO_REGCLASS (!cast<Instruction>(InstrStr#rrkz)
                                          (COPY_TO_REGCLASS GR32:$mask, VK1WM),
                                          (_.VT _.RC:$src0),
                                          (COPY_TO_REGCLASS _.FRC:$src1, _.RC)),
                            _.RC)>;

}

multiclass avx512_store_scalar_lowering<string InstrStr, AVX512VLVectorVTInfo _,
                                        dag Mask, RegisterClass MaskRC> {

def : Pat<(masked_store addr:$dst, Mask,
             (_.info512.VT (insert_subvector undef,
                               (_.info256.VT (insert_subvector undef,
                                                 (_.info128.VT _.info128.RC:$src),
                                                 (iPTR 0))),
                               (iPTR 0)))),
          (!cast<Instruction>(InstrStr#mrk) addr:$dst,
                      (i1 (COPY_TO_REGCLASS MaskRC:$mask, VK1WM)),
                      (COPY_TO_REGCLASS _.info128.RC:$src, _.info128.FRC))>;

}

multiclass avx512_load_scalar_lowering<string InstrStr, AVX512VLVectorVTInfo _,
                                       dag Mask, RegisterClass MaskRC> {

def : Pat<(_.info128.VT (extract_subvector
                         (_.info512.VT (masked_load addr:$srcAddr, Mask,
                                        (_.info512.VT (bitconvert
                                                       (v16i32 immAllZerosV))))),
                           (iPTR 0))),
          (!cast<Instruction>(InstrStr#rmkz)
                      (i1 (COPY_TO_REGCLASS MaskRC:$mask, VK1WM)),
                      addr:$srcAddr)>;

def : Pat<(_.info128.VT (extract_subvector
                (_.info512.VT (masked_load addr:$srcAddr, Mask,
                      (_.info512.VT (insert_subvector undef,
                            (_.info256.VT (insert_subvector undef,
                                  (_.info128.VT (X86vzmovl _.info128.RC:$src)),
                                  (iPTR 0))),
                            (iPTR 0))))),
                (iPTR 0))),
          (!cast<Instruction>(InstrStr#rmk) _.info128.RC:$src,
                      (i1 (COPY_TO_REGCLASS MaskRC:$mask, VK1WM)),
                      addr:$srcAddr)>;

}

defm : avx512_move_scalar_lowering<"VMOVSSZ", X86Movss, fp32imm0, v4f32x_info>;
defm : avx512_move_scalar_lowering<"VMOVSDZ", X86Movsd, fp64imm0, v2f64x_info>;

defm : avx512_store_scalar_lowering<"VMOVSSZ", avx512vl_f32_info,
                   (v16i1 (bitconvert (i16 (trunc (and GR32:$mask, (i32 1)))))), GR32>;
defm : avx512_store_scalar_lowering<"VMOVSSZ", avx512vl_f32_info,
                   (v16i1 (bitconvert (i16 (and GR16:$mask, (i16 1))))), GR16>;
defm : avx512_store_scalar_lowering<"VMOVSDZ", avx512vl_f64_info,
                   (v8i1 (bitconvert (i8 (and GR8:$mask, (i8 1))))), GR8>;

defm : avx512_load_scalar_lowering<"VMOVSSZ", avx512vl_f32_info,
                   (v16i1 (bitconvert (i16 (trunc (and GR32:$mask, (i32 1)))))), GR32>;
defm : avx512_load_scalar_lowering<"VMOVSSZ", avx512vl_f32_info,
                   (v16i1 (bitconvert (i16 (and GR16:$mask, (i16 1))))), GR16>;
defm : avx512_load_scalar_lowering<"VMOVSDZ", avx512vl_f64_info,
                   (v8i1 (bitconvert (i8 (and GR8:$mask, (i8 1))))), GR8>;

def : Pat<(f32 (X86selects VK1WM:$mask, (f32 FR32X:$src1), (f32 FR32X:$src2))),
          (COPY_TO_REGCLASS (VMOVSSZrrk (COPY_TO_REGCLASS FR32X:$src2, VR128X),
           VK1WM:$mask, (v4f32 (IMPLICIT_DEF)),(COPY_TO_REGCLASS FR32X:$src1, VR128X)), FR32X)>;

def : Pat<(f64 (X86selects VK1WM:$mask, (f64 FR64X:$src1), (f64 FR64X:$src2))),
          (COPY_TO_REGCLASS (VMOVSDZrrk (COPY_TO_REGCLASS FR64X:$src2, VR128X),
           VK1WM:$mask, (v2f64 (IMPLICIT_DEF)), (COPY_TO_REGCLASS FR64X:$src1, VR128X)), FR64X)>;

def : Pat<(int_x86_avx512_mask_store_ss addr:$dst, VR128X:$src, GR8:$mask),
          (VMOVSSZmrk addr:$dst, (i1 (COPY_TO_REGCLASS GR8:$mask, VK1WM)),
           (COPY_TO_REGCLASS VR128X:$src, FR32X))>;

let hasSideEffects = 0 in
defm VMOVSSZrr_REV : AVX512_maskable_in_asm<0x11, MRMDestReg, f32x_info,
                           (outs VR128X:$dst), (ins VR128X:$src1, FR32X:$src2),
                           "vmovss.s", "$src2, $src1", "$src1, $src2", []>,
                           XS, EVEX_4V, VEX_LIG;

let hasSideEffects = 0 in
defm VMOVSDZrr_REV : AVX512_maskable_in_asm<0x11, MRMDestReg, f64x_info,
                           (outs VR128X:$dst), (ins VR128X:$src1, FR64X:$src2),
                           "vmovsd.s", "$src2, $src1", "$src1, $src2", []>,
                           XD, EVEX_4V, VEX_LIG, VEX_W;

let Predicates = [HasAVX512] in {
  let AddedComplexity = 15 in {
  // Move scalar to XMM zero-extended, zeroing a VR128X then do a
  // MOVS{S,D} to the lower bits.
  def : Pat<(v4f32 (X86vzmovl (v4f32 (scalar_to_vector FR32X:$src)))),
            (VMOVSSZrr (v4f32 (AVX512_128_SET0)), FR32X:$src)>;
  def : Pat<(v4f32 (X86vzmovl (v4f32 VR128X:$src))),
            (VMOVSSZrr (v4f32 (AVX512_128_SET0)), (COPY_TO_REGCLASS VR128X:$src, FR32X))>;
  def : Pat<(v4i32 (X86vzmovl (v4i32 VR128X:$src))),
            (VMOVSSZrr (v4i32 (AVX512_128_SET0)), (COPY_TO_REGCLASS VR128X:$src, FR32X))>;
  def : Pat<(v2f64 (X86vzmovl (v2f64 (scalar_to_vector FR64X:$src)))),
            (VMOVSDZrr (v2f64 (AVX512_128_SET0)), FR64X:$src)>;
  }

  // Move low f32 and clear high bits.
  def : Pat<(v8f32 (X86vzmovl (v8f32 VR256X:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSSZrr (v4f32 (AVX512_128_SET0)),
              (EXTRACT_SUBREG (v8f32 VR256X:$src), sub_xmm)), sub_xmm)>;
  def : Pat<(v8i32 (X86vzmovl (v8i32 VR256X:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSSZrr (v4i32 (AVX512_128_SET0)),
              (EXTRACT_SUBREG (v8i32 VR256X:$src), sub_xmm)), sub_xmm)>;
  def : Pat<(v16f32 (X86vzmovl (v16f32 VR512:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSSZrr (v4f32 (AVX512_128_SET0)),
              (EXTRACT_SUBREG (v16f32 VR512:$src), sub_xmm)), sub_xmm)>;
  def : Pat<(v16i32 (X86vzmovl (v16i32 VR512:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSSZrr (v4i32 (AVX512_128_SET0)),
              (EXTRACT_SUBREG (v16i32 VR512:$src), sub_xmm)), sub_xmm)>;

  let AddedComplexity = 20 in {
  // MOVSSrm zeros the high parts of the register; represent this
  // with SUBREG_TO_REG. The AVX versions also write: DST[255:128] <- 0
  def : Pat<(v4f32 (X86vzmovl (v4f32 (scalar_to_vector (loadf32 addr:$src))))),
            (COPY_TO_REGCLASS (VMOVSSZrm addr:$src), VR128X)>;
  def : Pat<(v4f32 (scalar_to_vector (loadf32 addr:$src))),
            (COPY_TO_REGCLASS (VMOVSSZrm addr:$src), VR128X)>;
  def : Pat<(v4f32 (X86vzmovl (loadv4f32 addr:$src))),
            (COPY_TO_REGCLASS (VMOVSSZrm addr:$src), VR128X)>;
  def : Pat<(v4f32 (X86vzload addr:$src)),
            (COPY_TO_REGCLASS (VMOVSSZrm addr:$src), VR128X)>;

  // MOVSDrm zeros the high parts of the register; represent this
  // with SUBREG_TO_REG. The AVX versions also write: DST[255:128] <- 0
  def : Pat<(v2f64 (X86vzmovl (v2f64 (scalar_to_vector (loadf64 addr:$src))))),
            (COPY_TO_REGCLASS (VMOVSDZrm addr:$src), VR128X)>;
  def : Pat<(v2f64 (scalar_to_vector (loadf64 addr:$src))),
            (COPY_TO_REGCLASS (VMOVSDZrm addr:$src), VR128X)>;
  def : Pat<(v2f64 (X86vzmovl (loadv2f64 addr:$src))),
            (COPY_TO_REGCLASS (VMOVSDZrm addr:$src), VR128X)>;
  def : Pat<(v2f64 (X86vzmovl (bc_v2f64 (loadv4f32 addr:$src)))),
            (COPY_TO_REGCLASS (VMOVSDZrm addr:$src), VR128X)>;
  def : Pat<(v2f64 (X86vzload addr:$src)),
            (COPY_TO_REGCLASS (VMOVSDZrm addr:$src), VR128X)>;

  // Represent the same patterns above but in the form they appear for
  // 256-bit types
  def : Pat<(v8i32 (X86vzmovl (insert_subvector undef,
                   (v4i32 (scalar_to_vector (loadi32 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVDI2PDIZrm addr:$src), sub_xmm)>;
  def : Pat<(v8f32 (X86vzmovl (insert_subvector undef,
                   (v4f32 (scalar_to_vector (loadf32 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVSSZrm addr:$src), sub_xmm)>;
  def : Pat<(v8f32 (X86vzload addr:$src)),
            (SUBREG_TO_REG (i32 0), (VMOVSSZrm addr:$src), sub_xmm)>;
  def : Pat<(v4f64 (X86vzmovl (insert_subvector undef,
                   (v2f64 (scalar_to_vector (loadf64 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVSDZrm addr:$src), sub_xmm)>;
  def : Pat<(v4f64 (X86vzload addr:$src)),
            (SUBREG_TO_REG (i32 0), (VMOVSDZrm addr:$src), sub_xmm)>;

  // Represent the same patterns above but in the form they appear for
  // 512-bit types
  def : Pat<(v16i32 (X86vzmovl (insert_subvector undef,
                   (v4i32 (scalar_to_vector (loadi32 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVDI2PDIZrm addr:$src), sub_xmm)>;
  def : Pat<(v16f32 (X86vzmovl (insert_subvector undef,
                   (v4f32 (scalar_to_vector (loadf32 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVSSZrm addr:$src), sub_xmm)>;
  def : Pat<(v16f32 (X86vzload addr:$src)),
            (SUBREG_TO_REG (i32 0), (VMOVSSZrm addr:$src), sub_xmm)>;
  def : Pat<(v8f64 (X86vzmovl (insert_subvector undef,
                   (v2f64 (scalar_to_vector (loadf64 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVSDZrm addr:$src), sub_xmm)>;
  def : Pat<(v8f64 (X86vzload addr:$src)),
            (SUBREG_TO_REG (i32 0), (VMOVSDZrm addr:$src), sub_xmm)>;
  }
  def : Pat<(v8f32 (X86vzmovl (insert_subvector undef,
                   (v4f32 (scalar_to_vector FR32X:$src)), (iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (v4f32 (VMOVSSZrr (v4f32 (AVX512_128_SET0)),
                                            FR32X:$src)), sub_xmm)>;
  def : Pat<(v4f64 (X86vzmovl (insert_subvector undef,
                   (v2f64 (scalar_to_vector FR64X:$src)), (iPTR 0)))),
            (SUBREG_TO_REG (i64 0), (v2f64 (VMOVSDZrr (v2f64 (AVX512_128_SET0)),
                                     FR64X:$src)), sub_xmm)>;
  def : Pat<(v4i64 (X86vzmovl (insert_subvector undef,
                   (v2i64 (scalar_to_vector (loadi64 addr:$src))), (iPTR 0)))),
            (SUBREG_TO_REG (i64 0), (VMOVQI2PQIZrm addr:$src), sub_xmm)>;

  // Move low f64 and clear high bits.
  def : Pat<(v4f64 (X86vzmovl (v4f64 VR256X:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSDZrr (v2f64 (AVX512_128_SET0)),
                       (EXTRACT_SUBREG (v4f64 VR256X:$src), sub_xmm)), sub_xmm)>;
  def : Pat<(v8f64 (X86vzmovl (v8f64 VR512:$src))),
            (SUBREG_TO_REG (i32 0),
             (VMOVSDZrr (v2f64 (AVX512_128_SET0)),
                       (EXTRACT_SUBREG (v8f64 VR512:$src), sub_xmm)), sub_xmm)>;

  def : Pat<(v4i64 (X86vzmovl (v4i64 VR256X:$src))),
            (SUBREG_TO_REG (i32 0), (VMOVSDZrr (v2i64 (AVX512_128_SET0)),
                       (EXTRACT_SUBREG (v4i64 VR256X:$src), sub_xmm)), sub_xmm)>;
  def : Pat<(v8i64 (X86vzmovl (v8i64 VR512:$src))),
            (SUBREG_TO_REG (i32 0), (VMOVSDZrr (v2i64 (AVX512_128_SET0)),
                       (EXTRACT_SUBREG (v8i64 VR512:$src), sub_xmm)), sub_xmm)>;

  // Extract and store.
  def : Pat<(store (f32 (extractelt (v4f32 VR128X:$src), (iPTR 0))),
                   addr:$dst),
            (VMOVSSZmr addr:$dst, (COPY_TO_REGCLASS (v4f32 VR128X:$src), FR32X))>;

  // Shuffle with VMOVSS
  def : Pat<(v4i32 (X86Movss VR128X:$src1, VR128X:$src2)),
            (VMOVSSZrr (v4i32 VR128X:$src1),
                      (COPY_TO_REGCLASS (v4i32 VR128X:$src2), FR32X))>;
  def : Pat<(v4f32 (X86Movss VR128X:$src1, VR128X:$src2)),
            (VMOVSSZrr (v4f32 VR128X:$src1),
                      (COPY_TO_REGCLASS (v4f32 VR128X:$src2), FR32X))>;

  // 256-bit variants
  def : Pat<(v8i32 (X86Movss VR256X:$src1, VR256X:$src2)),
            (SUBREG_TO_REG (i32 0),
              (VMOVSSZrr (EXTRACT_SUBREG (v8i32 VR256X:$src1), sub_xmm),
                        (EXTRACT_SUBREG (v8i32 VR256X:$src2), sub_xmm)),
              sub_xmm)>;
  def : Pat<(v8f32 (X86Movss VR256X:$src1, VR256X:$src2)),
            (SUBREG_TO_REG (i32 0),
              (VMOVSSZrr (EXTRACT_SUBREG (v8f32 VR256X:$src1), sub_xmm),
                        (EXTRACT_SUBREG (v8f32 VR256X:$src2), sub_xmm)),
              sub_xmm)>;

  // Shuffle with VMOVSD
  def : Pat<(v2i64 (X86Movsd VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;
  def : Pat<(v2f64 (X86Movsd VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;

  // 256-bit variants
  def : Pat<(v4i64 (X86Movsd VR256X:$src1, VR256X:$src2)),
            (SUBREG_TO_REG (i32 0),
              (VMOVSDZrr (EXTRACT_SUBREG (v4i64 VR256X:$src1), sub_xmm),
                        (EXTRACT_SUBREG (v4i64 VR256X:$src2), sub_xmm)),
              sub_xmm)>;
  def : Pat<(v4f64 (X86Movsd VR256X:$src1, VR256X:$src2)),
            (SUBREG_TO_REG (i32 0),
              (VMOVSDZrr (EXTRACT_SUBREG (v4f64 VR256X:$src1), sub_xmm),
                        (EXTRACT_SUBREG (v4f64 VR256X:$src2), sub_xmm)),
              sub_xmm)>;

  def : Pat<(v2f64 (X86Movlpd VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;
  def : Pat<(v2i64 (X86Movlpd VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;
  def : Pat<(v4f32 (X86Movlps VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;
  def : Pat<(v4i32 (X86Movlps VR128X:$src1, VR128X:$src2)),
            (VMOVSDZrr VR128X:$src1, (COPY_TO_REGCLASS VR128X:$src2, FR64X))>;
}

let AddedComplexity = 15 in
def VMOVZPQILo2PQIZrr : AVX512XSI<0x7E, MRMSrcReg, (outs VR128X:$dst),
                                (ins VR128X:$src),
                                "vmovq\t{$src, $dst|$dst, $src}",
                                [(set VR128X:$dst, (v2i64 (X86vzmovl
                                                   (v2i64 VR128X:$src))))],
                                IIC_SSE_MOVQ_RR>, EVEX, VEX_W;

let Predicates = [HasAVX512] in {
  let AddedComplexity = 15 in {
    def : Pat<(v4i32 (X86vzmovl (v4i32 (scalar_to_vector GR32:$src)))),
              (VMOVDI2PDIZrr GR32:$src)>;

    def : Pat<(v2i64 (X86vzmovl (v2i64 (scalar_to_vector GR64:$src)))),
              (VMOV64toPQIZrr GR64:$src)>;

    def : Pat<(v4i64 (X86vzmovl (insert_subvector undef,
                                 (v2i64 (scalar_to_vector GR64:$src)),(iPTR 0)))),
              (SUBREG_TO_REG (i64 0), (VMOV64toPQIZrr GR64:$src), sub_xmm)>;

    def : Pat<(v8i64 (X86vzmovl (insert_subvector undef,
                                 (v2i64 (scalar_to_vector GR64:$src)),(iPTR 0)))),
              (SUBREG_TO_REG (i64 0), (VMOV64toPQIZrr GR64:$src), sub_xmm)>;
  }
  // AVX 128-bit movd/movq instruction write zeros in the high 128-bit part.
  let AddedComplexity = 20 in {
    def : Pat<(v2i64 (X86vzmovl (v2i64 (scalar_to_vector (zextloadi64i32 addr:$src))))),
              (VMOVDI2PDIZrm addr:$src)>;
    def : Pat<(v4i32 (X86vzmovl (v4i32 (scalar_to_vector (loadi32 addr:$src))))),
              (VMOVDI2PDIZrm addr:$src)>;
    def : Pat<(v4i32 (X86vzmovl (bc_v4i32 (loadv4f32 addr:$src)))),
              (VMOVDI2PDIZrm addr:$src)>;
    def : Pat<(v4i32 (X86vzmovl (bc_v4i32 (loadv2i64 addr:$src)))),
              (VMOVDI2PDIZrm addr:$src)>;
    def : Pat<(v4i32 (X86vzload addr:$src)),
              (VMOVDI2PDIZrm addr:$src)>;
    def : Pat<(v8i32 (X86vzload addr:$src)),
              (SUBREG_TO_REG (i32 0), (VMOVDI2PDIZrm addr:$src), sub_xmm)>;
    def : Pat<(v2i64 (X86vzmovl (loadv2i64 addr:$src))),
              (VMOVQI2PQIZrm addr:$src)>;
    def : Pat<(v2f64 (X86vzmovl (v2f64 VR128X:$src))),
              (VMOVZPQILo2PQIZrr VR128X:$src)>;
    def : Pat<(v2i64 (X86vzload addr:$src)),
              (VMOVQI2PQIZrm addr:$src)>;
    def : Pat<(v4i64 (X86vzload addr:$src)),
              (SUBREG_TO_REG (i64 0), (VMOVQI2PQIZrm addr:$src), sub_xmm)>;
  }

  // Use regular 128-bit instructions to match 256-bit scalar_to_vec+zext.
  def : Pat<(v8i32 (X86vzmovl (insert_subvector undef,
                               (v4i32 (scalar_to_vector GR32:$src)),(iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVDI2PDIZrr GR32:$src), sub_xmm)>;
  def : Pat<(v16i32 (X86vzmovl (insert_subvector undef,
                                (v4i32 (scalar_to_vector GR32:$src)),(iPTR 0)))),
            (SUBREG_TO_REG (i32 0), (VMOVDI2PDIZrr GR32:$src), sub_xmm)>;

  // Use regular 128-bit instructions to match 512-bit scalar_to_vec+zext.
  def : Pat<(v16i32 (X86vzload addr:$src)),
            (SUBREG_TO_REG (i32 0), (VMOVDI2PDIZrm addr:$src), sub_xmm)>;
  def : Pat<(v8i64 (X86vzload addr:$src)),
            (SUBREG_TO_REG (i64 0), (VMOVQI2PQIZrm addr:$src), sub_xmm)>;
}
//===----------------------------------------------------------------------===//
// AVX-512 - Non-temporals
//===----------------------------------------------------------------------===//
let SchedRW = [WriteLoad] in {
  def VMOVNTDQAZrm : AVX512PI<0x2A, MRMSrcMem, (outs VR512:$dst),
                        (ins i512mem:$src), "vmovntdqa\t{$src, $dst|$dst, $src}",
                        [(set VR512:$dst, (int_x86_avx512_movntdqa addr:$src))],
                        SSEPackedInt>, EVEX, T8PD, EVEX_V512,
                        EVEX_CD8<64, CD8VF>;

  let Predicates = [HasVLX] in {
    def VMOVNTDQAZ256rm : AVX512PI<0x2A, MRMSrcMem, (outs VR256X:$dst),
                         (ins i256mem:$src),
                         "vmovntdqa\t{$src, $dst|$dst, $src}",
                         [(set VR256X:$dst, (int_x86_avx2_movntdqa addr:$src))],
                         SSEPackedInt>, EVEX, T8PD, EVEX_V256,
                         EVEX_CD8<64, CD8VF>;

    def VMOVNTDQAZ128rm : AVX512PI<0x2A, MRMSrcMem, (outs VR128X:$dst),
                        (ins i128mem:$src),
                        "vmovntdqa\t{$src, $dst|$dst, $src}",
                        [(set VR128X:$dst, (int_x86_sse41_movntdqa addr:$src))],
                        SSEPackedInt>, EVEX, T8PD, EVEX_V128,
                        EVEX_CD8<64, CD8VF>;
  }
}

multiclass avx512_movnt<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                        PatFrag st_frag = alignednontemporalstore,
                        InstrItinClass itin = IIC_SSE_MOVNT> {
  let SchedRW = [WriteStore], AddedComplexity = 400 in
  def mr : AVX512PI<opc, MRMDestMem, (outs), (ins _.MemOp:$dst, _.RC:$src),
                    !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                    [(st_frag (_.VT _.RC:$src), addr:$dst)],
                    _.ExeDomain, itin>, EVEX, EVEX_CD8<_.EltSize, CD8VF>;
}

multiclass avx512_movnt_vl<bits<8> opc, string OpcodeStr,
                                                  AVX512VLVectorVTInfo VTInfo> {
  let Predicates = [HasAVX512] in
    defm Z : avx512_movnt<opc, OpcodeStr, VTInfo.info512>, EVEX_V512;

  let Predicates = [HasAVX512, HasVLX] in {
    defm Z256 : avx512_movnt<opc, OpcodeStr, VTInfo.info256>, EVEX_V256;
    defm Z128 : avx512_movnt<opc, OpcodeStr, VTInfo.info128>, EVEX_V128;
  }
}

defm VMOVNTDQ : avx512_movnt_vl<0xE7, "vmovntdq", avx512vl_i64_info>, PD;
defm VMOVNTPD : avx512_movnt_vl<0x2B, "vmovntpd", avx512vl_f64_info>, PD, VEX_W;
defm VMOVNTPS : avx512_movnt_vl<0x2B, "vmovntps", avx512vl_f32_info>, PS;

let Predicates = [HasAVX512], AddedComplexity = 400 in {
  def : Pat<(alignednontemporalstore (v16i32 VR512:$src), addr:$dst),
            (VMOVNTDQZmr addr:$dst, VR512:$src)>;
  def : Pat<(alignednontemporalstore (v32i16 VR512:$src), addr:$dst),
            (VMOVNTDQZmr addr:$dst, VR512:$src)>;
  def : Pat<(alignednontemporalstore (v64i8 VR512:$src), addr:$dst),
            (VMOVNTDQZmr addr:$dst, VR512:$src)>;

  def : Pat<(v8f64 (alignednontemporalload addr:$src)),
            (VMOVNTDQAZrm addr:$src)>;
  def : Pat<(v16f32 (alignednontemporalload addr:$src)),
            (VMOVNTDQAZrm addr:$src)>;
  def : Pat<(v8i64 (alignednontemporalload addr:$src)),
            (VMOVNTDQAZrm addr:$src)>;
  def : Pat<(v16i32 (bitconvert (v8i64 (alignednontemporalload addr:$src)))),
            (VMOVNTDQAZrm addr:$src)>;
  def : Pat<(v32i16 (bitconvert (v8i64 (alignednontemporalload addr:$src)))),
            (VMOVNTDQAZrm addr:$src)>;
  def : Pat<(v64i8 (bitconvert (v8i64 (alignednontemporalload addr:$src)))),
            (VMOVNTDQAZrm addr:$src)>;
}

let Predicates = [HasVLX], AddedComplexity = 400 in {
  def : Pat<(alignednontemporalstore (v8i32 VR256X:$src), addr:$dst),
            (VMOVNTDQZ256mr addr:$dst, VR256X:$src)>;
  def : Pat<(alignednontemporalstore (v16i16 VR256X:$src), addr:$dst),
            (VMOVNTDQZ256mr addr:$dst, VR256X:$src)>;
  def : Pat<(alignednontemporalstore (v32i8 VR256X:$src), addr:$dst),
            (VMOVNTDQZ256mr addr:$dst, VR256X:$src)>;

  def : Pat<(v4f64 (alignednontemporalload addr:$src)),
            (VMOVNTDQAZ256rm addr:$src)>;
  def : Pat<(v8f32 (alignednontemporalload addr:$src)),
            (VMOVNTDQAZ256rm addr:$src)>;
  def : Pat<(v4i64 (alignednontemporalload addr:$src)),
            (VMOVNTDQAZ256rm addr:$src)>;
  def : Pat<(v8i32 (bitconvert (v2i64 (alignednontemporalload addr:$src)))),
            (VMOVNTDQAZ256rm addr:$src)>;
  def : Pat<(v16i16 (bitconvert (v2i64 (alignednontemporalload addr:$src)))),
            (VMOVNTDQAZ256rm addr:$src)>;
  def : Pat<(v32i8 (bitconvert (v2i64 (alignednontemporalload addr:$src)))),
            (VMOVNTDQAZ256rm addr:$src)>;

  def : Pat<(alignednontemporalstore (v4i32 VR128X:$src), addr:$dst),
            (VMOVNTDQZ128mr addr:$dst, VR128X:$src)>;
  def : Pat<(alignednontemporalstore (v8i16 VR128X:$src), addr:$dst),
            (VMOVNTDQZ128mr addr:$dst, VR128X:$src)>;
  def : Pat<(alignednontemporalstore (v16i8 VR128X:$src), addr:$dst),
            (VMOVNTDQZ128mr addr:$dst, VR128X:$src)>;

  def : Pat<(v2f64 (alignednontemporalload addr:$src)),
            (VMOVNTDQAZ128rm addr:$src)>;
  def : Pat<(v4f32 (alignednontemporalload addr:$src)),
            (VMOVNTDQAZ128rm addr:$src)>;
  def : Pat<(v2i64 (alignednontemporalload addr:$src)),
            (VMOVNTDQAZ128rm addr:$src)>;
  def : Pat<(v4i32 (bitconvert (v2i64 (alignednontemporalload addr:$src)))),
            (VMOVNTDQAZ128rm addr:$src)>;
  def : Pat<(v8i16 (bitconvert (v2i64 (alignednontemporalload addr:$src)))),
            (VMOVNTDQAZ128rm addr:$src)>;
  def : Pat<(v16i8 (bitconvert (v2i64 (alignednontemporalload addr:$src)))),
            (VMOVNTDQAZ128rm addr:$src)>;
}

//===----------------------------------------------------------------------===//
// AVX-512 - Integer arithmetic
//
multiclass avx512_binop_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           X86VectorVTInfo _, OpndItins itins,
                           bit IsCommutable = 0> {
  defm rr : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                    (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                    "$src2, $src1", "$src1, $src2",
                    (_.VT (OpNode _.RC:$src1, _.RC:$src2)),
                    itins.rr, IsCommutable>,
            AVX512BIBase, EVEX_4V;

  defm rm : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.MemOp:$src2), OpcodeStr,
                  "$src2, $src1", "$src1, $src2",
                  (_.VT (OpNode _.RC:$src1,
                                (bitconvert (_.LdFrag addr:$src2)))),
                  itins.rm>,
            AVX512BIBase, EVEX_4V;
}

multiclass avx512_binop_rmb<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _, OpndItins itins,
                            bit IsCommutable = 0> :
           avx512_binop_rm<opc, OpcodeStr, OpNode, _, itins, IsCommutable> {
  defm rmb : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr,
                  "${src2}"##_.BroadcastStr##", $src1",
                  "$src1, ${src2}"##_.BroadcastStr,
                  (_.VT (OpNode _.RC:$src1,
                                (X86VBroadcast
                                    (_.ScalarLdFrag addr:$src2)))),
                  itins.rm>,
             AVX512BIBase, EVEX_4V, EVEX_B;
}

multiclass avx512_binop_rm_vl<bits<8> opc, string OpcodeStr, SDNode OpNode,
                              AVX512VLVectorVTInfo VTInfo, OpndItins itins,
                              Predicate prd, bit IsCommutable = 0> {
  let Predicates = [prd] in
    defm Z : avx512_binop_rm<opc, OpcodeStr, OpNode, VTInfo.info512, itins,
                             IsCommutable>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_binop_rm<opc, OpcodeStr, OpNode, VTInfo.info256, itins,
                             IsCommutable>, EVEX_V256;
    defm Z128 : avx512_binop_rm<opc, OpcodeStr, OpNode, VTInfo.info128, itins,
                             IsCommutable>, EVEX_V128;
  }
}

multiclass avx512_binop_rmb_vl<bits<8> opc, string OpcodeStr, SDNode OpNode,
                               AVX512VLVectorVTInfo VTInfo, OpndItins itins,
                               Predicate prd, bit IsCommutable = 0> {
  let Predicates = [prd] in
    defm Z : avx512_binop_rmb<opc, OpcodeStr, OpNode, VTInfo.info512, itins,
                             IsCommutable>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_binop_rmb<opc, OpcodeStr, OpNode, VTInfo.info256, itins,
                             IsCommutable>, EVEX_V256;
    defm Z128 : avx512_binop_rmb<opc, OpcodeStr, OpNode, VTInfo.info128, itins,
                             IsCommutable>, EVEX_V128;
  }
}

multiclass avx512_binop_rm_vl_q<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                OpndItins itins, Predicate prd,
                                bit IsCommutable = 0> {
  defm NAME : avx512_binop_rmb_vl<opc, OpcodeStr, OpNode, avx512vl_i64_info,
                               itins, prd, IsCommutable>,
                               VEX_W, EVEX_CD8<64, CD8VF>;
}

multiclass avx512_binop_rm_vl_d<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                OpndItins itins, Predicate prd,
                                bit IsCommutable = 0> {
  defm NAME : avx512_binop_rmb_vl<opc, OpcodeStr, OpNode, avx512vl_i32_info,
                               itins, prd, IsCommutable>, EVEX_CD8<32, CD8VF>;
}

multiclass avx512_binop_rm_vl_w<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                OpndItins itins, Predicate prd,
                                bit IsCommutable = 0> {
  defm NAME : avx512_binop_rm_vl<opc, OpcodeStr, OpNode, avx512vl_i16_info,
                              itins, prd, IsCommutable>, EVEX_CD8<16, CD8VF>;
}

multiclass avx512_binop_rm_vl_b<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                OpndItins itins, Predicate prd,
                                bit IsCommutable = 0> {
  defm NAME : avx512_binop_rm_vl<opc, OpcodeStr, OpNode, avx512vl_i8_info,
                              itins, prd, IsCommutable>, EVEX_CD8<8, CD8VF>;
}

multiclass avx512_binop_rm_vl_dq<bits<8> opc_d, bits<8> opc_q, string OpcodeStr,
                                 SDNode OpNode, OpndItins itins, Predicate prd,
                                 bit IsCommutable = 0> {
  defm Q : avx512_binop_rm_vl_q<opc_q, OpcodeStr#"q", OpNode, itins, prd,
                                   IsCommutable>;

  defm D : avx512_binop_rm_vl_d<opc_d, OpcodeStr#"d", OpNode, itins, prd,
                                   IsCommutable>;
}

multiclass avx512_binop_rm_vl_bw<bits<8> opc_b, bits<8> opc_w, string OpcodeStr,
                                 SDNode OpNode, OpndItins itins, Predicate prd,
                                 bit IsCommutable = 0> {
  defm W : avx512_binop_rm_vl_w<opc_w, OpcodeStr#"w", OpNode, itins, prd,
                                   IsCommutable>;

  defm B : avx512_binop_rm_vl_b<opc_b, OpcodeStr#"b", OpNode, itins, prd,
                                   IsCommutable>;
}

multiclass avx512_binop_rm_vl_all<bits<8> opc_b, bits<8> opc_w,
                                  bits<8> opc_d, bits<8> opc_q,
                                  string OpcodeStr, SDNode OpNode,
                                  OpndItins itins, bit IsCommutable = 0> {
  defm NAME : avx512_binop_rm_vl_dq<opc_d, opc_q, OpcodeStr, OpNode,
                                    itins, HasAVX512, IsCommutable>,
              avx512_binop_rm_vl_bw<opc_b, opc_w, OpcodeStr, OpNode,
                                    itins, HasBWI, IsCommutable>;
}

multiclass avx512_binop_rm2<bits<8> opc, string OpcodeStr, OpndItins itins,
                            SDNode OpNode,X86VectorVTInfo _Src,
                            X86VectorVTInfo _Dst, X86VectorVTInfo _Brdct,
                            bit IsCommutable = 0> {
  defm rr : AVX512_maskable<opc, MRMSrcReg, _Dst, (outs _Dst.RC:$dst),
                            (ins _Src.RC:$src1, _Src.RC:$src2), OpcodeStr,
                            "$src2, $src1","$src1, $src2",
                            (_Dst.VT (OpNode
                                         (_Src.VT _Src.RC:$src1),
                                         (_Src.VT _Src.RC:$src2))),
                            itins.rr, IsCommutable>,
                            AVX512BIBase, EVEX_4V;
  defm rm : AVX512_maskable<opc, MRMSrcMem, _Dst, (outs _Dst.RC:$dst),
                        (ins _Src.RC:$src1, _Src.MemOp:$src2), OpcodeStr,
                        "$src2, $src1", "$src1, $src2",
                        (_Dst.VT (OpNode (_Src.VT _Src.RC:$src1),
                                      (bitconvert (_Src.LdFrag addr:$src2)))),
                        itins.rm>,
                        AVX512BIBase, EVEX_4V;

  defm rmb : AVX512_maskable<opc, MRMSrcMem, _Dst, (outs _Dst.RC:$dst),
                    (ins _Src.RC:$src1, _Brdct.ScalarMemOp:$src2),
                    OpcodeStr,
                    "${src2}"##_Brdct.BroadcastStr##", $src1",
                     "$src1, ${src2}"##_Brdct.BroadcastStr,
                    (_Dst.VT (OpNode (_Src.VT _Src.RC:$src1), (bitconvert
                                 (_Brdct.VT (X86VBroadcast
                                          (_Brdct.ScalarLdFrag addr:$src2)))))),
                    itins.rm>,
                    AVX512BIBase, EVEX_4V, EVEX_B;
}

defm VPADD : avx512_binop_rm_vl_all<0xFC, 0xFD, 0xFE, 0xD4, "vpadd", add,
                                    SSE_INTALU_ITINS_P, 1>;
defm VPSUB : avx512_binop_rm_vl_all<0xF8, 0xF9, 0xFA, 0xFB, "vpsub", sub,
                                    SSE_INTALU_ITINS_P, 0>;
defm VPADDS : avx512_binop_rm_vl_bw<0xEC, 0xED, "vpadds", X86adds,
                                    SSE_INTALU_ITINS_P, HasBWI, 1>;
defm VPSUBS : avx512_binop_rm_vl_bw<0xE8, 0xE9, "vpsubs", X86subs,
                                    SSE_INTALU_ITINS_P, HasBWI, 0>;
defm VPADDUS : avx512_binop_rm_vl_bw<0xDC, 0xDD, "vpaddus", X86addus,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>;
defm VPSUBUS : avx512_binop_rm_vl_bw<0xD8, 0xD9, "vpsubus", X86subus,
                                     SSE_INTALU_ITINS_P, HasBWI, 0>;
defm VPMULLD : avx512_binop_rm_vl_d<0x40, "vpmulld", mul,
                                    SSE_INTALU_ITINS_P, HasAVX512, 1>, T8PD;
defm VPMULLW : avx512_binop_rm_vl_w<0xD5, "vpmullw", mul,
                                    SSE_INTALU_ITINS_P, HasBWI, 1>;
defm VPMULLQ : avx512_binop_rm_vl_q<0x40, "vpmullq", mul,
                                    SSE_INTALU_ITINS_P, HasDQI, 1>, T8PD;
defm VPMULHW : avx512_binop_rm_vl_w<0xE5, "vpmulhw", mulhs, SSE_INTALU_ITINS_P,
                                    HasBWI, 1>;
defm VPMULHUW : avx512_binop_rm_vl_w<0xE4, "vpmulhuw", mulhu, SSE_INTMUL_ITINS_P,
                                     HasBWI, 1>;
defm VPMULHRSW : avx512_binop_rm_vl_w<0x0B, "vpmulhrsw", X86mulhrs, SSE_INTMUL_ITINS_P,
                                      HasBWI, 1>, T8PD;
defm VPAVG : avx512_binop_rm_vl_bw<0xE0, 0xE3, "vpavg", X86avg,
                                   SSE_INTALU_ITINS_P, HasBWI, 1>;

multiclass avx512_binop_all<bits<8> opc, string OpcodeStr, OpndItins itins,
                            AVX512VLVectorVTInfo _SrcVTInfo, AVX512VLVectorVTInfo _DstVTInfo,
                            SDNode OpNode, Predicate prd,  bit IsCommutable = 0> {
  let Predicates = [prd] in
    defm NAME#Z : avx512_binop_rm2<opc, OpcodeStr, itins, OpNode,
                                 _SrcVTInfo.info512, _DstVTInfo.info512,
                                 v8i64_info, IsCommutable>,
                                  EVEX_V512, EVEX_CD8<64, CD8VF>, VEX_W;
  let Predicates = [HasVLX, prd] in {
    defm NAME#Z256 : avx512_binop_rm2<opc, OpcodeStr, itins, OpNode,
                                      _SrcVTInfo.info256, _DstVTInfo.info256,
                                      v4i64x_info, IsCommutable>,
                                      EVEX_V256, EVEX_CD8<64, CD8VF>, VEX_W;
    defm NAME#Z128 : avx512_binop_rm2<opc, OpcodeStr, itins, OpNode,
                                      _SrcVTInfo.info128, _DstVTInfo.info128,
                                      v2i64x_info, IsCommutable>,
                                     EVEX_V128, EVEX_CD8<64, CD8VF>, VEX_W;
  }
}

defm VPMULDQ : avx512_binop_all<0x28, "vpmuldq", SSE_INTALU_ITINS_P,
                                avx512vl_i32_info, avx512vl_i64_info,
                                X86pmuldq, HasAVX512, 1>,T8PD;
defm VPMULUDQ : avx512_binop_all<0xF4, "vpmuludq", SSE_INTMUL_ITINS_P,
                                avx512vl_i32_info, avx512vl_i64_info,
                                X86pmuludq, HasAVX512, 1>;
defm VPMULTISHIFTQB : avx512_binop_all<0x83, "vpmultishiftqb", SSE_INTALU_ITINS_P,
                                avx512vl_i8_info, avx512vl_i8_info,
                                X86multishift, HasVBMI, 0>, T8PD;

multiclass avx512_packs_rmb<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _Src, X86VectorVTInfo _Dst> {
  defm rmb : AVX512_maskable<opc, MRMSrcMem, _Dst, (outs _Dst.RC:$dst),
                    (ins _Src.RC:$src1, _Src.ScalarMemOp:$src2),
                    OpcodeStr,
                    "${src2}"##_Src.BroadcastStr##", $src1",
                     "$src1, ${src2}"##_Src.BroadcastStr,
                    (_Dst.VT (OpNode (_Src.VT _Src.RC:$src1), (bitconvert
                                 (_Src.VT (X86VBroadcast
                                          (_Src.ScalarLdFrag addr:$src2))))))>,
                    EVEX_4V, EVEX_B, EVEX_CD8<_Src.EltSize, CD8VF>;
}

multiclass avx512_packs_rm<bits<8> opc, string OpcodeStr,
                            SDNode OpNode,X86VectorVTInfo _Src,
                            X86VectorVTInfo _Dst, bit IsCommutable = 0> {
  defm rr : AVX512_maskable<opc, MRMSrcReg, _Dst, (outs _Dst.RC:$dst),
                            (ins _Src.RC:$src1, _Src.RC:$src2), OpcodeStr,
                            "$src2, $src1","$src1, $src2",
                            (_Dst.VT (OpNode
                                         (_Src.VT _Src.RC:$src1),
                                         (_Src.VT _Src.RC:$src2))),
                            NoItinerary, IsCommutable>,
                            EVEX_CD8<_Src.EltSize, CD8VF>, EVEX_4V;
  defm rm : AVX512_maskable<opc, MRMSrcMem, _Dst, (outs _Dst.RC:$dst),
                        (ins _Src.RC:$src1, _Src.MemOp:$src2), OpcodeStr,
                        "$src2, $src1", "$src1, $src2",
                        (_Dst.VT (OpNode (_Src.VT _Src.RC:$src1),
                                      (bitconvert (_Src.LdFrag addr:$src2))))>,
                         EVEX_4V, EVEX_CD8<_Src.EltSize, CD8VF>;
}

multiclass avx512_packs_all_i32_i16<bits<8> opc, string OpcodeStr,
                                    SDNode OpNode> {
  let Predicates = [HasBWI] in
  defm NAME#Z : avx512_packs_rm<opc, OpcodeStr, OpNode, v16i32_info,
                                 v32i16_info>,
                avx512_packs_rmb<opc, OpcodeStr, OpNode, v16i32_info,
                                 v32i16_info>, EVEX_V512;
  let Predicates = [HasBWI, HasVLX] in {
    defm NAME#Z256 : avx512_packs_rm<opc, OpcodeStr, OpNode, v8i32x_info,
                                     v16i16x_info>,
                     avx512_packs_rmb<opc, OpcodeStr, OpNode, v8i32x_info,
                                     v16i16x_info>, EVEX_V256;
    defm NAME#Z128 : avx512_packs_rm<opc, OpcodeStr, OpNode, v4i32x_info,
                                     v8i16x_info>,
                     avx512_packs_rmb<opc, OpcodeStr, OpNode, v4i32x_info,
                                     v8i16x_info>, EVEX_V128;
  }
}
multiclass avx512_packs_all_i16_i8<bits<8> opc, string OpcodeStr,
                            SDNode OpNode> {
  let Predicates = [HasBWI] in
  defm NAME#Z : avx512_packs_rm<opc, OpcodeStr, OpNode, v32i16_info,
                                v64i8_info>, EVEX_V512;
  let Predicates = [HasBWI, HasVLX] in {
    defm NAME#Z256 : avx512_packs_rm<opc, OpcodeStr, OpNode, v16i16x_info,
                                    v32i8x_info>, EVEX_V256;
    defm NAME#Z128 : avx512_packs_rm<opc, OpcodeStr, OpNode, v8i16x_info,
                                    v16i8x_info>, EVEX_V128;
  }
}

multiclass avx512_vpmadd<bits<8> opc, string OpcodeStr,
                            SDNode OpNode, AVX512VLVectorVTInfo _Src,
                            AVX512VLVectorVTInfo _Dst, bit IsCommutable = 0> {
  let Predicates = [HasBWI] in
  defm NAME#Z : avx512_packs_rm<opc, OpcodeStr, OpNode, _Src.info512,
                                _Dst.info512, IsCommutable>, EVEX_V512;
  let Predicates = [HasBWI, HasVLX] in {
    defm NAME#Z256 : avx512_packs_rm<opc, OpcodeStr, OpNode, _Src.info256,
                                     _Dst.info256, IsCommutable>, EVEX_V256;
    defm NAME#Z128 : avx512_packs_rm<opc, OpcodeStr, OpNode, _Src.info128,
                                     _Dst.info128, IsCommutable>, EVEX_V128;
  }
}

defm VPACKSSDW : avx512_packs_all_i32_i16<0x6B, "vpackssdw", X86Packss>, AVX512BIBase;
defm VPACKUSDW : avx512_packs_all_i32_i16<0x2b, "vpackusdw", X86Packus>, AVX5128IBase;
defm VPACKSSWB : avx512_packs_all_i16_i8 <0x63, "vpacksswb", X86Packss>, AVX512BIBase;
defm VPACKUSWB : avx512_packs_all_i16_i8 <0x67, "vpackuswb", X86Packus>, AVX512BIBase;

defm VPMADDUBSW : avx512_vpmadd<0x04, "vpmaddubsw", X86vpmaddubsw,
                     avx512vl_i8_info, avx512vl_i16_info>, AVX512BIBase, T8PD;
defm VPMADDWD   : avx512_vpmadd<0xF5, "vpmaddwd", X86vpmaddwd,
                     avx512vl_i16_info, avx512vl_i32_info, 1>, AVX512BIBase;

defm VPMAXSB : avx512_binop_rm_vl_b<0x3C, "vpmaxsb", smax,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>, T8PD;
defm VPMAXSW : avx512_binop_rm_vl_w<0xEE, "vpmaxsw", smax,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>;
defm VPMAXS : avx512_binop_rm_vl_dq<0x3D, 0x3D, "vpmaxs", smax,
                                     SSE_INTALU_ITINS_P, HasAVX512, 1>, T8PD;

defm VPMAXUB : avx512_binop_rm_vl_b<0xDE, "vpmaxub", umax,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>;
defm VPMAXUW : avx512_binop_rm_vl_w<0x3E, "vpmaxuw", umax,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>, T8PD;
defm VPMAXU : avx512_binop_rm_vl_dq<0x3F, 0x3F, "vpmaxu", umax,
                                     SSE_INTALU_ITINS_P, HasAVX512, 1>, T8PD;

defm VPMINSB : avx512_binop_rm_vl_b<0x38, "vpminsb", smin,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>, T8PD;
defm VPMINSW : avx512_binop_rm_vl_w<0xEA, "vpminsw", smin,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>;
defm VPMINS : avx512_binop_rm_vl_dq<0x39, 0x39, "vpmins", smin,
                                     SSE_INTALU_ITINS_P, HasAVX512, 1>, T8PD;

defm VPMINUB : avx512_binop_rm_vl_b<0xDA, "vpminub", umin,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>;
defm VPMINUW : avx512_binop_rm_vl_w<0x3A, "vpminuw", umin,
                                     SSE_INTALU_ITINS_P, HasBWI, 1>, T8PD;
defm VPMINU : avx512_binop_rm_vl_dq<0x3B, 0x3B, "vpminu", umin,
                                     SSE_INTALU_ITINS_P, HasAVX512, 1>, T8PD;

// PMULLQ: Use 512bit version to implement 128/256 bit in case NoVLX.
let Predicates = [HasDQI, NoVLX] in {
  def : Pat<(v4i64 (mul (v4i64 VR256X:$src1), (v4i64 VR256X:$src2))),
            (EXTRACT_SUBREG
                (VPMULLQZrr
                    (INSERT_SUBREG (v8i64 (IMPLICIT_DEF)), VR256X:$src1, sub_ymm),
                    (INSERT_SUBREG (v8i64 (IMPLICIT_DEF)), VR256X:$src2, sub_ymm)),
             sub_ymm)>;

  def : Pat<(v2i64 (mul (v2i64 VR128X:$src1), (v2i64 VR128X:$src2))),
            (EXTRACT_SUBREG
                (VPMULLQZrr
                    (INSERT_SUBREG (v8i64 (IMPLICIT_DEF)), VR128X:$src1, sub_xmm),
                    (INSERT_SUBREG (v8i64 (IMPLICIT_DEF)), VR128X:$src2, sub_xmm)),
             sub_xmm)>;
}

//===----------------------------------------------------------------------===//
// AVX-512  Logical Instructions
//===----------------------------------------------------------------------===//

multiclass avx512_logic_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           X86VectorVTInfo _, bit IsCommutable = 0> {
  defm rr : AVX512_maskable_logic<opc, MRMSrcReg, _, (outs _.RC:$dst),
                    (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                    "$src2, $src1", "$src1, $src2",
                    (_.i64VT (OpNode (bitconvert (_.VT _.RC:$src1)),
                                     (bitconvert (_.VT _.RC:$src2)))),
                    (_.VT (bitconvert (_.i64VT (OpNode _.RC:$src1,
                                                       _.RC:$src2)))),
                    IIC_SSE_BIT_P_RR, IsCommutable>,
            AVX512BIBase, EVEX_4V;

  defm rm : AVX512_maskable_logic<opc, MRMSrcMem, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.MemOp:$src2), OpcodeStr,
                  "$src2, $src1", "$src1, $src2",
                  (_.i64VT (OpNode (bitconvert (_.VT _.RC:$src1)),
                                   (bitconvert (_.LdFrag addr:$src2)))),
                  (_.VT (bitconvert (_.i64VT (OpNode _.RC:$src1,
                                     (bitconvert (_.LdFrag addr:$src2)))))),
                  IIC_SSE_BIT_P_RM>,
            AVX512BIBase, EVEX_4V;
}

multiclass avx512_logic_rmb<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _, bit IsCommutable = 0> :
           avx512_logic_rm<opc, OpcodeStr, OpNode, _, IsCommutable> {
  defm rmb : AVX512_maskable_logic<opc, MRMSrcMem, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr,
                  "${src2}"##_.BroadcastStr##", $src1",
                  "$src1, ${src2}"##_.BroadcastStr,
                  (_.i64VT (OpNode _.RC:$src1,
                                   (bitconvert
                                    (_.VT (X86VBroadcast
                                            (_.ScalarLdFrag addr:$src2)))))),
                  (_.VT (bitconvert (_.i64VT (OpNode _.RC:$src1,
                                     (bitconvert
                                      (_.VT (X86VBroadcast
                                             (_.ScalarLdFrag addr:$src2)))))))),
                  IIC_SSE_BIT_P_RM>,
             AVX512BIBase, EVEX_4V, EVEX_B;
}

multiclass avx512_logic_rmb_vl<bits<8> opc, string OpcodeStr, SDNode OpNode,
                               AVX512VLVectorVTInfo VTInfo,
                               bit IsCommutable = 0> {
  let Predicates = [HasAVX512] in
    defm Z : avx512_logic_rmb<opc, OpcodeStr, OpNode, VTInfo.info512,
                             IsCommutable>, EVEX_V512;

  let Predicates = [HasAVX512, HasVLX] in {
    defm Z256 : avx512_logic_rmb<opc, OpcodeStr, OpNode, VTInfo.info256,
                             IsCommutable>, EVEX_V256;
    defm Z128 : avx512_logic_rmb<opc, OpcodeStr, OpNode, VTInfo.info128,
                             IsCommutable>, EVEX_V128;
  }
}

multiclass avx512_logic_rm_vl_d<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                bit IsCommutable = 0> {
  defm NAME : avx512_logic_rmb_vl<opc, OpcodeStr, OpNode, avx512vl_i32_info,
                                  IsCommutable>, EVEX_CD8<32, CD8VF>;
}

multiclass avx512_logic_rm_vl_q<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                bit IsCommutable = 0> {
  defm NAME : avx512_logic_rmb_vl<opc, OpcodeStr, OpNode, avx512vl_i64_info,
                                  IsCommutable>,
                                  VEX_W, EVEX_CD8<64, CD8VF>;
}

multiclass avx512_logic_rm_vl_dq<bits<8> opc_d, bits<8> opc_q, string OpcodeStr,
                                 SDNode OpNode, bit IsCommutable = 0> {
  defm Q : avx512_logic_rm_vl_q<opc_q, OpcodeStr#"q", OpNode, IsCommutable>;
  defm D : avx512_logic_rm_vl_d<opc_d, OpcodeStr#"d", OpNode, IsCommutable>;
}

defm VPAND : avx512_logic_rm_vl_dq<0xDB, 0xDB, "vpand", and, 1>;
defm VPOR : avx512_logic_rm_vl_dq<0xEB, 0xEB, "vpor", or, 1>;
defm VPXOR : avx512_logic_rm_vl_dq<0xEF, 0xEF, "vpxor", xor, 1>;
defm VPANDN : avx512_logic_rm_vl_dq<0xDF, 0xDF, "vpandn", X86andnp>;

//===----------------------------------------------------------------------===//
// AVX-512  FP arithmetic
//===----------------------------------------------------------------------===//
multiclass avx512_fp_scalar<bits<8> opc, string OpcodeStr,X86VectorVTInfo _,
                         SDNode OpNode, SDNode VecNode, OpndItins itins,
                         bit IsCommutable> {
  let ExeDomain = _.ExeDomain in {
  defm rr_Int : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                           (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                           "$src2, $src1", "$src1, $src2",
                           (_.VT (VecNode _.RC:$src1, _.RC:$src2,
                                          (i32 FROUND_CURRENT))),
                           itins.rr>;

  defm rm_Int : AVX512_maskable_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _.IntScalarMemOp:$src2), OpcodeStr,
                         "$src2, $src1", "$src1, $src2",
                         (_.VT (VecNode _.RC:$src1,
                                        _.ScalarIntMemCPat:$src2,
                                        (i32 FROUND_CURRENT))),
                         itins.rm>;
  let isCodeGenOnly = 1, Predicates = [HasAVX512] in {
  def rr : I< opc, MRMSrcReg, (outs _.FRC:$dst),
                         (ins _.FRC:$src1, _.FRC:$src2),
                          OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                          [(set _.FRC:$dst, (OpNode _.FRC:$src1, _.FRC:$src2))],
                          itins.rr> {
    let isCommutable = IsCommutable;
  }
  def rm : I< opc, MRMSrcMem, (outs _.FRC:$dst),
                         (ins _.FRC:$src1, _.ScalarMemOp:$src2),
                         OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                         [(set _.FRC:$dst, (OpNode _.FRC:$src1,
                         (_.ScalarLdFrag addr:$src2)))], itins.rm>;
  }
  }
}

multiclass avx512_fp_scalar_round<bits<8> opc, string OpcodeStr,X86VectorVTInfo _,
                         SDNode VecNode, OpndItins itins, bit IsCommutable = 0> {
  let ExeDomain = _.ExeDomain in
  defm rrb : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                          (ins _.RC:$src1, _.RC:$src2, AVX512RC:$rc), OpcodeStr,
                          "$rc, $src2, $src1", "$src1, $src2, $rc",
                          (VecNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                          (i32 imm:$rc)), itins.rr, IsCommutable>,
                          EVEX_B, EVEX_RC;
}
multiclass avx512_fp_scalar_sae<bits<8> opc, string OpcodeStr,X86VectorVTInfo _,
                                SDNode OpNode, SDNode VecNode, SDNode SaeNode,
                                OpndItins itins, bit IsCommutable> {
  let ExeDomain = _.ExeDomain in {
  defm rr_Int : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                           (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                           "$src2, $src1", "$src1, $src2",
                           (_.VT (VecNode _.RC:$src1, _.RC:$src2)),
                           itins.rr>;

  defm rm_Int : AVX512_maskable_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _.IntScalarMemOp:$src2), OpcodeStr,
                         "$src2, $src1", "$src1, $src2",
                         (_.VT (VecNode _.RC:$src1,
                                        _.ScalarIntMemCPat:$src2)),
                         itins.rm>;

  let isCodeGenOnly = 1, Predicates = [HasAVX512] in {
  def rr : I< opc, MRMSrcReg, (outs _.FRC:$dst),
                         (ins _.FRC:$src1, _.FRC:$src2),
                          OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                          [(set _.FRC:$dst, (OpNode _.FRC:$src1, _.FRC:$src2))],
                          itins.rr> {
    let isCommutable = IsCommutable;
  }
  def rm : I< opc, MRMSrcMem, (outs _.FRC:$dst),
                         (ins _.FRC:$src1, _.ScalarMemOp:$src2),
                         OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                         [(set _.FRC:$dst, (OpNode _.FRC:$src1,
                         (_.ScalarLdFrag addr:$src2)))], itins.rm>;
  }

  defm rrb : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                            (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                            "{sae}, $src2, $src1", "$src1, $src2, {sae}",
                            (SaeNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                            (i32 FROUND_NO_EXC))>, EVEX_B;
  }
}

multiclass avx512_binop_s_round<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  SDNode VecNode,
                                  SizeItins itins, bit IsCommutable> {
  defm SSZ : avx512_fp_scalar<opc, OpcodeStr#"ss", f32x_info, OpNode, VecNode,
                              itins.s, IsCommutable>,
             avx512_fp_scalar_round<opc, OpcodeStr#"ss", f32x_info, VecNode,
                              itins.s, IsCommutable>,
                              XS, EVEX_4V, VEX_LIG,  EVEX_CD8<32, CD8VT1>;
  defm SDZ : avx512_fp_scalar<opc, OpcodeStr#"sd", f64x_info, OpNode, VecNode,
                              itins.d,                  IsCommutable>,
             avx512_fp_scalar_round<opc, OpcodeStr#"sd", f64x_info, VecNode,
                              itins.d, IsCommutable>,
                              XD, VEX_W, EVEX_4V, VEX_LIG, EVEX_CD8<64, CD8VT1>;
}

multiclass avx512_binop_s_sae<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  SDNode VecNode, SDNode SaeNode,
                                  SizeItins itins, bit IsCommutable> {
  defm SSZ : avx512_fp_scalar_sae<opc, OpcodeStr#"ss", f32x_info, OpNode,
                              VecNode, SaeNode, itins.s, IsCommutable>,
                              XS, EVEX_4V, VEX_LIG,  EVEX_CD8<32, CD8VT1>;
  defm SDZ : avx512_fp_scalar_sae<opc, OpcodeStr#"sd", f64x_info, OpNode,
                              VecNode, SaeNode, itins.d, IsCommutable>,
                              XD, VEX_W, EVEX_4V, VEX_LIG, EVEX_CD8<64, CD8VT1>;
}
defm VADD : avx512_binop_s_round<0x58, "vadd", fadd, X86faddRnds, SSE_ALU_ITINS_S, 1>;
defm VMUL : avx512_binop_s_round<0x59, "vmul", fmul, X86fmulRnds, SSE_MUL_ITINS_S, 1>;
defm VSUB : avx512_binop_s_round<0x5C, "vsub", fsub, X86fsubRnds, SSE_ALU_ITINS_S, 0>;
defm VDIV : avx512_binop_s_round<0x5E, "vdiv", fdiv, X86fdivRnds, SSE_DIV_ITINS_S, 0>;
defm VMIN : avx512_binop_s_sae  <0x5D, "vmin", X86fmin, X86fmins, X86fminRnds,
                                 SSE_ALU_ITINS_S, 0>;
defm VMAX : avx512_binop_s_sae  <0x5F, "vmax", X86fmax, X86fmaxs, X86fmaxRnds,
                                 SSE_ALU_ITINS_S, 0>;

// MIN/MAX nodes are commutable under "unsafe-fp-math". In this case we use
// X86fminc and X86fmaxc instead of X86fmin and X86fmax
multiclass avx512_comutable_binop_s<bits<8> opc, string OpcodeStr,
                          X86VectorVTInfo _, SDNode OpNode, OpndItins itins> {
  let isCodeGenOnly = 1, Predicates = [HasAVX512], ExeDomain = _.ExeDomain in {
  def rr : I< opc, MRMSrcReg, (outs _.FRC:$dst),
                         (ins _.FRC:$src1, _.FRC:$src2),
                          OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                          [(set _.FRC:$dst, (OpNode _.FRC:$src1, _.FRC:$src2))],
                          itins.rr> {
    let isCommutable = 1;
  }
  def rm : I< opc, MRMSrcMem, (outs _.FRC:$dst),
                         (ins _.FRC:$src1, _.ScalarMemOp:$src2),
                         OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                         [(set _.FRC:$dst, (OpNode _.FRC:$src1,
                         (_.ScalarLdFrag addr:$src2)))], itins.rm>;
  }
}
defm VMINCSSZ : avx512_comutable_binop_s<0x5D, "vminss", f32x_info, X86fminc,
                                SSE_ALU_ITINS_S.s>, XS, EVEX_4V, VEX_LIG,
                                EVEX_CD8<32, CD8VT1>;

defm VMINCSDZ : avx512_comutable_binop_s<0x5D, "vminsd", f64x_info, X86fminc,
                                SSE_ALU_ITINS_S.d>, XD, VEX_W, EVEX_4V, VEX_LIG,
                                EVEX_CD8<64, CD8VT1>;

defm VMAXCSSZ : avx512_comutable_binop_s<0x5F, "vmaxss", f32x_info, X86fmaxc,
                                SSE_ALU_ITINS_S.s>, XS, EVEX_4V, VEX_LIG,
                                EVEX_CD8<32, CD8VT1>;

defm VMAXCSDZ : avx512_comutable_binop_s<0x5F, "vmaxsd", f64x_info, X86fmaxc,
                                SSE_ALU_ITINS_S.d>, XD, VEX_W, EVEX_4V, VEX_LIG,
                                EVEX_CD8<64, CD8VT1>;

multiclass avx512_fp_packed<bits<8> opc, string OpcodeStr, SDPatternOperator OpNode,
                            X86VectorVTInfo _, OpndItins itins,
                            bit IsCommutable> {
  let ExeDomain = _.ExeDomain, hasSideEffects = 0 in {
  defm rr: AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.RC:$src2), OpcodeStr##_.Suffix,
                  "$src2, $src1", "$src1, $src2",
                  (_.VT (OpNode _.RC:$src1, _.RC:$src2)), itins.rr,
                  IsCommutable>, EVEX_4V;
  let mayLoad = 1 in {
    defm rm: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.RC:$src1, _.MemOp:$src2), OpcodeStr##_.Suffix,
                    "$src2, $src1", "$src1, $src2",
                    (OpNode _.RC:$src1, (_.LdFrag addr:$src2)), itins.rm>,
                    EVEX_4V;
    defm rmb: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                     (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr##_.Suffix,
                     "${src2}"##_.BroadcastStr##", $src1",
                     "$src1, ${src2}"##_.BroadcastStr,
                     (OpNode  _.RC:$src1, (_.VT (X86VBroadcast
                                                (_.ScalarLdFrag addr:$src2)))),
                     itins.rm>, EVEX_4V, EVEX_B;
    }
  }
}

multiclass avx512_fp_round_packed<bits<8> opc, string OpcodeStr, SDPatternOperator OpNodeRnd,
                                  X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in
  defm rb: AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.RC:$src2, AVX512RC:$rc), OpcodeStr##_.Suffix,
                  "$rc, $src2, $src1", "$src1, $src2, $rc",
                  (_.VT (OpNodeRnd _.RC:$src1, _.RC:$src2, (i32 imm:$rc)))>,
                  EVEX_4V, EVEX_B, EVEX_RC;
}


multiclass avx512_fp_sae_packed<bits<8> opc, string OpcodeStr, SDPatternOperator OpNodeRnd,
                                X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in
  defm rb: AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.RC:$src2), OpcodeStr##_.Suffix,
                  "{sae}, $src2, $src1", "$src1, $src2, {sae}",
                  (_.VT (OpNodeRnd _.RC:$src1, _.RC:$src2, (i32 FROUND_NO_EXC)))>,
                  EVEX_4V, EVEX_B;
}

multiclass avx512_fp_binop_p<bits<8> opc, string OpcodeStr, SDPatternOperator OpNode,
                             Predicate prd, SizeItins itins,
                             bit IsCommutable = 0> {
  let Predicates = [prd] in {
  defm PSZ : avx512_fp_packed<opc, OpcodeStr, OpNode, v16f32_info,
                              itins.s, IsCommutable>, EVEX_V512, PS,
                              EVEX_CD8<32, CD8VF>;
  defm PDZ : avx512_fp_packed<opc, OpcodeStr, OpNode, v8f64_info,
                              itins.d, IsCommutable>, EVEX_V512, PD, VEX_W,
                              EVEX_CD8<64, CD8VF>;
  }

    // Define only if AVX512VL feature is present.
  let Predicates = [prd, HasVLX] in {
    defm PSZ128 : avx512_fp_packed<opc, OpcodeStr, OpNode, v4f32x_info,
                                   itins.s, IsCommutable>, EVEX_V128, PS,
                                   EVEX_CD8<32, CD8VF>;
    defm PSZ256 : avx512_fp_packed<opc, OpcodeStr, OpNode, v8f32x_info,
                                   itins.s, IsCommutable>, EVEX_V256, PS,
                                   EVEX_CD8<32, CD8VF>;
    defm PDZ128 : avx512_fp_packed<opc, OpcodeStr, OpNode, v2f64x_info,
                                   itins.d, IsCommutable>, EVEX_V128, PD, VEX_W,
                                   EVEX_CD8<64, CD8VF>;
    defm PDZ256 : avx512_fp_packed<opc, OpcodeStr, OpNode, v4f64x_info,
                                   itins.d, IsCommutable>, EVEX_V256, PD, VEX_W,
                                   EVEX_CD8<64, CD8VF>;
  }
}

multiclass avx512_fp_binop_p_round<bits<8> opc, string OpcodeStr, SDNode OpNodeRnd> {
  defm PSZ : avx512_fp_round_packed<opc, OpcodeStr, OpNodeRnd, v16f32_info>,
                              EVEX_V512, PS, EVEX_CD8<32, CD8VF>;
  defm PDZ : avx512_fp_round_packed<opc, OpcodeStr, OpNodeRnd, v8f64_info>,
                              EVEX_V512, PD, VEX_W,EVEX_CD8<64, CD8VF>;
}

multiclass avx512_fp_binop_p_sae<bits<8> opc, string OpcodeStr, SDNode OpNodeRnd> {
  defm PSZ : avx512_fp_sae_packed<opc, OpcodeStr, OpNodeRnd, v16f32_info>,
                              EVEX_V512, PS, EVEX_CD8<32, CD8VF>;
  defm PDZ : avx512_fp_sae_packed<opc, OpcodeStr, OpNodeRnd, v8f64_info>,
                              EVEX_V512, PD, VEX_W,EVEX_CD8<64, CD8VF>;
}

defm VADD : avx512_fp_binop_p<0x58, "vadd", fadd, HasAVX512,
                              SSE_ALU_ITINS_P, 1>,
            avx512_fp_binop_p_round<0x58, "vadd", X86faddRnd>;
defm VMUL : avx512_fp_binop_p<0x59, "vmul", fmul, HasAVX512,
                              SSE_MUL_ITINS_P, 1>,
            avx512_fp_binop_p_round<0x59, "vmul", X86fmulRnd>;
defm VSUB : avx512_fp_binop_p<0x5C, "vsub", fsub, HasAVX512, SSE_ALU_ITINS_P>,
            avx512_fp_binop_p_round<0x5C, "vsub", X86fsubRnd>;
defm VDIV : avx512_fp_binop_p<0x5E, "vdiv", fdiv, HasAVX512, SSE_DIV_ITINS_P>,
            avx512_fp_binop_p_round<0x5E, "vdiv", X86fdivRnd>;
defm VMIN : avx512_fp_binop_p<0x5D, "vmin", X86fmin, HasAVX512,
                              SSE_ALU_ITINS_P, 0>,
            avx512_fp_binop_p_sae<0x5D, "vmin", X86fminRnd>;
defm VMAX : avx512_fp_binop_p<0x5F, "vmax", X86fmax, HasAVX512,
                              SSE_ALU_ITINS_P, 0>,
            avx512_fp_binop_p_sae<0x5F, "vmax", X86fmaxRnd>;
let isCodeGenOnly = 1 in {
  defm VMINC : avx512_fp_binop_p<0x5D, "vmin", X86fminc, HasAVX512,
                                 SSE_ALU_ITINS_P, 1>;
  defm VMAXC : avx512_fp_binop_p<0x5F, "vmax", X86fmaxc, HasAVX512,
                                 SSE_ALU_ITINS_P, 1>;
}
defm VAND  : avx512_fp_binop_p<0x54, "vand", null_frag, HasDQI,
                               SSE_ALU_ITINS_P, 1>;
defm VANDN : avx512_fp_binop_p<0x55, "vandn", null_frag, HasDQI,
                               SSE_ALU_ITINS_P, 0>;
defm VOR   : avx512_fp_binop_p<0x56, "vor", null_frag, HasDQI,
                               SSE_ALU_ITINS_P, 1>;
defm VXOR  : avx512_fp_binop_p<0x57, "vxor", null_frag, HasDQI,
                               SSE_ALU_ITINS_P, 1>;

// Patterns catch floating point selects with bitcasted integer logic ops.
multiclass avx512_fp_logical_lowering<string InstrStr, SDNode OpNode,
                                      X86VectorVTInfo _, Predicate prd> {
let Predicates = [prd] in {
  // Masked register-register logical operations.
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (bitconvert (_.i64VT (OpNode _.RC:$src1, _.RC:$src2))),
                   _.RC:$src0)),
            (!cast<Instruction>(InstrStr#rrk) _.RC:$src0, _.KRCWM:$mask,
             _.RC:$src1, _.RC:$src2)>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (bitconvert (_.i64VT (OpNode _.RC:$src1, _.RC:$src2))),
                   _.ImmAllZerosV)),
            (!cast<Instruction>(InstrStr#rrkz) _.KRCWM:$mask, _.RC:$src1,
             _.RC:$src2)>;
  // Masked register-memory logical operations.
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (bitconvert (_.i64VT (OpNode _.RC:$src1,
                                         (load addr:$src2)))),
                   _.RC:$src0)),
            (!cast<Instruction>(InstrStr#rmk) _.RC:$src0, _.KRCWM:$mask,
             _.RC:$src1, addr:$src2)>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (bitconvert (_.i64VT (OpNode _.RC:$src1, (load addr:$src2)))),
                   _.ImmAllZerosV)),
            (!cast<Instruction>(InstrStr#rmkz) _.KRCWM:$mask, _.RC:$src1,
             addr:$src2)>;
  // Register-broadcast logical operations.
  def : Pat<(_.i64VT (OpNode _.RC:$src1,
                      (bitconvert (_.VT (X86VBroadcast
                                         (_.ScalarLdFrag addr:$src2)))))),
            (!cast<Instruction>(InstrStr#rmb) _.RC:$src1, addr:$src2)>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (bitconvert
                    (_.i64VT (OpNode _.RC:$src1,
                              (bitconvert (_.VT
                                           (X86VBroadcast
                                            (_.ScalarLdFrag addr:$src2))))))),
                   _.RC:$src0)),
            (!cast<Instruction>(InstrStr#rmbk) _.RC:$src0, _.KRCWM:$mask,
             _.RC:$src1, addr:$src2)>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (bitconvert
                    (_.i64VT (OpNode _.RC:$src1,
                              (bitconvert (_.VT
                                           (X86VBroadcast
                                            (_.ScalarLdFrag addr:$src2))))))),
                   _.ImmAllZerosV)),
            (!cast<Instruction>(InstrStr#rmbkz)  _.KRCWM:$mask,
             _.RC:$src1, addr:$src2)>;
}
}

multiclass avx512_fp_logical_lowering_sizes<string InstrStr, SDNode OpNode> {
  defm : avx512_fp_logical_lowering<InstrStr#DZ128, OpNode, v4f32x_info, HasVLX>;
  defm : avx512_fp_logical_lowering<InstrStr#QZ128, OpNode, v2f64x_info, HasVLX>;
  defm : avx512_fp_logical_lowering<InstrStr#DZ256, OpNode, v8f32x_info, HasVLX>;
  defm : avx512_fp_logical_lowering<InstrStr#QZ256, OpNode, v4f64x_info, HasVLX>;
  defm : avx512_fp_logical_lowering<InstrStr#DZ, OpNode, v16f32_info, HasAVX512>;
  defm : avx512_fp_logical_lowering<InstrStr#QZ, OpNode, v8f64_info, HasAVX512>;
}

defm : avx512_fp_logical_lowering_sizes<"VPAND", and>;
defm : avx512_fp_logical_lowering_sizes<"VPOR", or>;
defm : avx512_fp_logical_lowering_sizes<"VPXOR", xor>;
defm : avx512_fp_logical_lowering_sizes<"VPANDN", X86andnp>;

let Predicates = [HasVLX,HasDQI] in {
  // Use packed logical operations for scalar ops.
  def : Pat<(f64 (X86fand FR64X:$src1, FR64X:$src2)),
            (COPY_TO_REGCLASS (VANDPDZ128rr
                               (COPY_TO_REGCLASS FR64X:$src1, VR128X),
                               (COPY_TO_REGCLASS FR64X:$src2, VR128X)), FR64X)>;
  def : Pat<(f64 (X86for FR64X:$src1, FR64X:$src2)),
            (COPY_TO_REGCLASS (VORPDZ128rr
                               (COPY_TO_REGCLASS FR64X:$src1, VR128X),
                               (COPY_TO_REGCLASS FR64X:$src2, VR128X)), FR64X)>;
  def : Pat<(f64 (X86fxor FR64X:$src1, FR64X:$src2)),
            (COPY_TO_REGCLASS (VXORPDZ128rr
                               (COPY_TO_REGCLASS FR64X:$src1, VR128X),
                               (COPY_TO_REGCLASS FR64X:$src2, VR128X)), FR64X)>;
  def : Pat<(f64 (X86fandn FR64X:$src1, FR64X:$src2)),
            (COPY_TO_REGCLASS (VANDNPDZ128rr
                               (COPY_TO_REGCLASS FR64X:$src1, VR128X),
                               (COPY_TO_REGCLASS FR64X:$src2, VR128X)), FR64X)>;

  def : Pat<(f32 (X86fand FR32X:$src1, FR32X:$src2)),
            (COPY_TO_REGCLASS (VANDPSZ128rr
                               (COPY_TO_REGCLASS FR32X:$src1, VR128X),
                               (COPY_TO_REGCLASS FR32X:$src2, VR128X)), FR32X)>;
  def : Pat<(f32 (X86for FR32X:$src1, FR32X:$src2)),
            (COPY_TO_REGCLASS (VORPSZ128rr
                               (COPY_TO_REGCLASS FR32X:$src1, VR128X),
                               (COPY_TO_REGCLASS FR32X:$src2, VR128X)), FR32X)>;
  def : Pat<(f32 (X86fxor FR32X:$src1, FR32X:$src2)),
            (COPY_TO_REGCLASS (VXORPSZ128rr
                               (COPY_TO_REGCLASS FR32X:$src1, VR128X),
                               (COPY_TO_REGCLASS FR32X:$src2, VR128X)), FR32X)>;
  def : Pat<(f32 (X86fandn FR32X:$src1, FR32X:$src2)),
            (COPY_TO_REGCLASS (VANDNPSZ128rr
                               (COPY_TO_REGCLASS FR32X:$src1, VR128X),
                               (COPY_TO_REGCLASS FR32X:$src2, VR128X)), FR32X)>;
}

multiclass avx512_fp_scalef_p<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
  defm rr: AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.RC:$src2), OpcodeStr##_.Suffix,
                  "$src2, $src1", "$src1, $src2",
                  (_.VT (OpNode _.RC:$src1, _.RC:$src2, (i32 FROUND_CURRENT)))>, EVEX_4V;
  defm rm: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.MemOp:$src2), OpcodeStr##_.Suffix,
                  "$src2, $src1", "$src1, $src2",
                  (OpNode _.RC:$src1, (_.LdFrag addr:$src2), (i32 FROUND_CURRENT))>, EVEX_4V;
  defm rmb: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                   (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr##_.Suffix,
                   "${src2}"##_.BroadcastStr##", $src1",
                   "$src1, ${src2}"##_.BroadcastStr,
                   (OpNode  _.RC:$src1, (_.VT (X86VBroadcast
                                              (_.ScalarLdFrag addr:$src2))), (i32 FROUND_CURRENT))>,
                   EVEX_4V, EVEX_B;
  }
}

multiclass avx512_fp_scalef_scalar<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
  defm rr: AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.RC:$src2), OpcodeStr##_.Suffix,
                  "$src2, $src1", "$src1, $src2",
                  (_.VT (OpNode _.RC:$src1, _.RC:$src2, (i32 FROUND_CURRENT)))>;
  defm rm: AVX512_maskable_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr##_.Suffix,
                  "$src2, $src1", "$src1, $src2",
                  (OpNode _.RC:$src1,
                          (_.VT (scalar_to_vector (_.ScalarLdFrag addr:$src2))),
                          (i32 FROUND_CURRENT))>;
  }
}

multiclass avx512_fp_scalef_all<bits<8> opc, bits<8> opcScaler, string OpcodeStr, SDNode OpNode, SDNode OpNodeScal> {
  defm PSZ : avx512_fp_scalef_p<opc, OpcodeStr, OpNode, v16f32_info>,
             avx512_fp_round_packed<opc, OpcodeStr, OpNode, v16f32_info>,
                              EVEX_V512, EVEX_CD8<32, CD8VF>;
  defm PDZ : avx512_fp_scalef_p<opc, OpcodeStr, OpNode, v8f64_info>,
             avx512_fp_round_packed<opc, OpcodeStr, OpNode, v8f64_info>,
                              EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
  defm SSZ128 : avx512_fp_scalef_scalar<opcScaler, OpcodeStr, OpNodeScal, f32x_info>,
                avx512_fp_scalar_round<opcScaler, OpcodeStr##"ss", f32x_info, OpNodeScal, SSE_ALU_ITINS_S.s>,
                              EVEX_4V,EVEX_CD8<32, CD8VT1>;
  defm SDZ128 : avx512_fp_scalef_scalar<opcScaler, OpcodeStr, OpNodeScal, f64x_info>,
                avx512_fp_scalar_round<opcScaler, OpcodeStr##"sd", f64x_info, OpNodeScal, SSE_ALU_ITINS_S.d>,
                              EVEX_4V, EVEX_CD8<64, CD8VT1>, VEX_W;

  // Define only if AVX512VL feature is present.
  let Predicates = [HasVLX] in {
    defm PSZ128 : avx512_fp_scalef_p<opc, OpcodeStr, OpNode, v4f32x_info>,
                                   EVEX_V128, EVEX_CD8<32, CD8VF>;
    defm PSZ256 : avx512_fp_scalef_p<opc, OpcodeStr, OpNode, v8f32x_info>,
                                   EVEX_V256, EVEX_CD8<32, CD8VF>;
    defm PDZ128 : avx512_fp_scalef_p<opc, OpcodeStr, OpNode, v2f64x_info>,
                                   EVEX_V128, VEX_W, EVEX_CD8<64, CD8VF>;
    defm PDZ256 : avx512_fp_scalef_p<opc, OpcodeStr, OpNode, v4f64x_info>,
                                   EVEX_V256, VEX_W, EVEX_CD8<64, CD8VF>;
  }
}
defm VSCALEF : avx512_fp_scalef_all<0x2C, 0x2D, "vscalef", X86scalef, X86scalefs>, T8PD;

//===----------------------------------------------------------------------===//
// AVX-512  VPTESTM instructions
//===----------------------------------------------------------------------===//

multiclass avx512_vptest<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _> {
  let isCommutable = 1 in
  defm rr : AVX512_maskable_cmp<opc, MRMSrcReg, _, (outs _.KRC:$dst),
                   (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                      "$src2, $src1", "$src1, $src2",
                   (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2))>,
                    EVEX_4V;
  defm rm : AVX512_maskable_cmp<opc, MRMSrcMem, _, (outs _.KRC:$dst),
                   (ins _.RC:$src1, _.MemOp:$src2), OpcodeStr,
                       "$src2, $src1", "$src1, $src2",
                   (OpNode (_.VT _.RC:$src1),
                    (_.VT (bitconvert (_.LdFrag addr:$src2))))>,
                    EVEX_4V,
                   EVEX_CD8<_.EltSize, CD8VF>;
}

multiclass avx512_vptest_mb<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _> {
  defm rmb : AVX512_maskable_cmp<opc, MRMSrcMem, _, (outs _.KRC:$dst),
                    (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr,
                    "${src2}"##_.BroadcastStr##", $src1",
                    "$src1, ${src2}"##_.BroadcastStr,
                    (OpNode (_.VT _.RC:$src1), (_.VT (X86VBroadcast
                                                (_.ScalarLdFrag addr:$src2))))>,
                    EVEX_B, EVEX_4V, EVEX_CD8<_.EltSize, CD8VF>;
}

// Use 512bit version to implement 128/256 bit in case NoVLX.
multiclass avx512_vptest_lowering<SDNode OpNode, X86VectorVTInfo ExtendInfo,
                                  X86VectorVTInfo _, string Suffix> {
    def : Pat<(_.KVT (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2))),
              (_.KVT (COPY_TO_REGCLASS
                       (!cast<Instruction>(NAME # Suffix # "Zrr")
                         (INSERT_SUBREG (ExtendInfo.VT (IMPLICIT_DEF)),
                                        _.RC:$src1, _.SubRegIdx),
                         (INSERT_SUBREG (ExtendInfo.VT (IMPLICIT_DEF)),
                                        _.RC:$src2, _.SubRegIdx)),
                     _.KRC))>;
}

multiclass avx512_vptest_dq_sizes<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  AVX512VLVectorVTInfo _, string Suffix> {
  let Predicates  = [HasAVX512] in
  defm Z : avx512_vptest<opc, OpcodeStr, OpNode, _.info512>,
           avx512_vptest_mb<opc, OpcodeStr, OpNode, _.info512>, EVEX_V512;

  let Predicates = [HasAVX512, HasVLX] in {
  defm Z256 : avx512_vptest<opc, OpcodeStr, OpNode, _.info256>,
              avx512_vptest_mb<opc, OpcodeStr, OpNode, _.info256>, EVEX_V256;
  defm Z128 : avx512_vptest<opc, OpcodeStr, OpNode, _.info128>,
              avx512_vptest_mb<opc, OpcodeStr, OpNode, _.info128>, EVEX_V128;
  }
  let Predicates = [HasAVX512, NoVLX] in {
  defm Z256_Alt : avx512_vptest_lowering< OpNode, _.info512, _.info256, Suffix>;
  defm Z128_Alt : avx512_vptest_lowering< OpNode, _.info512, _.info128, Suffix>;
  }
}

multiclass avx512_vptest_dq<bits<8> opc, string OpcodeStr, SDNode OpNode> {
  defm D : avx512_vptest_dq_sizes<opc, OpcodeStr#"d", OpNode,
                                 avx512vl_i32_info, "D">;
  defm Q : avx512_vptest_dq_sizes<opc, OpcodeStr#"q", OpNode,
                                 avx512vl_i64_info, "Q">, VEX_W;
}

multiclass avx512_vptest_wb<bits<8> opc, string OpcodeStr,
                                 SDNode OpNode> {
  let Predicates = [HasBWI] in {
  defm WZ:    avx512_vptest<opc, OpcodeStr#"w", OpNode, v32i16_info>,
              EVEX_V512, VEX_W;
  defm BZ:    avx512_vptest<opc, OpcodeStr#"b", OpNode, v64i8_info>,
              EVEX_V512;
  }
  let Predicates = [HasVLX, HasBWI] in {

  defm WZ256: avx512_vptest<opc, OpcodeStr#"w", OpNode, v16i16x_info>,
              EVEX_V256, VEX_W;
  defm WZ128: avx512_vptest<opc, OpcodeStr#"w", OpNode, v8i16x_info>,
              EVEX_V128, VEX_W;
  defm BZ256: avx512_vptest<opc, OpcodeStr#"b", OpNode, v32i8x_info>,
              EVEX_V256;
  defm BZ128: avx512_vptest<opc, OpcodeStr#"b", OpNode, v16i8x_info>,
              EVEX_V128;
  }

  let Predicates = [HasAVX512, NoVLX] in {
  defm BZ256_Alt : avx512_vptest_lowering< OpNode, v64i8_info, v32i8x_info, "B">;
  defm BZ128_Alt : avx512_vptest_lowering< OpNode, v64i8_info, v16i8x_info, "B">;
  defm WZ256_Alt : avx512_vptest_lowering< OpNode, v32i16_info, v16i16x_info, "W">;
  defm WZ128_Alt : avx512_vptest_lowering< OpNode, v32i16_info, v8i16x_info, "W">;
  }

}

multiclass avx512_vptest_all_forms<bits<8> opc_wb, bits<8> opc_dq, string OpcodeStr,
                                   SDNode OpNode> :
  avx512_vptest_wb <opc_wb, OpcodeStr, OpNode>,
  avx512_vptest_dq<opc_dq, OpcodeStr, OpNode>;

defm VPTESTM   : avx512_vptest_all_forms<0x26, 0x27, "vptestm", X86testm>, T8PD;
defm VPTESTNM  : avx512_vptest_all_forms<0x26, 0x27, "vptestnm", X86testnm>, T8XS;


//===----------------------------------------------------------------------===//
// AVX-512  Shift instructions
//===----------------------------------------------------------------------===//
multiclass avx512_shift_rmi<bits<8> opc, Format ImmFormR, Format ImmFormM,
                         string OpcodeStr, SDNode OpNode, X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
  defm ri : AVX512_maskable<opc, ImmFormR, _, (outs _.RC:$dst),
                   (ins _.RC:$src1, u8imm:$src2), OpcodeStr,
                      "$src2, $src1", "$src1, $src2",
                   (_.VT (OpNode _.RC:$src1, (i8 imm:$src2))),
                   SSE_INTSHIFT_ITINS_P.rr>;
  defm mi : AVX512_maskable<opc, ImmFormM, _, (outs _.RC:$dst),
                   (ins _.MemOp:$src1, u8imm:$src2), OpcodeStr,
                       "$src2, $src1", "$src1, $src2",
                   (_.VT (OpNode (_.VT (bitconvert (_.LdFrag addr:$src1))),
                          (i8 imm:$src2))),
                   SSE_INTSHIFT_ITINS_P.rm>;
  }
}

multiclass avx512_shift_rmbi<bits<8> opc, Format ImmFormM,
                         string OpcodeStr, SDNode OpNode, X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in
  defm mbi : AVX512_maskable<opc, ImmFormM, _, (outs _.RC:$dst),
                   (ins _.ScalarMemOp:$src1, u8imm:$src2), OpcodeStr,
      "$src2, ${src1}"##_.BroadcastStr, "${src1}"##_.BroadcastStr##", $src2",
     (_.VT (OpNode (X86VBroadcast (_.ScalarLdFrag addr:$src1)), (i8 imm:$src2))),
     SSE_INTSHIFT_ITINS_P.rm>, EVEX_B;
}

multiclass avx512_shift_rrm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                         ValueType SrcVT, PatFrag bc_frag, X86VectorVTInfo _> {
   // src2 is always 128-bit
  let ExeDomain = _.ExeDomain in {
  defm rr : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                   (ins _.RC:$src1, VR128X:$src2), OpcodeStr,
                      "$src2, $src1", "$src1, $src2",
                   (_.VT (OpNode _.RC:$src1, (SrcVT VR128X:$src2))),
                   SSE_INTSHIFT_ITINS_P.rr>, AVX512BIBase, EVEX_4V;
  defm rm : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                   (ins _.RC:$src1, i128mem:$src2), OpcodeStr,
                       "$src2, $src1", "$src1, $src2",
                   (_.VT (OpNode _.RC:$src1, (bc_frag (loadv2i64 addr:$src2)))),
                   SSE_INTSHIFT_ITINS_P.rm>, AVX512BIBase,
                   EVEX_4V;
  }
}

multiclass avx512_shift_sizes<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  ValueType SrcVT, PatFrag bc_frag,
                                  AVX512VLVectorVTInfo VTInfo, Predicate prd> {
  let Predicates = [prd] in
  defm Z    : avx512_shift_rrm<opc, OpcodeStr, OpNode, SrcVT, bc_frag,
                            VTInfo.info512>, EVEX_V512,
                            EVEX_CD8<VTInfo.info512.EltSize, CD8VQ> ;
  let Predicates = [prd, HasVLX] in {
  defm Z256 : avx512_shift_rrm<opc, OpcodeStr, OpNode, SrcVT, bc_frag,
                            VTInfo.info256>, EVEX_V256,
                            EVEX_CD8<VTInfo.info256.EltSize, CD8VH>;
  defm Z128 : avx512_shift_rrm<opc, OpcodeStr, OpNode, SrcVT, bc_frag,
                            VTInfo.info128>, EVEX_V128,
                            EVEX_CD8<VTInfo.info128.EltSize, CD8VF>;
  }
}

multiclass avx512_shift_types<bits<8> opcd, bits<8> opcq, bits<8> opcw,
                              string OpcodeStr, SDNode OpNode> {
  defm D : avx512_shift_sizes<opcd, OpcodeStr#"d", OpNode, v4i32, bc_v4i32,
                                 avx512vl_i32_info, HasAVX512>;
  defm Q : avx512_shift_sizes<opcq, OpcodeStr#"q", OpNode, v2i64, bc_v2i64,
                                 avx512vl_i64_info, HasAVX512>, VEX_W;
  defm W : avx512_shift_sizes<opcw, OpcodeStr#"w", OpNode, v8i16, bc_v8i16,
                                 avx512vl_i16_info, HasBWI>;
}

multiclass avx512_shift_rmi_sizes<bits<8> opc, Format ImmFormR, Format ImmFormM,
                                 string OpcodeStr, SDNode OpNode,
                                 AVX512VLVectorVTInfo VTInfo> {
  let Predicates = [HasAVX512] in
  defm Z:    avx512_shift_rmi<opc, ImmFormR, ImmFormM, OpcodeStr, OpNode,
                              VTInfo.info512>,
             avx512_shift_rmbi<opc, ImmFormM, OpcodeStr, OpNode,
                              VTInfo.info512>, EVEX_V512;
  let Predicates = [HasAVX512, HasVLX] in {
  defm Z256: avx512_shift_rmi<opc, ImmFormR, ImmFormM, OpcodeStr, OpNode,
                              VTInfo.info256>,
             avx512_shift_rmbi<opc, ImmFormM, OpcodeStr, OpNode,
                              VTInfo.info256>, EVEX_V256;
  defm Z128: avx512_shift_rmi<opc, ImmFormR, ImmFormM, OpcodeStr, OpNode,
                              VTInfo.info128>,
             avx512_shift_rmbi<opc, ImmFormM, OpcodeStr, OpNode,
                              VTInfo.info128>, EVEX_V128;
  }
}

multiclass avx512_shift_rmi_w<bits<8> opcw,
                                 Format ImmFormR, Format ImmFormM,
                                 string OpcodeStr, SDNode OpNode> {
  let Predicates = [HasBWI] in
  defm WZ:    avx512_shift_rmi<opcw, ImmFormR, ImmFormM, OpcodeStr, OpNode,
                               v32i16_info>, EVEX_V512;
  let Predicates = [HasVLX, HasBWI] in {
  defm WZ256: avx512_shift_rmi<opcw, ImmFormR, ImmFormM, OpcodeStr, OpNode,
                               v16i16x_info>, EVEX_V256;
  defm WZ128: avx512_shift_rmi<opcw, ImmFormR, ImmFormM, OpcodeStr, OpNode,
                               v8i16x_info>, EVEX_V128;
  }
}

multiclass avx512_shift_rmi_dq<bits<8> opcd, bits<8> opcq,
                                 Format ImmFormR, Format ImmFormM,
                                 string OpcodeStr, SDNode OpNode> {
  defm D: avx512_shift_rmi_sizes<opcd, ImmFormR, ImmFormM, OpcodeStr#"d", OpNode,
                                 avx512vl_i32_info>, EVEX_CD8<32, CD8VF>;
  defm Q: avx512_shift_rmi_sizes<opcq, ImmFormR, ImmFormM, OpcodeStr#"q", OpNode,
                                 avx512vl_i64_info>, EVEX_CD8<64, CD8VF>, VEX_W;
}

defm VPSRL : avx512_shift_rmi_dq<0x72, 0x73, MRM2r, MRM2m, "vpsrl", X86vsrli>,
             avx512_shift_rmi_w<0x71, MRM2r, MRM2m, "vpsrlw", X86vsrli>, AVX512BIi8Base, EVEX_4V;

defm VPSLL : avx512_shift_rmi_dq<0x72, 0x73, MRM6r, MRM6m, "vpsll", X86vshli>,
             avx512_shift_rmi_w<0x71, MRM6r, MRM6m, "vpsllw", X86vshli>, AVX512BIi8Base, EVEX_4V;

defm VPSRA : avx512_shift_rmi_dq<0x72, 0x72, MRM4r, MRM4m, "vpsra", X86vsrai>,
             avx512_shift_rmi_w<0x71, MRM4r, MRM4m, "vpsraw", X86vsrai>, AVX512BIi8Base, EVEX_4V;

defm VPROR : avx512_shift_rmi_dq<0x72, 0x72, MRM0r, MRM0m, "vpror", X86vrotri>, AVX512BIi8Base, EVEX_4V;
defm VPROL : avx512_shift_rmi_dq<0x72, 0x72, MRM1r, MRM1m, "vprol", X86vrotli>, AVX512BIi8Base, EVEX_4V;

defm VPSLL : avx512_shift_types<0xF2, 0xF3, 0xF1, "vpsll", X86vshl>;
defm VPSRA : avx512_shift_types<0xE2, 0xE2, 0xE1, "vpsra", X86vsra>;
defm VPSRL : avx512_shift_types<0xD2, 0xD3, 0xD1, "vpsrl", X86vsrl>;

// Use 512bit VPSRA/VPSRAI version to implement v2i64/v4i64 in case NoVLX.
let Predicates = [HasAVX512, NoVLX] in {
  def : Pat<(v4i64 (X86vsra (v4i64 VR256X:$src1), (v2i64 VR128X:$src2))),
            (EXTRACT_SUBREG (v8i64
              (VPSRAQZrr
                (v8i64 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src1, sub_ymm)),
                 VR128X:$src2)), sub_ymm)>;

  def : Pat<(v2i64 (X86vsra (v2i64 VR128X:$src1), (v2i64 VR128X:$src2))),
            (EXTRACT_SUBREG (v8i64
              (VPSRAQZrr
                (v8i64 (INSERT_SUBREG (IMPLICIT_DEF), VR128X:$src1, sub_xmm)),
                 VR128X:$src2)), sub_xmm)>;

  def : Pat<(v4i64 (X86vsrai (v4i64 VR256X:$src1), (i8 imm:$src2))),
            (EXTRACT_SUBREG (v8i64
              (VPSRAQZri
                (v8i64 (INSERT_SUBREG (IMPLICIT_DEF), VR256X:$src1, sub_ymm)),
                 imm:$src2)), sub_ymm)>;

  def : Pat<(v2i64 (X86vsrai (v2i64 VR128X:$src1), (i8 imm:$src2))),
            (EXTRACT_SUBREG (v8i64
              (VPSRAQZri
                (v8i64 (INSERT_SUBREG (IMPLICIT_DEF), VR128X:$src1, sub_xmm)),
                 imm:$src2)), sub_xmm)>;
}

//===-------------------------------------------------------------------===//
// Variable Bit Shifts
//===-------------------------------------------------------------------===//
multiclass avx512_var_shift<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
  defm rr : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                   (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                      "$src2, $src1", "$src1, $src2",
                   (_.VT (OpNode _.RC:$src1, (_.VT _.RC:$src2))),
                   SSE_INTSHIFT_ITINS_P.rr>, AVX5128IBase, EVEX_4V;
  defm rm : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                   (ins _.RC:$src1, _.MemOp:$src2), OpcodeStr,
                       "$src2, $src1", "$src1, $src2",
                   (_.VT (OpNode _.RC:$src1,
                   (_.VT (bitconvert (_.LdFrag addr:$src2))))),
                   SSE_INTSHIFT_ITINS_P.rm>, AVX5128IBase, EVEX_4V,
                   EVEX_CD8<_.EltSize, CD8VF>;
  }
}

multiclass avx512_var_shift_mb<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in
  defm rmb : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr,
                    "${src2}"##_.BroadcastStr##", $src1",
                    "$src1, ${src2}"##_.BroadcastStr,
                    (_.VT (OpNode _.RC:$src1, (_.VT (X86VBroadcast
                                                (_.ScalarLdFrag addr:$src2))))),
                    SSE_INTSHIFT_ITINS_P.rm>, AVX5128IBase, EVEX_B,
                    EVEX_4V, EVEX_CD8<_.EltSize, CD8VF>;
}

multiclass avx512_var_shift_sizes<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  AVX512VLVectorVTInfo _> {
  let Predicates  = [HasAVX512] in
  defm Z : avx512_var_shift<opc, OpcodeStr, OpNode, _.info512>,
           avx512_var_shift_mb<opc, OpcodeStr, OpNode, _.info512>, EVEX_V512;

  let Predicates = [HasAVX512, HasVLX] in {
  defm Z256 : avx512_var_shift<opc, OpcodeStr, OpNode, _.info256>,
              avx512_var_shift_mb<opc, OpcodeStr, OpNode, _.info256>, EVEX_V256;
  defm Z128 : avx512_var_shift<opc, OpcodeStr, OpNode, _.info128>,
              avx512_var_shift_mb<opc, OpcodeStr, OpNode, _.info128>, EVEX_V128;
  }
}

multiclass avx512_var_shift_types<bits<8> opc, string OpcodeStr,
                                 SDNode OpNode> {
  defm D : avx512_var_shift_sizes<opc, OpcodeStr#"d", OpNode,
                                 avx512vl_i32_info>;
  defm Q : avx512_var_shift_sizes<opc, OpcodeStr#"q", OpNode,
                                 avx512vl_i64_info>, VEX_W;
}

// Use 512bit version to implement 128/256 bit in case NoVLX.
multiclass avx512_var_shift_lowering<AVX512VLVectorVTInfo _, string OpcodeStr,
                                     SDNode OpNode, list<Predicate> p> {
  let Predicates = p in {
  def : Pat<(_.info256.VT (OpNode (_.info256.VT _.info256.RC:$src1),
                                  (_.info256.VT _.info256.RC:$src2))),
            (EXTRACT_SUBREG
                (!cast<Instruction>(OpcodeStr#"Zrr")
                    (INSERT_SUBREG (_.info512.VT (IMPLICIT_DEF)), VR256X:$src1, sub_ymm),
                    (INSERT_SUBREG (_.info512.VT (IMPLICIT_DEF)), VR256X:$src2, sub_ymm)),
             sub_ymm)>;

  def : Pat<(_.info128.VT (OpNode (_.info128.VT _.info128.RC:$src1),
                                  (_.info128.VT _.info128.RC:$src2))),
            (EXTRACT_SUBREG
                (!cast<Instruction>(OpcodeStr#"Zrr")
                    (INSERT_SUBREG (_.info512.VT (IMPLICIT_DEF)), VR128X:$src1, sub_xmm),
                    (INSERT_SUBREG (_.info512.VT (IMPLICIT_DEF)), VR128X:$src2, sub_xmm)),
             sub_xmm)>;
  }
}
multiclass avx512_var_shift_w<bits<8> opc, string OpcodeStr,
                                 SDNode OpNode> {
  let Predicates = [HasBWI] in
  defm WZ:    avx512_var_shift<opc, OpcodeStr, OpNode, v32i16_info>,
              EVEX_V512, VEX_W;
  let Predicates = [HasVLX, HasBWI] in {

  defm WZ256: avx512_var_shift<opc, OpcodeStr, OpNode, v16i16x_info>,
              EVEX_V256, VEX_W;
  defm WZ128: avx512_var_shift<opc, OpcodeStr, OpNode, v8i16x_info>,
              EVEX_V128, VEX_W;
  }
}

defm VPSLLV : avx512_var_shift_types<0x47, "vpsllv", shl>,
              avx512_var_shift_w<0x12, "vpsllvw", shl>;

defm VPSRAV : avx512_var_shift_types<0x46, "vpsrav", sra>,
              avx512_var_shift_w<0x11, "vpsravw", sra>;

defm VPSRLV : avx512_var_shift_types<0x45, "vpsrlv", srl>,
              avx512_var_shift_w<0x10, "vpsrlvw", srl>;

defm VPRORV : avx512_var_shift_types<0x14, "vprorv", rotr>;
defm VPROLV : avx512_var_shift_types<0x15, "vprolv", rotl>;

defm : avx512_var_shift_lowering<avx512vl_i64_info, "VPSRAVQ", sra, [HasAVX512, NoVLX]>;
defm : avx512_var_shift_lowering<avx512vl_i16_info, "VPSLLVW", shl, [HasBWI, NoVLX]>;
defm : avx512_var_shift_lowering<avx512vl_i16_info, "VPSRAVW", sra, [HasBWI, NoVLX]>;
defm : avx512_var_shift_lowering<avx512vl_i16_info, "VPSRLVW", srl, [HasBWI, NoVLX]>;

// Special handing for handling VPSRAV intrinsics.
multiclass avx512_var_shift_int_lowering<string InstrStr, X86VectorVTInfo _,
                                         list<Predicate> p> {
  let Predicates = p in {
    def : Pat<(_.VT (X86vsrav _.RC:$src1, _.RC:$src2)),
              (!cast<Instruction>(InstrStr#_.ZSuffix#rr) _.RC:$src1,
               _.RC:$src2)>;
    def : Pat<(_.VT (X86vsrav _.RC:$src1, (bitconvert (_.LdFrag addr:$src2)))),
              (!cast<Instruction>(InstrStr#_.ZSuffix##rm)
               _.RC:$src1, addr:$src2)>;
    def : Pat<(_.VT (vselect _.KRCWM:$mask,
                     (X86vsrav _.RC:$src1, _.RC:$src2), _.RC:$src0)),
              (!cast<Instruction>(InstrStr#_.ZSuffix#rrk) _.RC:$src0,
               _.KRC:$mask, _.RC:$src1, _.RC:$src2)>;
    def : Pat<(_.VT (vselect _.KRCWM:$mask,
                     (X86vsrav _.RC:$src1, (bitconvert (_.LdFrag addr:$src2))),
                     _.RC:$src0)),
              (!cast<Instruction>(InstrStr#_.ZSuffix##rmk) _.RC:$src0,
               _.KRC:$mask, _.RC:$src1, addr:$src2)>;
    def : Pat<(_.VT (vselect _.KRCWM:$mask,
                     (X86vsrav _.RC:$src1, _.RC:$src2), _.ImmAllZerosV)),
              (!cast<Instruction>(InstrStr#_.ZSuffix#rrkz) _.KRC:$mask,
               _.RC:$src1, _.RC:$src2)>;
    def : Pat<(_.VT (vselect _.KRCWM:$mask,
                     (X86vsrav _.RC:$src1, (bitconvert (_.LdFrag addr:$src2))),
                     _.ImmAllZerosV)),
              (!cast<Instruction>(InstrStr#_.ZSuffix##rmkz) _.KRC:$mask,
               _.RC:$src1, addr:$src2)>;
  }
}

multiclass avx512_var_shift_int_lowering_mb<string InstrStr, X86VectorVTInfo _,
                                         list<Predicate> p> :
           avx512_var_shift_int_lowering<InstrStr, _, p> {
  let Predicates = p in {
    def : Pat<(_.VT (X86vsrav _.RC:$src1,
                     (X86VBroadcast (_.ScalarLdFrag addr:$src2)))),
              (!cast<Instruction>(InstrStr#_.ZSuffix##rmb)
               _.RC:$src1, addr:$src2)>;
    def : Pat<(_.VT (vselect _.KRCWM:$mask,
                     (X86vsrav _.RC:$src1,
                      (X86VBroadcast (_.ScalarLdFrag addr:$src2))),
                     _.RC:$src0)),
              (!cast<Instruction>(InstrStr#_.ZSuffix##rmbk) _.RC:$src0,
               _.KRC:$mask, _.RC:$src1, addr:$src2)>;
    def : Pat<(_.VT (vselect _.KRCWM:$mask,
                     (X86vsrav _.RC:$src1,
                      (X86VBroadcast (_.ScalarLdFrag addr:$src2))),
                     _.ImmAllZerosV)),
              (!cast<Instruction>(InstrStr#_.ZSuffix##rmbkz) _.KRC:$mask,
               _.RC:$src1, addr:$src2)>;
  }
}

defm : avx512_var_shift_int_lowering<"VPSRAVW", v8i16x_info, [HasVLX, HasBWI]>;
defm : avx512_var_shift_int_lowering<"VPSRAVW", v16i16x_info, [HasVLX, HasBWI]>;
defm : avx512_var_shift_int_lowering<"VPSRAVW", v32i16_info, [HasBWI]>;
defm : avx512_var_shift_int_lowering_mb<"VPSRAVD", v4i32x_info, [HasVLX]>;
defm : avx512_var_shift_int_lowering_mb<"VPSRAVD", v8i32x_info, [HasVLX]>;
defm : avx512_var_shift_int_lowering_mb<"VPSRAVD", v16i32_info, [HasAVX512]>;
defm : avx512_var_shift_int_lowering_mb<"VPSRAVQ", v2i64x_info, [HasVLX]>;
defm : avx512_var_shift_int_lowering_mb<"VPSRAVQ", v4i64x_info, [HasVLX]>;
defm : avx512_var_shift_int_lowering_mb<"VPSRAVQ", v8i64_info, [HasAVX512]>;

//===-------------------------------------------------------------------===//
// 1-src variable permutation VPERMW/D/Q
//===-------------------------------------------------------------------===//
multiclass avx512_vperm_dq_sizes<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  AVX512VLVectorVTInfo _> {
  let Predicates  = [HasAVX512] in
  defm Z : avx512_var_shift<opc, OpcodeStr, OpNode, _.info512>,
           avx512_var_shift_mb<opc, OpcodeStr, OpNode, _.info512>, EVEX_V512;

  let Predicates = [HasAVX512, HasVLX] in
  defm Z256 : avx512_var_shift<opc, OpcodeStr, OpNode, _.info256>,
              avx512_var_shift_mb<opc, OpcodeStr, OpNode, _.info256>, EVEX_V256;
}

multiclass avx512_vpermi_dq_sizes<bits<8> opc, Format ImmFormR, Format ImmFormM,
                                 string OpcodeStr, SDNode OpNode,
                                 AVX512VLVectorVTInfo VTInfo> {
  let Predicates = [HasAVX512] in
  defm Z:    avx512_shift_rmi<opc, ImmFormR, ImmFormM, OpcodeStr, OpNode,
                              VTInfo.info512>,
             avx512_shift_rmbi<opc, ImmFormM, OpcodeStr, OpNode,
                              VTInfo.info512>, EVEX_V512;
  let Predicates = [HasAVX512, HasVLX] in
  defm Z256: avx512_shift_rmi<opc, ImmFormR, ImmFormM, OpcodeStr, OpNode,
                              VTInfo.info256>,
             avx512_shift_rmbi<opc, ImmFormM, OpcodeStr, OpNode,
                              VTInfo.info256>, EVEX_V256;
}

multiclass avx512_vperm_bw<bits<8> opc, string OpcodeStr,
                              Predicate prd, SDNode OpNode,
                              AVX512VLVectorVTInfo _> {
  let Predicates = [prd] in
  defm Z:    avx512_var_shift<opc, OpcodeStr, OpNode, _.info512>,
              EVEX_V512 ;
  let Predicates = [HasVLX, prd] in {
  defm Z256: avx512_var_shift<opc, OpcodeStr, OpNode, _.info256>,
              EVEX_V256 ;
  defm Z128: avx512_var_shift<opc, OpcodeStr, OpNode, _.info128>,
              EVEX_V128 ;
  }
}

defm VPERMW  : avx512_vperm_bw<0x8D, "vpermw", HasBWI, X86VPermv,
                                  avx512vl_i16_info>, VEX_W;
defm VPERMB  : avx512_vperm_bw<0x8D, "vpermb", HasVBMI, X86VPermv,
                                  avx512vl_i8_info>;

defm VPERMD : avx512_vperm_dq_sizes<0x36, "vpermd", X86VPermv,
                                    avx512vl_i32_info>;
defm VPERMQ : avx512_vperm_dq_sizes<0x36, "vpermq", X86VPermv,
                                    avx512vl_i64_info>, VEX_W;
defm VPERMPS : avx512_vperm_dq_sizes<0x16, "vpermps", X86VPermv,
                                    avx512vl_f32_info>;
defm VPERMPD : avx512_vperm_dq_sizes<0x16, "vpermpd", X86VPermv,
                                    avx512vl_f64_info>, VEX_W;

defm VPERMQ : avx512_vpermi_dq_sizes<0x00, MRMSrcReg, MRMSrcMem, "vpermq",
                             X86VPermi, avx512vl_i64_info>,
                             EVEX, AVX512AIi8Base, EVEX_CD8<64, CD8VF>, VEX_W;
defm VPERMPD : avx512_vpermi_dq_sizes<0x01, MRMSrcReg, MRMSrcMem, "vpermpd",
                             X86VPermi, avx512vl_f64_info>,
                             EVEX, AVX512AIi8Base, EVEX_CD8<64, CD8VF>, VEX_W;
//===----------------------------------------------------------------------===//
// AVX-512 - VPERMIL
//===----------------------------------------------------------------------===//

multiclass avx512_permil_vec<bits<8> OpcVar, string OpcodeStr,  SDNode OpNode,
                             X86VectorVTInfo _, X86VectorVTInfo Ctrl> {
  defm rr: AVX512_maskable<OpcVar, MRMSrcReg, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, Ctrl.RC:$src2), OpcodeStr,
                  "$src2, $src1", "$src1, $src2",
                  (_.VT (OpNode _.RC:$src1,
                               (Ctrl.VT Ctrl.RC:$src2)))>,
                  T8PD, EVEX_4V;
  defm rm: AVX512_maskable<OpcVar, MRMSrcMem, _, (outs _.RC:$dst),
                  (ins _.RC:$src1, Ctrl.MemOp:$src2), OpcodeStr,
                  "$src2, $src1", "$src1, $src2",
                  (_.VT (OpNode
                           _.RC:$src1,
                           (Ctrl.VT (bitconvert(Ctrl.LdFrag addr:$src2)))))>,
                  T8PD, EVEX_4V, EVEX_CD8<_.EltSize, CD8VF>;
  defm rmb: AVX512_maskable<OpcVar, MRMSrcMem, _, (outs _.RC:$dst),
                   (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr,
                   "${src2}"##_.BroadcastStr##", $src1",
                   "$src1, ${src2}"##_.BroadcastStr,
                   (_.VT (OpNode
                            _.RC:$src1,
                            (Ctrl.VT (X86VBroadcast
                                       (Ctrl.ScalarLdFrag addr:$src2)))))>,
                   T8PD, EVEX_4V, EVEX_B, EVEX_CD8<_.EltSize, CD8VF>;
}

multiclass avx512_permil_vec_common<string OpcodeStr, bits<8> OpcVar,
                             AVX512VLVectorVTInfo _, AVX512VLVectorVTInfo Ctrl>{
  let Predicates = [HasAVX512] in {
    defm Z    : avx512_permil_vec<OpcVar, OpcodeStr, X86VPermilpv, _.info512,
                                  Ctrl.info512>, EVEX_V512;
  }
  let Predicates = [HasAVX512, HasVLX] in {
    defm Z128 : avx512_permil_vec<OpcVar, OpcodeStr, X86VPermilpv, _.info128,
                                  Ctrl.info128>, EVEX_V128;
    defm Z256 : avx512_permil_vec<OpcVar, OpcodeStr, X86VPermilpv, _.info256,
                                  Ctrl.info256>, EVEX_V256;
  }
}

multiclass avx512_permil<string OpcodeStr, bits<8> OpcImm, bits<8> OpcVar,
                         AVX512VLVectorVTInfo _, AVX512VLVectorVTInfo Ctrl>{

  defm NAME: avx512_permil_vec_common<OpcodeStr, OpcVar, _, Ctrl>;
  defm NAME: avx512_shift_rmi_sizes<OpcImm, MRMSrcReg, MRMSrcMem, OpcodeStr,
                                    X86VPermilpi, _>,
                    EVEX, AVX512AIi8Base, EVEX_CD8<_.info128.EltSize, CD8VF>;
}

let ExeDomain = SSEPackedSingle in
defm VPERMILPS : avx512_permil<"vpermilps", 0x04, 0x0C, avx512vl_f32_info,
                               avx512vl_i32_info>;
let ExeDomain = SSEPackedDouble in
defm VPERMILPD : avx512_permil<"vpermilpd", 0x05, 0x0D, avx512vl_f64_info,
                               avx512vl_i64_info>, VEX_W;
//===----------------------------------------------------------------------===//
// AVX-512 - VPSHUFD, VPSHUFLW, VPSHUFHW
//===----------------------------------------------------------------------===//

defm VPSHUFD : avx512_shift_rmi_sizes<0x70, MRMSrcReg, MRMSrcMem, "vpshufd",
                             X86PShufd, avx512vl_i32_info>,
                             EVEX, AVX512BIi8Base, EVEX_CD8<32, CD8VF>;
defm VPSHUFH : avx512_shift_rmi_w<0x70, MRMSrcReg, MRMSrcMem, "vpshufhw",
                                  X86PShufhw>, EVEX, AVX512XSIi8Base;
defm VPSHUFL : avx512_shift_rmi_w<0x70, MRMSrcReg, MRMSrcMem, "vpshuflw",
                                  X86PShuflw>, EVEX, AVX512XDIi8Base;

multiclass avx512_pshufb_sizes<bits<8> opc, string OpcodeStr, SDNode OpNode> {
  let Predicates = [HasBWI] in
  defm Z:    avx512_var_shift<opc, OpcodeStr, OpNode, v64i8_info>, EVEX_V512;

  let Predicates = [HasVLX, HasBWI] in {
  defm Z256: avx512_var_shift<opc, OpcodeStr, OpNode, v32i8x_info>, EVEX_V256;
  defm Z128: avx512_var_shift<opc, OpcodeStr, OpNode, v16i8x_info>, EVEX_V128;
  }
}

defm VPSHUFB: avx512_pshufb_sizes<0x00, "vpshufb", X86pshufb>;

//===----------------------------------------------------------------------===//
// Move Low to High and High to Low packed FP Instructions
//===----------------------------------------------------------------------===//
def VMOVLHPSZrr : AVX512PSI<0x16, MRMSrcReg, (outs VR128X:$dst),
          (ins VR128X:$src1, VR128X:$src2),
          "vmovlhps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          [(set VR128X:$dst, (v4f32 (X86Movlhps VR128X:$src1, VR128X:$src2)))],
           IIC_SSE_MOV_LH>, EVEX_4V;
def VMOVHLPSZrr : AVX512PSI<0x12, MRMSrcReg, (outs VR128X:$dst),
          (ins VR128X:$src1, VR128X:$src2),
          "vmovhlps\t{$src2, $src1, $dst|$dst, $src1, $src2}",
          [(set VR128X:$dst, (v4f32 (X86Movhlps VR128X:$src1, VR128X:$src2)))],
          IIC_SSE_MOV_LH>, EVEX_4V;

let Predicates = [HasAVX512] in {
  // MOVLHPS patterns
  def : Pat<(v4i32 (X86Movlhps VR128X:$src1, VR128X:$src2)),
            (VMOVLHPSZrr VR128X:$src1, VR128X:$src2)>;
  def : Pat<(v2i64 (X86Movlhps VR128X:$src1, VR128X:$src2)),
            (VMOVLHPSZrr (v2i64 VR128X:$src1), VR128X:$src2)>;

  // MOVHLPS patterns
  def : Pat<(v4i32 (X86Movhlps VR128X:$src1, VR128X:$src2)),
            (VMOVHLPSZrr VR128X:$src1, VR128X:$src2)>;
}

//===----------------------------------------------------------------------===//
// VMOVHPS/PD VMOVLPS Instructions
// All patterns was taken from SSS implementation.
//===----------------------------------------------------------------------===//
multiclass avx512_mov_hilo_packed<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in
  def rm : AVX512<opc, MRMSrcMem, (outs _.RC:$dst),
                  (ins _.RC:$src1, f64mem:$src2),
                  !strconcat(OpcodeStr,
                             "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                  [(set _.RC:$dst,
                     (OpNode _.RC:$src1,
                       (_.VT (bitconvert
                         (v2f64 (scalar_to_vector (loadf64 addr:$src2)))))))],
                  IIC_SSE_MOV_LH>, EVEX_4V;
}

defm VMOVHPSZ128 : avx512_mov_hilo_packed<0x16, "vmovhps", X86Movlhps,
                                  v4f32x_info>, EVEX_CD8<32, CD8VT2>, PS;
defm VMOVHPDZ128 : avx512_mov_hilo_packed<0x16, "vmovhpd", X86Movlhpd,
                                  v2f64x_info>, EVEX_CD8<64, CD8VT1>, PD, VEX_W;
defm VMOVLPSZ128 : avx512_mov_hilo_packed<0x12, "vmovlps", X86Movlps,
                                  v4f32x_info>, EVEX_CD8<32, CD8VT2>, PS;
defm VMOVLPDZ128 : avx512_mov_hilo_packed<0x12, "vmovlpd", X86Movlpd,
                                  v2f64x_info>, EVEX_CD8<64, CD8VT1>, PD, VEX_W;

let Predicates = [HasAVX512] in {
  // VMOVHPS patterns
  def : Pat<(X86Movlhps VR128X:$src1,
               (bc_v4f32 (v2i64 (scalar_to_vector (loadi64 addr:$src2))))),
          (VMOVHPSZ128rm VR128X:$src1, addr:$src2)>;
  def : Pat<(X86Movlhps VR128X:$src1,
               (bc_v4i32 (v2i64 (X86vzload addr:$src2)))),
          (VMOVHPSZ128rm VR128X:$src1, addr:$src2)>;
  // VMOVHPD patterns
  def : Pat<(v2f64 (X86Unpckl VR128X:$src1,
                    (scalar_to_vector (loadf64 addr:$src2)))),
           (VMOVHPDZ128rm VR128X:$src1, addr:$src2)>;
  def : Pat<(v2f64 (X86Unpckl VR128X:$src1,
                    (bc_v2f64 (v2i64 (scalar_to_vector (loadi64 addr:$src2)))))),
           (VMOVHPDZ128rm VR128X:$src1, addr:$src2)>;
  // VMOVLPS patterns
  def : Pat<(v4f32 (X86Movlps VR128X:$src1, (load addr:$src2))),
          (VMOVLPSZ128rm VR128X:$src1, addr:$src2)>;
  def : Pat<(v4i32 (X86Movlps VR128X:$src1, (load addr:$src2))),
          (VMOVLPSZ128rm VR128X:$src1, addr:$src2)>;
  // VMOVLPD patterns
  def : Pat<(v2f64 (X86Movlpd VR128X:$src1, (load addr:$src2))),
          (VMOVLPDZ128rm VR128X:$src1, addr:$src2)>;
  def : Pat<(v2i64 (X86Movlpd VR128X:$src1, (load addr:$src2))),
          (VMOVLPDZ128rm VR128X:$src1, addr:$src2)>;
  def : Pat<(v2f64 (X86Movsd VR128X:$src1,
                           (v2f64 (scalar_to_vector (loadf64 addr:$src2))))),
          (VMOVLPDZ128rm VR128X:$src1, addr:$src2)>;
}

def VMOVHPSZ128mr : AVX512PSI<0x17, MRMDestMem, (outs),
                       (ins f64mem:$dst, VR128X:$src),
                       "vmovhps\t{$src, $dst|$dst, $src}",
                       [(store (f64 (extractelt
                                     (X86Unpckh (bc_v2f64 (v4f32 VR128X:$src)),
                                                (bc_v2f64 (v4f32 VR128X:$src))),
                                     (iPTR 0))), addr:$dst)], IIC_SSE_MOV_LH>,
                       EVEX, EVEX_CD8<32, CD8VT2>;
def VMOVHPDZ128mr : AVX512PDI<0x17, MRMDestMem, (outs),
                       (ins f64mem:$dst, VR128X:$src),
                       "vmovhpd\t{$src, $dst|$dst, $src}",
                       [(store (f64 (extractelt
                                     (v2f64 (X86Unpckh VR128X:$src, VR128X:$src)),
                                     (iPTR 0))), addr:$dst)], IIC_SSE_MOV_LH>,
                       EVEX, EVEX_CD8<64, CD8VT1>, VEX_W;
def VMOVLPSZ128mr : AVX512PSI<0x13, MRMDestMem, (outs),
                       (ins f64mem:$dst, VR128X:$src),
                       "vmovlps\t{$src, $dst|$dst, $src}",
                       [(store (f64 (extractelt (bc_v2f64 (v4f32 VR128X:$src)),
                                     (iPTR 0))), addr:$dst)],
                                     IIC_SSE_MOV_LH>,
                       EVEX, EVEX_CD8<32, CD8VT2>;
def VMOVLPDZ128mr : AVX512PDI<0x13, MRMDestMem, (outs),
                       (ins f64mem:$dst, VR128X:$src),
                       "vmovlpd\t{$src, $dst|$dst, $src}",
                       [(store (f64 (extractelt (v2f64 VR128X:$src),
                                     (iPTR 0))), addr:$dst)],
                                     IIC_SSE_MOV_LH>,
                       EVEX, EVEX_CD8<64, CD8VT1>, VEX_W;

let Predicates = [HasAVX512] in {
  // VMOVHPD patterns
  def : Pat<(store (f64 (extractelt
                           (v2f64 (X86VPermilpi VR128X:$src, (i8 1))),
                           (iPTR 0))), addr:$dst),
           (VMOVHPDZ128mr addr:$dst, VR128X:$src)>;
  // VMOVLPS patterns
  def : Pat<(store (v4f32 (X86Movlps (load addr:$src1), VR128X:$src2)),
                   addr:$src1),
            (VMOVLPSZ128mr addr:$src1, VR128X:$src2)>;
  def : Pat<(store (v4i32 (X86Movlps
                   (bc_v4i32 (loadv2i64 addr:$src1)), VR128X:$src2)), addr:$src1),
            (VMOVLPSZ128mr addr:$src1, VR128X:$src2)>;
  // VMOVLPD patterns
  def : Pat<(store (v2f64 (X86Movlpd (load addr:$src1), VR128X:$src2)),
                   addr:$src1),
            (VMOVLPDZ128mr addr:$src1, VR128X:$src2)>;
  def : Pat<(store (v2i64 (X86Movlpd (load addr:$src1), VR128X:$src2)),
                   addr:$src1),
            (VMOVLPDZ128mr addr:$src1, VR128X:$src2)>;
}
//===----------------------------------------------------------------------===//
// FMA - Fused Multiply Operations
//

multiclass avx512_fma3p_213_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                               X86VectorVTInfo _, string Suff> {
  let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in {
  defm r: AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.RC:$src3),
          OpcodeStr, "$src3, $src2", "$src2, $src3",
          (_.VT (OpNode _.RC:$src2, _.RC:$src1, _.RC:$src3)), 1, 1>,
         AVX512FMA3Base;

  defm m: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.MemOp:$src3),
          OpcodeStr, "$src3, $src2", "$src2, $src3",
          (_.VT (OpNode _.RC:$src2, _.RC:$src1, (_.LdFrag addr:$src3))), 1, 0>,
          AVX512FMA3Base;

  defm mb: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
            (ins _.RC:$src2, _.ScalarMemOp:$src3),
            OpcodeStr,   !strconcat("${src3}", _.BroadcastStr,", $src2"),
            !strconcat("$src2, ${src3}", _.BroadcastStr ),
            (OpNode _.RC:$src2,
             _.RC:$src1,(_.VT (X86VBroadcast (_.ScalarLdFrag addr:$src3)))), 1, 0>,
            AVX512FMA3Base, EVEX_B;
  }

  // Additional pattern for folding broadcast nodes in other orders.
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode _.RC:$src1, _.RC:$src2,
                    (X86VBroadcast (_.ScalarLdFrag addr:$src3))),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#Suff#_.ZSuffix#mbk) _.RC:$src1,
             _.KRCWM:$mask, _.RC:$src2, addr:$src3)>;
}

multiclass avx512_fma3_213_round<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                 X86VectorVTInfo _, string Suff> {
  let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in
  defm rb: AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.RC:$src3, AVX512RC:$rc),
          OpcodeStr, "$rc, $src3, $src2", "$src2, $src3, $rc",
          (_.VT ( OpNode _.RC:$src2, _.RC:$src1, _.RC:$src3, (i32 imm:$rc))), 1, 1>,
          AVX512FMA3Base, EVEX_B, EVEX_RC;
}

multiclass avx512_fma3p_213_common<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                   SDNode OpNodeRnd, AVX512VLVectorVTInfo _,
                                   string Suff> {
  let Predicates = [HasAVX512] in {
    defm Z      : avx512_fma3p_213_rm<opc, OpcodeStr, OpNode, _.info512, Suff>,
                  avx512_fma3_213_round<opc, OpcodeStr, OpNodeRnd, _.info512,
                      Suff>, EVEX_V512, EVEX_CD8<_.info512.EltSize, CD8VF>;
  }
  let Predicates = [HasVLX, HasAVX512] in {
    defm Z256 : avx512_fma3p_213_rm<opc, OpcodeStr, OpNode, _.info256, Suff>,
                      EVEX_V256, EVEX_CD8<_.info256.EltSize, CD8VF>;
    defm Z128 : avx512_fma3p_213_rm<opc, OpcodeStr, OpNode, _.info128, Suff>,
                      EVEX_V128, EVEX_CD8<_.info128.EltSize, CD8VF>;
  }
}

multiclass avx512_fma3p_213_f<bits<8> opc, string OpcodeStr, SDNode OpNode,
                              SDNode OpNodeRnd > {
    defm PS : avx512_fma3p_213_common<opc, OpcodeStr#"ps", OpNode, OpNodeRnd,
                                      avx512vl_f32_info, "PS">;
    defm PD : avx512_fma3p_213_common<opc, OpcodeStr#"pd", OpNode, OpNodeRnd,
                                      avx512vl_f64_info, "PD">, VEX_W;
}

defm VFMADD213    : avx512_fma3p_213_f<0xA8, "vfmadd213", X86Fmadd, X86FmaddRnd>;
defm VFMSUB213    : avx512_fma3p_213_f<0xAA, "vfmsub213", X86Fmsub, X86FmsubRnd>;
defm VFMADDSUB213 : avx512_fma3p_213_f<0xA6, "vfmaddsub213", X86Fmaddsub, X86FmaddsubRnd>;
defm VFMSUBADD213 : avx512_fma3p_213_f<0xA7, "vfmsubadd213", X86Fmsubadd, X86FmsubaddRnd>;
defm VFNMADD213   : avx512_fma3p_213_f<0xAC, "vfnmadd213", X86Fnmadd, X86FnmaddRnd>;
defm VFNMSUB213   : avx512_fma3p_213_f<0xAE, "vfnmsub213", X86Fnmsub, X86FnmsubRnd>;


multiclass avx512_fma3p_231_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                               X86VectorVTInfo _, string Suff> {
  let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in {
  defm r: AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.RC:$src3),
          OpcodeStr, "$src3, $src2", "$src2, $src3",
          (_.VT (OpNode _.RC:$src2, _.RC:$src3, _.RC:$src1)), 1, 1>,
         AVX512FMA3Base;

  defm m: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.MemOp:$src3),
          OpcodeStr, "$src3, $src2", "$src2, $src3",
          (_.VT (OpNode _.RC:$src2, (_.LdFrag addr:$src3), _.RC:$src1)), 1, 0>,
         AVX512FMA3Base;

  defm mb: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
         (ins _.RC:$src2, _.ScalarMemOp:$src3),
         OpcodeStr, "${src3}"##_.BroadcastStr##", $src2",
         "$src2, ${src3}"##_.BroadcastStr,
         (_.VT (OpNode _.RC:$src2,
                      (_.VT (X86VBroadcast(_.ScalarLdFrag addr:$src3))),
                      _.RC:$src1)), 1, 0>, AVX512FMA3Base, EVEX_B;
  }

  // Additional patterns for folding broadcast nodes in other orders.
  def : Pat<(_.VT (OpNode (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                   _.RC:$src2, _.RC:$src1)),
            (!cast<Instruction>(NAME#Suff#_.ZSuffix#mb) _.RC:$src1,
             _.RC:$src2, addr:$src3)>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                    _.RC:$src2, _.RC:$src1),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#Suff#_.ZSuffix#mbk) _.RC:$src1,
             _.KRCWM:$mask, _.RC:$src2, addr:$src3)>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                    _.RC:$src2, _.RC:$src1),
                   _.ImmAllZerosV)),
            (!cast<Instruction>(NAME#Suff#_.ZSuffix#mbkz) _.RC:$src1,
             _.KRCWM:$mask, _.RC:$src2, addr:$src3)>;
}

multiclass avx512_fma3_231_round<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                 X86VectorVTInfo _, string Suff> {
  let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in
  defm rb: AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.RC:$src3, AVX512RC:$rc),
          OpcodeStr, "$rc, $src3, $src2", "$src2, $src3, $rc",
          (_.VT ( OpNode _.RC:$src2, _.RC:$src3, _.RC:$src1, (i32 imm:$rc))), 1, 1>,
          AVX512FMA3Base, EVEX_B, EVEX_RC;
}

multiclass avx512_fma3p_231_common<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                   SDNode OpNodeRnd, AVX512VLVectorVTInfo _,
                                   string Suff> {
  let Predicates = [HasAVX512] in {
    defm Z      : avx512_fma3p_231_rm<opc, OpcodeStr, OpNode, _.info512, Suff>,
                  avx512_fma3_231_round<opc, OpcodeStr, OpNodeRnd, _.info512,
                      Suff>, EVEX_V512, EVEX_CD8<_.info512.EltSize, CD8VF>;
  }
  let Predicates = [HasVLX, HasAVX512] in {
    defm Z256 : avx512_fma3p_231_rm<opc, OpcodeStr, OpNode, _.info256, Suff>,
                      EVEX_V256, EVEX_CD8<_.info256.EltSize, CD8VF>;
    defm Z128 : avx512_fma3p_231_rm<opc, OpcodeStr, OpNode, _.info128, Suff>,
                      EVEX_V128, EVEX_CD8<_.info128.EltSize, CD8VF>;
  }
}

multiclass avx512_fma3p_231_f<bits<8> opc, string OpcodeStr, SDNode OpNode,
                              SDNode OpNodeRnd > {
    defm PS : avx512_fma3p_231_common<opc, OpcodeStr#"ps", OpNode, OpNodeRnd,
                                      avx512vl_f32_info, "PS">;
    defm PD : avx512_fma3p_231_common<opc, OpcodeStr#"pd", OpNode, OpNodeRnd,
                                      avx512vl_f64_info, "PD">, VEX_W;
}

defm VFMADD231    : avx512_fma3p_231_f<0xB8, "vfmadd231", X86Fmadd, X86FmaddRnd>;
defm VFMSUB231    : avx512_fma3p_231_f<0xBA, "vfmsub231", X86Fmsub, X86FmsubRnd>;
defm VFMADDSUB231 : avx512_fma3p_231_f<0xB6, "vfmaddsub231", X86Fmaddsub, X86FmaddsubRnd>;
defm VFMSUBADD231 : avx512_fma3p_231_f<0xB7, "vfmsubadd231", X86Fmsubadd, X86FmsubaddRnd>;
defm VFNMADD231   : avx512_fma3p_231_f<0xBC, "vfnmadd231", X86Fnmadd, X86FnmaddRnd>;
defm VFNMSUB231   : avx512_fma3p_231_f<0xBE, "vfnmsub231", X86Fnmsub, X86FnmsubRnd>;

multiclass avx512_fma3p_132_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                               X86VectorVTInfo _, string Suff> {
  let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in {
  defm r: AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.RC:$src3),
          OpcodeStr, "$src3, $src2", "$src2, $src3",
          (_.VT (OpNode _.RC:$src1, _.RC:$src3, _.RC:$src2)), 1, 1>,
         AVX512FMA3Base;

  defm m: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.MemOp:$src3),
          OpcodeStr, "$src3, $src2", "$src2, $src3",
          (_.VT (OpNode _.RC:$src1, (_.LdFrag addr:$src3), _.RC:$src2)), 1, 0>,
         AVX512FMA3Base;

  defm mb: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
         (ins _.RC:$src2, _.ScalarMemOp:$src3),
         OpcodeStr, "${src3}"##_.BroadcastStr##", $src2",
         "$src2, ${src3}"##_.BroadcastStr,
         (_.VT (OpNode _.RC:$src1,
                      (_.VT (X86VBroadcast(_.ScalarLdFrag addr:$src3))),
                      _.RC:$src2)), 1, 0>, AVX512FMA3Base, EVEX_B;
  }

  // Additional patterns for folding broadcast nodes in other orders.
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                    _.RC:$src1, _.RC:$src2),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#Suff#_.ZSuffix#mbk) _.RC:$src1,
             _.KRCWM:$mask, _.RC:$src2, addr:$src3)>;
}

multiclass avx512_fma3_132_round<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                 X86VectorVTInfo _, string Suff> {
  let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in
  defm rb: AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.RC:$src3, AVX512RC:$rc),
          OpcodeStr, "$rc, $src3, $src2", "$src2, $src3, $rc",
          (_.VT ( OpNode _.RC:$src1, _.RC:$src3, _.RC:$src2, (i32 imm:$rc))), 1, 1>,
          AVX512FMA3Base, EVEX_B, EVEX_RC;
}

multiclass avx512_fma3p_132_common<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                   SDNode OpNodeRnd, AVX512VLVectorVTInfo _,
                                   string Suff> {
  let Predicates = [HasAVX512] in {
    defm Z      : avx512_fma3p_132_rm<opc, OpcodeStr, OpNode, _.info512, Suff>,
                  avx512_fma3_132_round<opc, OpcodeStr, OpNodeRnd, _.info512,
                      Suff>, EVEX_V512, EVEX_CD8<_.info512.EltSize, CD8VF>;
  }
  let Predicates = [HasVLX, HasAVX512] in {
    defm Z256 : avx512_fma3p_132_rm<opc, OpcodeStr, OpNode, _.info256, Suff>,
                      EVEX_V256, EVEX_CD8<_.info256.EltSize, CD8VF>;
    defm Z128 : avx512_fma3p_132_rm<opc, OpcodeStr, OpNode, _.info128, Suff>,
                      EVEX_V128, EVEX_CD8<_.info128.EltSize, CD8VF>;
  }
}

multiclass avx512_fma3p_132_f<bits<8> opc, string OpcodeStr, SDNode OpNode,
                              SDNode OpNodeRnd > {
    defm PS : avx512_fma3p_132_common<opc, OpcodeStr#"ps", OpNode, OpNodeRnd,
                                      avx512vl_f32_info, "PS">;
    defm PD : avx512_fma3p_132_common<opc, OpcodeStr#"pd", OpNode, OpNodeRnd,
                                      avx512vl_f64_info, "PD">, VEX_W;
}

defm VFMADD132    : avx512_fma3p_132_f<0x98, "vfmadd132", X86Fmadd, X86FmaddRnd>;
defm VFMSUB132    : avx512_fma3p_132_f<0x9A, "vfmsub132", X86Fmsub, X86FmsubRnd>;
defm VFMADDSUB132 : avx512_fma3p_132_f<0x96, "vfmaddsub132", X86Fmaddsub, X86FmaddsubRnd>;
defm VFMSUBADD132 : avx512_fma3p_132_f<0x97, "vfmsubadd132", X86Fmsubadd, X86FmsubaddRnd>;
defm VFNMADD132   : avx512_fma3p_132_f<0x9C, "vfnmadd132", X86Fnmadd, X86FnmaddRnd>;
defm VFNMSUB132   : avx512_fma3p_132_f<0x9E, "vfnmsub132", X86Fnmsub, X86FnmsubRnd>;

// Scalar FMA
let Constraints = "$src1 = $dst" in {
multiclass avx512_fma3s_common<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                               dag RHS_VEC_r, dag RHS_VEC_m, dag RHS_VEC_rb,
                                                        dag RHS_r, dag RHS_m > {
  defm r_Int: AVX512_maskable_3src_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.RC:$src3), OpcodeStr,
          "$src3, $src2", "$src2, $src3", RHS_VEC_r, 1, 1>, AVX512FMA3Base;

  defm m_Int: AVX512_maskable_3src_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.IntScalarMemOp:$src3), OpcodeStr,
          "$src3, $src2", "$src2, $src3", RHS_VEC_m, 1, 1>, AVX512FMA3Base;

  defm rb_Int: AVX512_maskable_3src_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
         (ins _.RC:$src2, _.RC:$src3, AVX512RC:$rc),
         OpcodeStr, "$rc, $src3, $src2", "$src2, $src3, $rc", RHS_VEC_rb, 1, 1>,
                                       AVX512FMA3Base, EVEX_B, EVEX_RC;

  let isCodeGenOnly = 1, isCommutable = 1 in {
    def r     : AVX512FMA3<opc, MRMSrcReg, (outs _.FRC:$dst),
                     (ins _.FRC:$src1, _.FRC:$src2, _.FRC:$src3),
                     !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                     [RHS_r]>;
    def m     : AVX512FMA3<opc, MRMSrcMem, (outs _.FRC:$dst),
                    (ins _.FRC:$src1, _.FRC:$src2, _.ScalarMemOp:$src3),
                    !strconcat(OpcodeStr,
                               "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                    [RHS_m]>;
  }// isCodeGenOnly = 1
}
}// Constraints = "$src1 = $dst"

multiclass avx512_fma3s_all<bits<8> opc213, bits<8> opc231, bits<8> opc132,
                            string OpcodeStr, SDNode OpNode, SDNode OpNodeRnds1,
                            SDNode OpNodeRnds3, X86VectorVTInfo _ , string SUFF> {
  let ExeDomain = _.ExeDomain in {
  defm NAME#213#SUFF#Z: avx512_fma3s_common<opc213, OpcodeStr#"213"#_.Suffix , _ ,
                // Operands for intrinsic are in 123 order to preserve passthu
                // semantics.
                (_.VT (OpNodeRnds1 _.RC:$src1, _.RC:$src2, _.RC:$src3, (i32 FROUND_CURRENT))),
                (_.VT (OpNodeRnds1 _.RC:$src1, _.RC:$src2,
                         _.ScalarIntMemCPat:$src3, (i32 FROUND_CURRENT))),
                (_.VT (OpNodeRnds1 _.RC:$src1, _.RC:$src2, _.RC:$src3,
                         (i32 imm:$rc))),
                (set _.FRC:$dst, (_.EltVT (OpNode _.FRC:$src2, _.FRC:$src1,
                         _.FRC:$src3))),
                (set _.FRC:$dst, (_.EltVT (OpNode _.FRC:$src2, _.FRC:$src1,
                         (_.ScalarLdFrag addr:$src3))))>;

  defm NAME#231#SUFF#Z: avx512_fma3s_common<opc231, OpcodeStr#"231"#_.Suffix , _ ,
                (_.VT (OpNodeRnds3 _.RC:$src2, _.RC:$src3, _.RC:$src1, (i32 FROUND_CURRENT))),
                (_.VT (OpNodeRnds3 _.RC:$src2, _.ScalarIntMemCPat:$src3,
                              _.RC:$src1, (i32 FROUND_CURRENT))),
                (_.VT ( OpNodeRnds3 _.RC:$src2, _.RC:$src3, _.RC:$src1,
                                  (i32 imm:$rc))),
                (set _.FRC:$dst, (_.EltVT (OpNode _.FRC:$src2, _.FRC:$src3,
                                          _.FRC:$src1))),
                (set _.FRC:$dst, (_.EltVT (OpNode _.FRC:$src2,
                            (_.ScalarLdFrag addr:$src3), _.FRC:$src1)))>;

  defm NAME#132#SUFF#Z: avx512_fma3s_common<opc132, OpcodeStr#"132"#_.Suffix , _ ,
                (_.VT (OpNodeRnds1 _.RC:$src1, _.RC:$src3, _.RC:$src2, (i32 FROUND_CURRENT))),
                (_.VT (OpNodeRnds1 _.RC:$src1, _.ScalarIntMemCPat:$src3,
                              _.RC:$src2, (i32 FROUND_CURRENT))),
                (_.VT (OpNodeRnds1 _.RC:$src1, _.RC:$src3, _.RC:$src2,
                         (i32 imm:$rc))),
                (set _.FRC:$dst, (_.EltVT (OpNode _.FRC:$src1, _.FRC:$src3,
                         _.FRC:$src2))),
                (set _.FRC:$dst, (_.EltVT (OpNode _.FRC:$src1,
                          (_.ScalarLdFrag addr:$src3), _.FRC:$src2)))>;
  }
}

multiclass avx512_fma3s<bits<8> opc213, bits<8> opc231, bits<8> opc132,
                        string OpcodeStr, SDNode OpNode, SDNode OpNodeRnds1,
                        SDNode OpNodeRnds3> {
  let Predicates = [HasAVX512] in {
    defm NAME : avx512_fma3s_all<opc213, opc231, opc132, OpcodeStr, OpNode,
                                 OpNodeRnds1, OpNodeRnds3, f32x_info, "SS">,
                                 EVEX_CD8<32, CD8VT1>, VEX_LIG;
    defm NAME : avx512_fma3s_all<opc213, opc231, opc132, OpcodeStr, OpNode,
                                 OpNodeRnds1, OpNodeRnds3, f64x_info, "SD">,
                                 EVEX_CD8<64, CD8VT1>, VEX_LIG, VEX_W;
  }
}

defm VFMADD  : avx512_fma3s<0xA9, 0xB9, 0x99, "vfmadd", X86Fmadd, X86FmaddRnds1,
                            X86FmaddRnds3>;
defm VFMSUB  : avx512_fma3s<0xAB, 0xBB, 0x9B, "vfmsub", X86Fmsub, X86FmsubRnds1,
                            X86FmsubRnds3>;
defm VFNMADD : avx512_fma3s<0xAD, 0xBD, 0x9D, "vfnmadd", X86Fnmadd,
                            X86FnmaddRnds1, X86FnmaddRnds3>;
defm VFNMSUB : avx512_fma3s<0xAF, 0xBF, 0x9F, "vfnmsub", X86Fnmsub,
                            X86FnmsubRnds1, X86FnmsubRnds3>;

//===----------------------------------------------------------------------===//
// AVX-512  Packed Multiply of Unsigned 52-bit Integers and Add the Low 52-bit IFMA
//===----------------------------------------------------------------------===//
let Constraints = "$src1 = $dst" in {
multiclass avx512_pmadd52_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                                            X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
  defm r: AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.RC:$src3),
          OpcodeStr, "$src3, $src2", "$src2, $src3",
          (_.VT (OpNode _.RC:$src1, _.RC:$src2, _.RC:$src3))>,
         AVX512FMA3Base;

  defm m: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
          (ins _.RC:$src2, _.MemOp:$src3),
          OpcodeStr, "$src3, $src2", "$src2, $src3",
          (_.VT (OpNode _.RC:$src1, _.RC:$src2, (_.LdFrag addr:$src3)))>,
          AVX512FMA3Base;

  defm mb: AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
            (ins _.RC:$src2, _.ScalarMemOp:$src3),
            OpcodeStr,   !strconcat("${src3}", _.BroadcastStr,", $src2"),
            !strconcat("$src2, ${src3}", _.BroadcastStr ),
            (OpNode _.RC:$src1,
             _.RC:$src2,(_.VT (X86VBroadcast (_.ScalarLdFrag addr:$src3))))>,
            AVX512FMA3Base, EVEX_B;
  }
}
} // Constraints = "$src1 = $dst"

multiclass avx512_pmadd52_common<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                     AVX512VLVectorVTInfo _> {
  let Predicates = [HasIFMA] in {
    defm Z      : avx512_pmadd52_rm<opc, OpcodeStr, OpNode, _.info512>,
                      EVEX_V512, EVEX_CD8<_.info512.EltSize, CD8VF>;
  }
  let Predicates = [HasVLX, HasIFMA] in {
    defm Z256 : avx512_pmadd52_rm<opc, OpcodeStr, OpNode, _.info256>,
                      EVEX_V256, EVEX_CD8<_.info256.EltSize, CD8VF>;
    defm Z128 : avx512_pmadd52_rm<opc, OpcodeStr, OpNode, _.info128>,
                      EVEX_V128, EVEX_CD8<_.info128.EltSize, CD8VF>;
  }
}

defm VPMADD52LUQ : avx512_pmadd52_common<0xb4, "vpmadd52luq", x86vpmadd52l,
                                  avx512vl_i64_info>, VEX_W;
defm VPMADD52HUQ : avx512_pmadd52_common<0xb5, "vpmadd52huq", x86vpmadd52h,
                                  avx512vl_i64_info>, VEX_W;

//===----------------------------------------------------------------------===//
// AVX-512  Scalar convert from sign integer to float/double
//===----------------------------------------------------------------------===//

multiclass avx512_vcvtsi<bits<8> opc, SDNode OpNode, RegisterClass SrcRC,
                    X86VectorVTInfo DstVT, X86MemOperand x86memop,
                    PatFrag ld_frag, string asm> {
  let hasSideEffects = 0 in {
    def rr : SI<opc, MRMSrcReg, (outs DstVT.FRC:$dst),
              (ins DstVT.FRC:$src1, SrcRC:$src),
              !strconcat(asm,"\t{$src, $src1, $dst|$dst, $src1, $src}"), []>,
              EVEX_4V;
    let mayLoad = 1 in
      def rm : SI<opc, MRMSrcMem, (outs DstVT.FRC:$dst),
              (ins DstVT.FRC:$src1, x86memop:$src),
              !strconcat(asm,"\t{$src, $src1, $dst|$dst, $src1, $src}"), []>,
              EVEX_4V;
  } // hasSideEffects = 0
  let isCodeGenOnly = 1 in {
    def rr_Int : SI<opc, MRMSrcReg, (outs DstVT.RC:$dst),
                  (ins DstVT.RC:$src1, SrcRC:$src2),
                  !strconcat(asm,"\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                  [(set DstVT.RC:$dst,
                        (OpNode (DstVT.VT DstVT.RC:$src1),
                                 SrcRC:$src2,
                                 (i32 FROUND_CURRENT)))]>, EVEX_4V;

    def rm_Int : SI<opc, MRMSrcMem, (outs DstVT.RC:$dst),
                  (ins DstVT.RC:$src1, x86memop:$src2),
                  !strconcat(asm,"\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                  [(set DstVT.RC:$dst,
                        (OpNode (DstVT.VT DstVT.RC:$src1),
                                 (ld_frag addr:$src2),
                                 (i32 FROUND_CURRENT)))]>, EVEX_4V;
  }//isCodeGenOnly = 1
}

multiclass avx512_vcvtsi_round<bits<8> opc, SDNode OpNode, RegisterClass SrcRC,
                    X86VectorVTInfo DstVT, string asm> {
  def rrb_Int : SI<opc, MRMSrcReg, (outs DstVT.RC:$dst),
              (ins DstVT.RC:$src1, SrcRC:$src2, AVX512RC:$rc),
              !strconcat(asm,
                  "\t{$src2, $rc, $src1, $dst|$dst, $src1, $rc, $src2}"),
              [(set DstVT.RC:$dst,
                    (OpNode (DstVT.VT DstVT.RC:$src1),
                             SrcRC:$src2,
                             (i32 imm:$rc)))]>, EVEX_4V, EVEX_B, EVEX_RC;
}

multiclass avx512_vcvtsi_common<bits<8> opc, SDNode OpNode, RegisterClass SrcRC,
                    X86VectorVTInfo DstVT, X86MemOperand x86memop,
                    PatFrag ld_frag, string asm> {
  defm NAME : avx512_vcvtsi_round<opc, OpNode, SrcRC, DstVT, asm>,
              avx512_vcvtsi<opc, OpNode, SrcRC, DstVT, x86memop, ld_frag, asm>,
                        VEX_LIG;
}

let Predicates = [HasAVX512] in {
defm VCVTSI2SSZ  : avx512_vcvtsi_common<0x2A, X86SintToFpRnd, GR32,
                                 v4f32x_info, i32mem, loadi32, "cvtsi2ss{l}">,
                                 XS, EVEX_CD8<32, CD8VT1>;
defm VCVTSI642SSZ: avx512_vcvtsi_common<0x2A, X86SintToFpRnd, GR64,
                                 v4f32x_info, i64mem, loadi64, "cvtsi2ss{q}">,
                                 XS, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VCVTSI2SDZ  : avx512_vcvtsi_common<0x2A, X86SintToFpRnd, GR32,
                                 v2f64x_info, i32mem, loadi32, "cvtsi2sd{l}">,
                                 XD, EVEX_CD8<32, CD8VT1>;
defm VCVTSI642SDZ: avx512_vcvtsi_common<0x2A, X86SintToFpRnd, GR64,
                                 v2f64x_info, i64mem, loadi64, "cvtsi2sd{q}">,
                                 XD, VEX_W, EVEX_CD8<64, CD8VT1>;

def : InstAlias<"vcvtsi2ss\t{$src, $src1, $dst|$dst, $src1, $src}",
              (VCVTSI2SSZrm FR64X:$dst, FR64X:$src1, i32mem:$src), 0>;
def : InstAlias<"vcvtsi2sd\t{$src, $src1, $dst|$dst, $src1, $src}",
              (VCVTSI2SDZrm FR64X:$dst, FR64X:$src1, i32mem:$src), 0>;

def : Pat<(f32 (sint_to_fp (loadi32 addr:$src))),
          (VCVTSI2SSZrm (f32 (IMPLICIT_DEF)), addr:$src)>;
def : Pat<(f32 (sint_to_fp (loadi64 addr:$src))),
          (VCVTSI642SSZrm (f32 (IMPLICIT_DEF)), addr:$src)>;
def : Pat<(f64 (sint_to_fp (loadi32 addr:$src))),
          (VCVTSI2SDZrm (f64 (IMPLICIT_DEF)), addr:$src)>;
def : Pat<(f64 (sint_to_fp (loadi64 addr:$src))),
          (VCVTSI642SDZrm (f64 (IMPLICIT_DEF)), addr:$src)>;

def : Pat<(f32 (sint_to_fp GR32:$src)),
          (VCVTSI2SSZrr (f32 (IMPLICIT_DEF)), GR32:$src)>;
def : Pat<(f32 (sint_to_fp GR64:$src)),
          (VCVTSI642SSZrr (f32 (IMPLICIT_DEF)), GR64:$src)>;
def : Pat<(f64 (sint_to_fp GR32:$src)),
          (VCVTSI2SDZrr (f64 (IMPLICIT_DEF)), GR32:$src)>;
def : Pat<(f64 (sint_to_fp GR64:$src)),
          (VCVTSI642SDZrr (f64 (IMPLICIT_DEF)), GR64:$src)>;

defm VCVTUSI2SSZ   : avx512_vcvtsi_common<0x7B, X86UintToFpRnd, GR32,
                                  v4f32x_info, i32mem, loadi32,
                                  "cvtusi2ss{l}">, XS, EVEX_CD8<32, CD8VT1>;
defm VCVTUSI642SSZ : avx512_vcvtsi_common<0x7B, X86UintToFpRnd, GR64,
                                  v4f32x_info, i64mem, loadi64, "cvtusi2ss{q}">,
                                  XS, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VCVTUSI2SDZ   : avx512_vcvtsi<0x7B, X86UintToFpRnd, GR32, v2f64x_info,
                                  i32mem, loadi32, "cvtusi2sd{l}">,
                                  XD, VEX_LIG, EVEX_CD8<32, CD8VT1>;
defm VCVTUSI642SDZ : avx512_vcvtsi_common<0x7B, X86UintToFpRnd, GR64,
                                  v2f64x_info, i64mem, loadi64, "cvtusi2sd{q}">,
                                  XD, VEX_W, EVEX_CD8<64, CD8VT1>;

def : InstAlias<"vcvtusi2ss\t{$src, $src1, $dst|$dst, $src1, $src}",
              (VCVTUSI2SSZrm FR64X:$dst, FR64X:$src1, i32mem:$src), 0>;
def : InstAlias<"vcvtusi2sd\t{$src, $src1, $dst|$dst, $src1, $src}",
              (VCVTUSI2SDZrm FR64X:$dst, FR64X:$src1, i32mem:$src), 0>;

def : Pat<(f32 (uint_to_fp (loadi32 addr:$src))),
          (VCVTUSI2SSZrm (f32 (IMPLICIT_DEF)), addr:$src)>;
def : Pat<(f32 (uint_to_fp (loadi64 addr:$src))),
          (VCVTUSI642SSZrm (f32 (IMPLICIT_DEF)), addr:$src)>;
def : Pat<(f64 (uint_to_fp (loadi32 addr:$src))),
          (VCVTUSI2SDZrm (f64 (IMPLICIT_DEF)), addr:$src)>;
def : Pat<(f64 (uint_to_fp (loadi64 addr:$src))),
          (VCVTUSI642SDZrm (f64 (IMPLICIT_DEF)), addr:$src)>;

def : Pat<(f32 (uint_to_fp GR32:$src)),
          (VCVTUSI2SSZrr (f32 (IMPLICIT_DEF)), GR32:$src)>;
def : Pat<(f32 (uint_to_fp GR64:$src)),
          (VCVTUSI642SSZrr (f32 (IMPLICIT_DEF)), GR64:$src)>;
def : Pat<(f64 (uint_to_fp GR32:$src)),
          (VCVTUSI2SDZrr (f64 (IMPLICIT_DEF)), GR32:$src)>;
def : Pat<(f64 (uint_to_fp GR64:$src)),
          (VCVTUSI642SDZrr (f64 (IMPLICIT_DEF)), GR64:$src)>;
}

//===----------------------------------------------------------------------===//
// AVX-512  Scalar convert from float/double to integer
//===----------------------------------------------------------------------===//
multiclass avx512_cvt_s_int_round<bits<8> opc, X86VectorVTInfo SrcVT ,
                                  X86VectorVTInfo DstVT, SDNode OpNode, string asm> {
  let Predicates = [HasAVX512] in {
    def rr : SI<opc, MRMSrcReg, (outs DstVT.RC:$dst), (ins SrcVT.RC:$src),
                !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
                [(set DstVT.RC:$dst, (OpNode (SrcVT.VT SrcVT.RC:$src),(i32 FROUND_CURRENT)))]>,
                EVEX, VEX_LIG;
    def rb : SI<opc, MRMSrcReg, (outs DstVT.RC:$dst), (ins SrcVT.RC:$src, AVX512RC:$rc),
                !strconcat(asm,"\t{$rc, $src, $dst|$dst, $src, $rc}"),
                [(set DstVT.RC:$dst, (OpNode (SrcVT.VT SrcVT.RC:$src),(i32 imm:$rc)))]>,
                EVEX, VEX_LIG, EVEX_B, EVEX_RC;
    def rm : SI<opc, MRMSrcMem, (outs DstVT.RC:$dst), (ins SrcVT.IntScalarMemOp:$src),
                !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
                [(set DstVT.RC:$dst, (OpNode
                      (SrcVT.VT SrcVT.ScalarIntMemCPat:$src),
                      (i32 FROUND_CURRENT)))]>,
                EVEX, VEX_LIG;
  } // Predicates = [HasAVX512]
}

// Convert float/double to signed/unsigned int 32/64
defm VCVTSS2SIZ: avx512_cvt_s_int_round<0x2D, f32x_info, i32x_info,
                                   X86cvts2si, "cvtss2si">,
                                   XS, EVEX_CD8<32, CD8VT1>;
defm VCVTSS2SI64Z: avx512_cvt_s_int_round<0x2D, f32x_info, i64x_info,
                                   X86cvts2si, "cvtss2si">,
                                   XS, VEX_W, EVEX_CD8<32, CD8VT1>;
defm VCVTSS2USIZ: avx512_cvt_s_int_round<0x79, f32x_info, i32x_info,
                                   X86cvts2usi, "cvtss2usi">,
                                   XS, EVEX_CD8<32, CD8VT1>;
defm VCVTSS2USI64Z: avx512_cvt_s_int_round<0x79, f32x_info, i64x_info,
                                   X86cvts2usi, "cvtss2usi">, XS, VEX_W,
                                   EVEX_CD8<32, CD8VT1>;
defm VCVTSD2SIZ: avx512_cvt_s_int_round<0x2D, f64x_info, i32x_info,
                                   X86cvts2si, "cvtsd2si">,
                                   XD, EVEX_CD8<64, CD8VT1>;
defm VCVTSD2SI64Z: avx512_cvt_s_int_round<0x2D, f64x_info, i64x_info,
                                   X86cvts2si, "cvtsd2si">,
                                   XD, VEX_W, EVEX_CD8<64, CD8VT1>;
defm VCVTSD2USIZ:   avx512_cvt_s_int_round<0x79, f64x_info, i32x_info,
                                   X86cvts2usi, "cvtsd2usi">,
                                   XD, EVEX_CD8<64, CD8VT1>;
defm VCVTSD2USI64Z: avx512_cvt_s_int_round<0x79, f64x_info, i64x_info,
                                   X86cvts2usi, "cvtsd2usi">, XD, VEX_W,
                                   EVEX_CD8<64, CD8VT1>;

// The SSE version of these instructions are disabled for AVX512.
// Therefore, the SSE intrinsics are mapped to the AVX512 instructions.
let Predicates = [HasAVX512] in {
  def : Pat<(i32 (int_x86_sse_cvtss2si (v4f32 VR128X:$src))),
            (VCVTSS2SIZrr VR128X:$src)>;
  def : Pat<(i32 (int_x86_sse_cvtss2si sse_load_f32:$src)),
            (VCVTSS2SIZrm sse_load_f32:$src)>;
  def : Pat<(i64 (int_x86_sse_cvtss2si64 (v4f32 VR128X:$src))),
            (VCVTSS2SI64Zrr VR128X:$src)>;
  def : Pat<(i64 (int_x86_sse_cvtss2si64 sse_load_f32:$src)),
            (VCVTSS2SI64Zrm sse_load_f32:$src)>;
  def : Pat<(i32 (int_x86_sse2_cvtsd2si (v2f64 VR128X:$src))),
            (VCVTSD2SIZrr VR128X:$src)>;
  def : Pat<(i32 (int_x86_sse2_cvtsd2si sse_load_f64:$src)),
            (VCVTSD2SIZrm sse_load_f64:$src)>;
  def : Pat<(i64 (int_x86_sse2_cvtsd2si64 (v2f64 VR128X:$src))),
            (VCVTSD2SI64Zrr VR128X:$src)>;
  def : Pat<(i64 (int_x86_sse2_cvtsd2si64 sse_load_f64:$src)),
            (VCVTSD2SI64Zrm sse_load_f64:$src)>;
} // HasAVX512

let Predicates = [HasAVX512] in {
  def : Pat<(int_x86_sse_cvtsi2ss VR128X:$src1, GR32:$src2),
            (VCVTSI2SSZrr_Int VR128X:$src1, GR32:$src2)>;
  def : Pat<(int_x86_sse_cvtsi2ss VR128X:$src1, (loadi32 addr:$src2)),
            (VCVTSI2SSZrm_Int VR128X:$src1, addr:$src2)>;
  def : Pat<(int_x86_sse_cvtsi642ss VR128X:$src1, GR64:$src2),
            (VCVTSI642SSZrr_Int VR128X:$src1, GR64:$src2)>;
  def : Pat<(int_x86_sse_cvtsi642ss VR128X:$src1, (loadi64 addr:$src2)),
            (VCVTSI642SSZrm_Int VR128X:$src1, addr:$src2)>;
  def : Pat<(int_x86_sse2_cvtsi2sd VR128X:$src1, GR32:$src2),
            (VCVTSI2SDZrr_Int VR128X:$src1, GR32:$src2)>;
  def : Pat<(int_x86_sse2_cvtsi2sd VR128X:$src1, (loadi32 addr:$src2)),
            (VCVTSI2SDZrm_Int VR128X:$src1, addr:$src2)>;
  def : Pat<(int_x86_sse2_cvtsi642sd VR128X:$src1, GR64:$src2),
            (VCVTSI642SDZrr_Int VR128X:$src1, GR64:$src2)>;
  def : Pat<(int_x86_sse2_cvtsi642sd VR128X:$src1, (loadi64 addr:$src2)),
            (VCVTSI642SDZrm_Int VR128X:$src1, addr:$src2)>;
  def : Pat<(int_x86_avx512_cvtusi2sd VR128X:$src1, GR32:$src2),
            (VCVTUSI2SDZrr_Int VR128X:$src1, GR32:$src2)>;
  def : Pat<(int_x86_avx512_cvtusi2sd VR128X:$src1, (loadi32 addr:$src2)),
            (VCVTUSI2SDZrm_Int VR128X:$src1, addr:$src2)>;
} // Predicates = [HasAVX512]

// Patterns used for matching vcvtsi2s{s,d} intrinsic sequences from clang
// which produce unnecessary vmovs{s,d} instructions
let Predicates = [HasAVX512] in {
def : Pat<(v4f32 (X86Movss
                   (v4f32 VR128X:$dst),
                   (v4f32 (scalar_to_vector (f32 (sint_to_fp GR64:$src)))))),
          (VCVTSI642SSZrr_Int VR128X:$dst, GR64:$src)>;

def : Pat<(v4f32 (X86Movss
                   (v4f32 VR128X:$dst),
                   (v4f32 (scalar_to_vector (f32 (sint_to_fp GR32:$src)))))),
          (VCVTSI2SSZrr_Int VR128X:$dst, GR32:$src)>;

def : Pat<(v2f64 (X86Movsd
                   (v2f64 VR128X:$dst),
                   (v2f64 (scalar_to_vector (f64 (sint_to_fp GR64:$src)))))),
          (VCVTSI642SDZrr_Int VR128X:$dst, GR64:$src)>;

def : Pat<(v2f64 (X86Movsd
                   (v2f64 VR128X:$dst),
                   (v2f64 (scalar_to_vector (f64 (sint_to_fp GR32:$src)))))),
          (VCVTSI2SDZrr_Int VR128X:$dst, GR32:$src)>;
} // Predicates = [HasAVX512]

// Convert float/double to signed/unsigned int 32/64 with truncation
multiclass avx512_cvt_s_all<bits<8> opc, string asm, X86VectorVTInfo _SrcRC,
                            X86VectorVTInfo _DstRC, SDNode OpNode,
                            SDNode OpNodeRnd, string aliasStr>{
let Predicates = [HasAVX512] in {
  def rr : AVX512<opc, MRMSrcReg, (outs _DstRC.RC:$dst), (ins _SrcRC.FRC:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
              [(set _DstRC.RC:$dst, (OpNode _SrcRC.FRC:$src))]>, EVEX;
  let hasSideEffects = 0 in
  def rb : AVX512<opc, MRMSrcReg, (outs _DstRC.RC:$dst), (ins _SrcRC.FRC:$src),
                !strconcat(asm,"\t{{sae}, $src, $dst|$dst, $src, {sae}}"),
                []>, EVEX, EVEX_B;
  def rm : AVX512<opc, MRMSrcMem, (outs _DstRC.RC:$dst), (ins _SrcRC.ScalarMemOp:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
              [(set _DstRC.RC:$dst, (OpNode (_SrcRC.ScalarLdFrag addr:$src)))]>,
              EVEX;

  def : InstAlias<asm # aliasStr # "\t{$src, $dst|$dst, $src}",
          (!cast<Instruction>(NAME # "rr") _DstRC.RC:$dst, _SrcRC.FRC:$src), 0>;
  def : InstAlias<asm # aliasStr # "\t\t{{sae}, $src, $dst|$dst, $src, {sae}}",
          (!cast<Instruction>(NAME # "rb") _DstRC.RC:$dst, _SrcRC.FRC:$src), 0>;
  def : InstAlias<asm # aliasStr # "\t{$src, $dst|$dst, $src}",
          (!cast<Instruction>(NAME # "rm") _DstRC.RC:$dst,
                                          _SrcRC.ScalarMemOp:$src), 0>;

  let isCodeGenOnly = 1 in {
    def rr_Int : AVX512<opc, MRMSrcReg, (outs _DstRC.RC:$dst), (ins _SrcRC.RC:$src),
              !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
             [(set _DstRC.RC:$dst, (OpNodeRnd (_SrcRC.VT _SrcRC.RC:$src),
                                   (i32 FROUND_CURRENT)))]>, EVEX, VEX_LIG;
    def rb_Int : AVX512<opc, MRMSrcReg, (outs _DstRC.RC:$dst), (ins _SrcRC.RC:$src),
              !strconcat(asm,"\t{{sae}, $src, $dst|$dst, $src, {sae}}"),
              [(set _DstRC.RC:$dst, (OpNodeRnd (_SrcRC.VT _SrcRC.RC:$src),
                                    (i32 FROUND_NO_EXC)))]>,
                                    EVEX,VEX_LIG , EVEX_B;
    let mayLoad = 1, hasSideEffects = 0 in
      def rm_Int : AVX512<opc, MRMSrcMem, (outs _DstRC.RC:$dst),
                  (ins _SrcRC.IntScalarMemOp:$src),
                  !strconcat(asm,"\t{$src, $dst|$dst, $src}"),
                  []>, EVEX, VEX_LIG;

  } // isCodeGenOnly = 1
} //HasAVX512
}


defm VCVTTSS2SIZ: avx512_cvt_s_all<0x2C, "vcvttss2si", f32x_info, i32x_info,
                        fp_to_sint, X86cvtts2IntRnd, "{l}">,
                        XS, EVEX_CD8<32, CD8VT1>;
defm VCVTTSS2SI64Z: avx512_cvt_s_all<0x2C, "vcvttss2si", f32x_info, i64x_info,
                        fp_to_sint, X86cvtts2IntRnd, "{q}">,
                        VEX_W, XS, EVEX_CD8<32, CD8VT1>;
defm VCVTTSD2SIZ: avx512_cvt_s_all<0x2C, "vcvttsd2si", f64x_info, i32x_info,
                        fp_to_sint, X86cvtts2IntRnd, "{l}">,
                        XD, EVEX_CD8<64, CD8VT1>;
defm VCVTTSD2SI64Z: avx512_cvt_s_all<0x2C, "vcvttsd2si", f64x_info, i64x_info,
                        fp_to_sint, X86cvtts2IntRnd, "{q}">,
                        VEX_W, XD, EVEX_CD8<64, CD8VT1>;

defm VCVTTSS2USIZ: avx512_cvt_s_all<0x78, "vcvttss2usi", f32x_info, i32x_info,
                        fp_to_uint, X86cvtts2UIntRnd, "{l}">,
                        XS, EVEX_CD8<32, CD8VT1>;
defm VCVTTSS2USI64Z: avx512_cvt_s_all<0x78, "vcvttss2usi", f32x_info, i64x_info,
                        fp_to_uint, X86cvtts2UIntRnd, "{q}">,
                        XS,VEX_W, EVEX_CD8<32, CD8VT1>;
defm VCVTTSD2USIZ: avx512_cvt_s_all<0x78, "vcvttsd2usi", f64x_info, i32x_info,
                        fp_to_uint, X86cvtts2UIntRnd, "{l}">,
                        XD, EVEX_CD8<64, CD8VT1>;
defm VCVTTSD2USI64Z: avx512_cvt_s_all<0x78, "vcvttsd2usi", f64x_info, i64x_info,
                        fp_to_uint, X86cvtts2UIntRnd, "{q}">,
                        XD, VEX_W, EVEX_CD8<64, CD8VT1>;
let Predicates = [HasAVX512] in {
  def : Pat<(i32 (int_x86_sse_cvttss2si (v4f32 VR128X:$src))),
            (VCVTTSS2SIZrr_Int VR128X:$src)>;
  def : Pat<(i32 (int_x86_sse_cvttss2si sse_load_f32:$src)),
            (VCVTTSS2SIZrm_Int ssmem:$src)>;
  def : Pat<(i64 (int_x86_sse_cvttss2si64 (v4f32 VR128X:$src))),
            (VCVTTSS2SI64Zrr_Int VR128X:$src)>;
  def : Pat<(i64 (int_x86_sse_cvttss2si64 sse_load_f32:$src)),
            (VCVTTSS2SI64Zrm_Int ssmem:$src)>;
  def : Pat<(i32 (int_x86_sse2_cvttsd2si (v2f64 VR128X:$src))),
            (VCVTTSD2SIZrr_Int VR128X:$src)>;
  def : Pat<(i32 (int_x86_sse2_cvttsd2si sse_load_f64:$src)),
            (VCVTTSD2SIZrm_Int sdmem:$src)>;
  def : Pat<(i64 (int_x86_sse2_cvttsd2si64 (v2f64 VR128X:$src))),
            (VCVTTSD2SI64Zrr_Int VR128X:$src)>;
  def : Pat<(i64 (int_x86_sse2_cvttsd2si64 sse_load_f64:$src)),
            (VCVTTSD2SI64Zrm_Int sdmem:$src)>;
} // HasAVX512
//===----------------------------------------------------------------------===//
// AVX-512  Convert form float to double and back
//===----------------------------------------------------------------------===//
multiclass avx512_cvt_fp_scalar<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                         X86VectorVTInfo _Src, SDNode OpNode> {
  defm rr_Int : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _Src.RC:$src2), OpcodeStr,
                         "$src2, $src1", "$src1, $src2",
                         (_.VT (OpNode (_.VT _.RC:$src1),
                                       (_Src.VT _Src.RC:$src2),
                                       (i32 FROUND_CURRENT)))>,
                         EVEX_4V, VEX_LIG, Sched<[WriteCvtF2F]>;
  defm rm_Int : AVX512_maskable_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _Src.IntScalarMemOp:$src2), OpcodeStr,
                         "$src2, $src1", "$src1, $src2",
                         (_.VT (OpNode (_.VT _.RC:$src1),
                                  (_Src.VT _Src.ScalarIntMemCPat:$src2),
                                  (i32 FROUND_CURRENT)))>,
                         EVEX_4V, VEX_LIG, Sched<[WriteCvtF2FLd, ReadAfterLd]>;

  let isCodeGenOnly = 1, hasSideEffects = 0 in {
    def rr : I<opc, MRMSrcReg, (outs _.FRC:$dst),
               (ins _.FRC:$src1, _Src.FRC:$src2),
               OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>,
               EVEX_4V, VEX_LIG, Sched<[WriteCvtF2F]>;
    let mayLoad = 1 in
    def rm : I<opc, MRMSrcMem, (outs _.FRC:$dst),
               (ins _.FRC:$src1, _Src.ScalarMemOp:$src2),
               OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>,
               EVEX_4V, VEX_LIG, Sched<[WriteCvtF2FLd, ReadAfterLd]>;
  }
}

// Scalar Coversion with SAE - suppress all exceptions
multiclass avx512_cvt_fp_sae_scalar<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                         X86VectorVTInfo _Src, SDNode OpNodeRnd> {
  defm rrb_Int : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                        (ins _.RC:$src1, _Src.RC:$src2), OpcodeStr,
                        "{sae}, $src2, $src1", "$src1, $src2, {sae}",
                        (_.VT (OpNodeRnd (_.VT _.RC:$src1),
                                         (_Src.VT _Src.RC:$src2),
                                         (i32 FROUND_NO_EXC)))>,
                        EVEX_4V, VEX_LIG, EVEX_B;
}

// Scalar Conversion with rounding control (RC)
multiclass avx512_cvt_fp_rc_scalar<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                         X86VectorVTInfo _Src, SDNode OpNodeRnd> {
  defm rrb_Int : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                        (ins _.RC:$src1, _Src.RC:$src2, AVX512RC:$rc), OpcodeStr,
                        "$rc, $src2, $src1", "$src1, $src2, $rc",
                        (_.VT (OpNodeRnd (_.VT _.RC:$src1),
                                         (_Src.VT _Src.RC:$src2), (i32 imm:$rc)))>,
                        EVEX_4V, VEX_LIG, Sched<[WriteCvtF2FLd, ReadAfterLd]>,
                        EVEX_B, EVEX_RC;
}
multiclass avx512_cvt_fp_scalar_sd2ss<bits<8> opc, string OpcodeStr,
                                  SDNode OpNodeRnd, X86VectorVTInfo _src,
                                                        X86VectorVTInfo _dst> {
  let Predicates = [HasAVX512] in {
    defm Z : avx512_cvt_fp_scalar<opc, OpcodeStr, _dst, _src, OpNodeRnd>,
             avx512_cvt_fp_rc_scalar<opc, OpcodeStr, _dst, _src,
                               OpNodeRnd>, VEX_W, EVEX_CD8<64, CD8VT1>, XD;
  }
}

multiclass avx512_cvt_fp_scalar_ss2sd<bits<8> opc, string OpcodeStr,
                                    SDNode OpNodeRnd, X86VectorVTInfo _src,
                                                          X86VectorVTInfo _dst> {
  let Predicates = [HasAVX512] in {
    defm Z : avx512_cvt_fp_scalar<opc, OpcodeStr, _dst, _src, OpNodeRnd>,
             avx512_cvt_fp_sae_scalar<opc, OpcodeStr, _dst, _src, OpNodeRnd>,
             EVEX_CD8<32, CD8VT1>, XS;
  }
}
defm VCVTSD2SS : avx512_cvt_fp_scalar_sd2ss<0x5A, "vcvtsd2ss",
                                         X86froundRnd, f64x_info, f32x_info>;
defm VCVTSS2SD : avx512_cvt_fp_scalar_ss2sd<0x5A, "vcvtss2sd",
                                          X86fpextRnd,f32x_info, f64x_info >;

def : Pat<(f64 (fpextend FR32X:$src)),
          (VCVTSS2SDZrr (COPY_TO_REGCLASS FR32X:$src, FR64X), FR32X:$src)>,
          Requires<[HasAVX512]>;
def : Pat<(f64 (fpextend (loadf32 addr:$src))),
          (VCVTSS2SDZrm (f64 (IMPLICIT_DEF)), addr:$src)>,
          Requires<[HasAVX512]>;

def : Pat<(f64 (extloadf32 addr:$src)),
          (VCVTSS2SDZrm (f64 (IMPLICIT_DEF)), addr:$src)>,
      Requires<[HasAVX512, OptForSize]>;

def : Pat<(f64 (extloadf32 addr:$src)),
          (VCVTSS2SDZrr (f64 (IMPLICIT_DEF)), (VMOVSSZrm addr:$src))>,
          Requires<[HasAVX512, OptForSpeed]>;

def : Pat<(f32 (fpround FR64X:$src)),
          (VCVTSD2SSZrr (COPY_TO_REGCLASS FR64X:$src, FR32X), FR64X:$src)>,
           Requires<[HasAVX512]>;

def : Pat<(v4f32 (X86Movss
                   (v4f32 VR128X:$dst),
                   (v4f32 (scalar_to_vector
                     (f32 (fpround (f64 (extractelt VR128X:$src, (iPTR 0))))))))),
          (VCVTSD2SSZrr_Int VR128X:$dst, VR128X:$src)>,
          Requires<[HasAVX512]>;

def : Pat<(v2f64 (X86Movsd
                   (v2f64 VR128X:$dst),
                   (v2f64 (scalar_to_vector
                     (f64 (fpextend (f32 (extractelt VR128X:$src, (iPTR 0))))))))),
          (VCVTSS2SDZrr_Int VR128X:$dst, VR128X:$src)>,
          Requires<[HasAVX512]>;

//===----------------------------------------------------------------------===//
// AVX-512  Vector convert from signed/unsigned integer to float/double
//          and from float/double to signed/unsigned integer
//===----------------------------------------------------------------------===//

multiclass avx512_vcvt_fp<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                         X86VectorVTInfo _Src, SDNode OpNode,
                         string Broadcast = _.BroadcastStr,
                         string Alias = "", X86MemOperand MemOp = _Src.MemOp> {

  defm rr : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _Src.RC:$src), OpcodeStr, "$src", "$src",
                         (_.VT (OpNode (_Src.VT _Src.RC:$src)))>, EVEX;

  defm rm : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins MemOp:$src), OpcodeStr#Alias, "$src", "$src",
                         (_.VT (OpNode (_Src.VT
                             (bitconvert (_Src.LdFrag addr:$src)))))>, EVEX;

  defm rmb : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _Src.ScalarMemOp:$src), OpcodeStr,
                         "${src}"##Broadcast, "${src}"##Broadcast,
                         (_.VT (OpNode (_Src.VT
                                  (X86VBroadcast (_Src.ScalarLdFrag addr:$src)))
                            ))>, EVEX, EVEX_B;
}
// Coversion with SAE - suppress all exceptions
multiclass avx512_vcvt_fp_sae<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                         X86VectorVTInfo _Src, SDNode OpNodeRnd> {
  defm rrb : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                        (ins _Src.RC:$src), OpcodeStr,
                        "{sae}, $src", "$src, {sae}",
                        (_.VT (OpNodeRnd (_Src.VT _Src.RC:$src),
                               (i32 FROUND_NO_EXC)))>,
                        EVEX, EVEX_B;
}

// Conversion with rounding control (RC)
multiclass avx512_vcvt_fp_rc<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                         X86VectorVTInfo _Src, SDNode OpNodeRnd> {
  defm rrb : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                        (ins _Src.RC:$src, AVX512RC:$rc), OpcodeStr,
                        "$rc, $src", "$src, $rc",
                        (_.VT (OpNodeRnd (_Src.VT _Src.RC:$src), (i32 imm:$rc)))>,
                        EVEX, EVEX_B, EVEX_RC;
}

// Extend Float to Double
multiclass avx512_cvtps2pd<bits<8> opc, string OpcodeStr> {
  let Predicates = [HasAVX512] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8f64_info, v8f32x_info, fpextend>,
             avx512_vcvt_fp_sae<opc, OpcodeStr, v8f64_info, v8f32x_info,
                                X86vfpextRnd>, EVEX_V512;
  }
  let Predicates = [HasVLX] in {
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v2f64x_info, v4f32x_info,
                               X86vfpext, "{1to2}", "", f64mem>, EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4f64x_info, v4f32x_info, fpextend>,
                                     EVEX_V256;
  }
}

// Truncate Double to Float
multiclass avx512_cvtpd2ps<bits<8> opc, string OpcodeStr> {
  let Predicates = [HasAVX512] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8f32x_info, v8f64_info, fpround>,
             avx512_vcvt_fp_rc<opc, OpcodeStr, v8f32x_info, v8f64_info,
                               X86vfproundRnd>, EVEX_V512;
  }
  let Predicates = [HasVLX] in {
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v4f32x_info, v2f64x_info,
                               X86vfpround, "{1to2}", "{x}">, EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4f32x_info, v4f64x_info, fpround,
                               "{1to4}", "{y}">, EVEX_V256;

    def : InstAlias<OpcodeStr##"x\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z128rr") VR128X:$dst, VR128X:$src), 0>;
    def : InstAlias<OpcodeStr##"x\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z128rm") VR128X:$dst, f128mem:$src), 0>;
    def : InstAlias<OpcodeStr##"y\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z256rr") VR128X:$dst, VR256X:$src), 0>;
    def : InstAlias<OpcodeStr##"y\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z256rm") VR128X:$dst, f256mem:$src), 0>;
  }
}

defm VCVTPD2PS : avx512_cvtpd2ps<0x5A, "vcvtpd2ps">,
                                  VEX_W, PD, EVEX_CD8<64, CD8VF>;
defm VCVTPS2PD : avx512_cvtps2pd<0x5A, "vcvtps2pd">,
                                  PS, EVEX_CD8<32, CD8VH>;

def : Pat<(v8f64 (extloadv8f32 addr:$src)),
            (VCVTPS2PDZrm addr:$src)>;

let Predicates = [HasVLX] in {
  let AddedComplexity = 15 in
  def : Pat<(X86vzmovl (v2f64 (bitconvert
                               (v4f32 (X86vfpround (v2f64 VR128X:$src)))))),
            (VCVTPD2PSZ128rr VR128X:$src)>;
  def : Pat<(v2f64 (extloadv2f32 addr:$src)),
              (VCVTPS2PDZ128rm addr:$src)>;
  def : Pat<(v4f64 (extloadv4f32 addr:$src)),
              (VCVTPS2PDZ256rm addr:$src)>;
}

// Convert Signed/Unsigned Doubleword to Double
multiclass avx512_cvtdq2pd<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           SDNode OpNode128> {
  // No rounding in this op
  let Predicates = [HasAVX512] in
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8f64_info, v8i32x_info, OpNode>,
                                     EVEX_V512;

  let Predicates = [HasVLX] in {
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v2f64x_info, v4i32x_info,
                                     OpNode128, "{1to2}", "", i64mem>, EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4f64x_info, v4i32x_info, OpNode>,
                                     EVEX_V256;
  }
}

// Convert Signed/Unsigned Doubleword to Float
multiclass avx512_cvtdq2ps<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           SDNode OpNodeRnd> {
  let Predicates = [HasAVX512] in
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v16f32_info, v16i32_info, OpNode>,
             avx512_vcvt_fp_rc<opc, OpcodeStr, v16f32_info, v16i32_info,
                               OpNodeRnd>, EVEX_V512;

  let Predicates = [HasVLX] in {
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v4f32x_info, v4i32x_info, OpNode>,
                                     EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v8f32x_info, v8i32x_info, OpNode>,
                                     EVEX_V256;
  }
}

// Convert Float to Signed/Unsigned Doubleword with truncation
multiclass avx512_cvttps2dq<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode, SDNode OpNodeRnd> {
  let Predicates = [HasAVX512] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v16i32_info, v16f32_info, OpNode>,
             avx512_vcvt_fp_sae<opc, OpcodeStr, v16i32_info, v16f32_info,
                                OpNodeRnd>, EVEX_V512;
  }
  let Predicates = [HasVLX] in {
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v4i32x_info, v4f32x_info, OpNode>,
                                     EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v8i32x_info, v8f32x_info, OpNode>,
                                     EVEX_V256;
  }
}

// Convert Float to Signed/Unsigned Doubleword
multiclass avx512_cvtps2dq<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode, SDNode OpNodeRnd> {
  let Predicates = [HasAVX512] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v16i32_info, v16f32_info, OpNode>,
             avx512_vcvt_fp_rc<opc, OpcodeStr, v16i32_info, v16f32_info,
                                OpNodeRnd>, EVEX_V512;
  }
  let Predicates = [HasVLX] in {
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v4i32x_info, v4f32x_info, OpNode>,
                                     EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v8i32x_info, v8f32x_info, OpNode>,
                                     EVEX_V256;
  }
}

// Convert Double to Signed/Unsigned Doubleword with truncation
multiclass avx512_cvttpd2dq<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            SDNode OpNode128, SDNode OpNodeRnd> {
  let Predicates = [HasAVX512] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8i32x_info, v8f64_info, OpNode>,
             avx512_vcvt_fp_sae<opc, OpcodeStr, v8i32x_info, v8f64_info,
                                OpNodeRnd>, EVEX_V512;
  }
  let Predicates = [HasVLX] in {
    // we need "x"/"y" suffixes in order to distinguish between 128 and 256
    // memory forms of these instructions in Asm Parser. They have the same
    // dest type - 'v4i32x_info'. We also specify the broadcast string explicitly
    // due to the same reason.
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v4i32x_info, v2f64x_info,
                               OpNode128, "{1to2}", "{x}">, EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4i32x_info, v4f64x_info, OpNode,
                               "{1to4}", "{y}">, EVEX_V256;

    def : InstAlias<OpcodeStr##"x\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z128rr") VR128X:$dst, VR128X:$src), 0>;
    def : InstAlias<OpcodeStr##"x\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z128rm") VR128X:$dst, i128mem:$src), 0>;
    def : InstAlias<OpcodeStr##"y\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z256rr") VR128X:$dst, VR256X:$src), 0>;
    def : InstAlias<OpcodeStr##"y\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z256rm") VR128X:$dst, i256mem:$src), 0>;
  }
}

// Convert Double to Signed/Unsigned Doubleword
multiclass avx512_cvtpd2dq<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode, SDNode OpNodeRnd> {
  let Predicates = [HasAVX512] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8i32x_info, v8f64_info, OpNode>,
             avx512_vcvt_fp_rc<opc, OpcodeStr, v8i32x_info, v8f64_info,
                               OpNodeRnd>, EVEX_V512;
  }
  let Predicates = [HasVLX] in {
    // we need "x"/"y" suffixes in order to distinguish between 128 and 256
    // memory forms of these instructions in Asm Parcer. They have the same
    // dest type - 'v4i32x_info'. We also specify the broadcast string explicitly
    // due to the same reason.
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v4i32x_info, v2f64x_info, OpNode,
                               "{1to2}", "{x}">, EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4i32x_info, v4f64x_info, OpNode,
                               "{1to4}", "{y}">, EVEX_V256;

    def : InstAlias<OpcodeStr##"x\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z128rr") VR128X:$dst, VR128X:$src), 0>;
    def : InstAlias<OpcodeStr##"x\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z128rm") VR128X:$dst, f128mem:$src), 0>;
    def : InstAlias<OpcodeStr##"y\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z256rr") VR128X:$dst, VR256X:$src), 0>;
    def : InstAlias<OpcodeStr##"y\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z256rm") VR128X:$dst, f256mem:$src), 0>;
  }
}

// Convert Double to Signed/Unsigned Quardword
multiclass avx512_cvtpd2qq<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode, SDNode OpNodeRnd> {
  let Predicates = [HasDQI] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8i64_info, v8f64_info, OpNode>,
             avx512_vcvt_fp_rc<opc, OpcodeStr, v8i64_info, v8f64_info,
                               OpNodeRnd>, EVEX_V512;
  }
  let Predicates = [HasDQI, HasVLX] in {
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v2i64x_info, v2f64x_info, OpNode>,
                               EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4i64x_info, v4f64x_info, OpNode>,
                               EVEX_V256;
  }
}

// Convert Double to Signed/Unsigned Quardword with truncation
multiclass avx512_cvttpd2qq<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode, SDNode OpNodeRnd> {
  let Predicates = [HasDQI] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8i64_info, v8f64_info, OpNode>,
             avx512_vcvt_fp_sae<opc, OpcodeStr, v8i64_info, v8f64_info,
                               OpNodeRnd>, EVEX_V512;
  }
  let Predicates = [HasDQI, HasVLX] in {
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v2i64x_info, v2f64x_info, OpNode>,
                               EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4i64x_info, v4f64x_info, OpNode>,
                               EVEX_V256;
  }
}

// Convert Signed/Unsigned Quardword to Double
multiclass avx512_cvtqq2pd<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode, SDNode OpNodeRnd> {
  let Predicates = [HasDQI] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8f64_info, v8i64_info, OpNode>,
             avx512_vcvt_fp_rc<opc, OpcodeStr, v8f64_info, v8i64_info,
                               OpNodeRnd>, EVEX_V512;
  }
  let Predicates = [HasDQI, HasVLX] in {
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v2f64x_info, v2i64x_info, OpNode>,
                               EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4f64x_info, v4i64x_info, OpNode>,
                               EVEX_V256;
  }
}

// Convert Float to Signed/Unsigned Quardword
multiclass avx512_cvtps2qq<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode, SDNode OpNodeRnd> {
  let Predicates = [HasDQI] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8i64_info, v8f32x_info, OpNode>,
             avx512_vcvt_fp_rc<opc, OpcodeStr, v8i64_info, v8f32x_info,
                               OpNodeRnd>, EVEX_V512;
  }
  let Predicates = [HasDQI, HasVLX] in {
    // Explicitly specified broadcast string, since we take only 2 elements
    // from v4f32x_info source
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v2i64x_info, v4f32x_info, OpNode,
                               "{1to2}", "", f64mem>, EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4i64x_info, v4f32x_info, OpNode>,
                               EVEX_V256;
  }
}

// Convert Float to Signed/Unsigned Quardword with truncation
multiclass avx512_cvttps2qq<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            SDNode OpNode128, SDNode OpNodeRnd> {
  let Predicates = [HasDQI] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8i64_info, v8f32x_info, OpNode>,
             avx512_vcvt_fp_sae<opc, OpcodeStr, v8i64_info, v8f32x_info,
                               OpNodeRnd>, EVEX_V512;
  }
  let Predicates = [HasDQI, HasVLX] in {
    // Explicitly specified broadcast string, since we take only 2 elements
    // from v4f32x_info source
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v2i64x_info, v4f32x_info, OpNode128,
                               "{1to2}", "", f64mem>, EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4i64x_info, v4f32x_info, OpNode>,
                               EVEX_V256;
  }
}

// Convert Signed/Unsigned Quardword to Float
multiclass avx512_cvtqq2ps<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  SDNode OpNode128, SDNode OpNodeRnd> {
  let Predicates = [HasDQI] in {
    defm Z : avx512_vcvt_fp<opc, OpcodeStr, v8f32x_info, v8i64_info, OpNode>,
             avx512_vcvt_fp_rc<opc, OpcodeStr, v8f32x_info, v8i64_info,
                               OpNodeRnd>, EVEX_V512;
  }
  let Predicates = [HasDQI, HasVLX] in {
    // we need "x"/"y" suffixes in order to distinguish between 128 and 256
    // memory forms of these instructions in Asm Parcer. They have the same
    // dest type - 'v4i32x_info'. We also specify the broadcast string explicitly
    // due to the same reason.
    defm Z128 : avx512_vcvt_fp<opc, OpcodeStr, v4f32x_info, v2i64x_info, OpNode128,
                               "{1to2}", "{x}">, EVEX_V128;
    defm Z256 : avx512_vcvt_fp<opc, OpcodeStr, v4f32x_info, v4i64x_info, OpNode,
                               "{1to4}", "{y}">, EVEX_V256;

    def : InstAlias<OpcodeStr##"x\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z128rr") VR128X:$dst, VR128X:$src), 0>;
    def : InstAlias<OpcodeStr##"x\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z128rm") VR128X:$dst, i128mem:$src), 0>;
    def : InstAlias<OpcodeStr##"y\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z256rr") VR128X:$dst, VR256X:$src), 0>;
    def : InstAlias<OpcodeStr##"y\t{$src, $dst|$dst, $src}",
                    (!cast<Instruction>(NAME # "Z256rm") VR128X:$dst, i256mem:$src), 0>;
  }
}

defm VCVTDQ2PD : avx512_cvtdq2pd<0xE6, "vcvtdq2pd", sint_to_fp, X86VSintToFP>,
                                XS, EVEX_CD8<32, CD8VH>;

defm VCVTDQ2PS : avx512_cvtdq2ps<0x5B, "vcvtdq2ps", sint_to_fp,
                                X86VSintToFpRnd>,
                                PS, EVEX_CD8<32, CD8VF>;

defm VCVTTPS2DQ : avx512_cvttps2dq<0x5B, "vcvttps2dq", fp_to_sint,
                                X86cvttp2siRnd>,
                                XS, EVEX_CD8<32, CD8VF>;

defm VCVTTPD2DQ : avx512_cvttpd2dq<0xE6, "vcvttpd2dq", fp_to_sint, X86cvttp2si,
                                 X86cvttp2siRnd>,
                                 PD, VEX_W, EVEX_CD8<64, CD8VF>;

defm VCVTTPS2UDQ : avx512_cvttps2dq<0x78, "vcvttps2udq", fp_to_uint,
                                 X86cvttp2uiRnd>, PS,
                                 EVEX_CD8<32, CD8VF>;

defm VCVTTPD2UDQ : avx512_cvttpd2dq<0x78, "vcvttpd2udq", fp_to_uint,
                                 X86cvttp2ui, X86cvttp2uiRnd>, PS, VEX_W,
                                 EVEX_CD8<64, CD8VF>;

defm VCVTUDQ2PD : avx512_cvtdq2pd<0x7A, "vcvtudq2pd", uint_to_fp, X86VUintToFP>,
                                 XS, EVEX_CD8<32, CD8VH>;

defm VCVTUDQ2PS : avx512_cvtdq2ps<0x7A, "vcvtudq2ps", uint_to_fp,
                                 X86VUintToFpRnd>, XD,
                                 EVEX_CD8<32, CD8VF>;

defm VCVTPS2DQ : avx512_cvtps2dq<0x5B, "vcvtps2dq", X86cvtp2Int,
                                 X86cvtp2IntRnd>, PD, EVEX_CD8<32, CD8VF>;

defm VCVTPD2DQ : avx512_cvtpd2dq<0xE6, "vcvtpd2dq", X86cvtp2Int,
                                 X86cvtp2IntRnd>, XD, VEX_W,
                                 EVEX_CD8<64, CD8VF>;

defm VCVTPS2UDQ : avx512_cvtps2dq<0x79, "vcvtps2udq", X86cvtp2UInt,
                                 X86cvtp2UIntRnd>,
                                 PS, EVEX_CD8<32, CD8VF>;
defm VCVTPD2UDQ : avx512_cvtpd2dq<0x79, "vcvtpd2udq", X86cvtp2UInt,
                                 X86cvtp2UIntRnd>, VEX_W,
                                 PS, EVEX_CD8<64, CD8VF>;

defm VCVTPD2QQ : avx512_cvtpd2qq<0x7B, "vcvtpd2qq", X86cvtp2Int,
                                 X86cvtp2IntRnd>, VEX_W,
                                 PD, EVEX_CD8<64, CD8VF>;

defm VCVTPS2QQ : avx512_cvtps2qq<0x7B, "vcvtps2qq", X86cvtp2Int,
                                 X86cvtp2IntRnd>, PD, EVEX_CD8<32, CD8VH>;

defm VCVTPD2UQQ : avx512_cvtpd2qq<0x79, "vcvtpd2uqq", X86cvtp2UInt,
                                 X86cvtp2UIntRnd>, VEX_W,
                                 PD, EVEX_CD8<64, CD8VF>;

defm VCVTPS2UQQ : avx512_cvtps2qq<0x79, "vcvtps2uqq", X86cvtp2UInt,
                                 X86cvtp2UIntRnd>, PD, EVEX_CD8<32, CD8VH>;

defm VCVTTPD2QQ : avx512_cvttpd2qq<0x7A, "vcvttpd2qq", fp_to_sint,
                                 X86cvttp2siRnd>, VEX_W,
                                 PD, EVEX_CD8<64, CD8VF>;

defm VCVTTPS2QQ : avx512_cvttps2qq<0x7A, "vcvttps2qq", fp_to_sint, X86cvttp2si,
                                 X86cvttp2siRnd>, PD, EVEX_CD8<32, CD8VH>;

defm VCVTTPD2UQQ : avx512_cvttpd2qq<0x78, "vcvttpd2uqq", fp_to_uint,
                                 X86cvttp2uiRnd>, VEX_W,
                                 PD, EVEX_CD8<64, CD8VF>;

defm VCVTTPS2UQQ : avx512_cvttps2qq<0x78, "vcvttps2uqq", fp_to_uint, X86cvttp2ui,
                                 X86cvttp2uiRnd>, PD, EVEX_CD8<32, CD8VH>;

defm VCVTQQ2PD : avx512_cvtqq2pd<0xE6, "vcvtqq2pd", sint_to_fp,
                            X86VSintToFpRnd>, VEX_W, XS, EVEX_CD8<64, CD8VF>;

defm VCVTUQQ2PD : avx512_cvtqq2pd<0x7A, "vcvtuqq2pd", uint_to_fp,
                            X86VUintToFpRnd>, VEX_W, XS, EVEX_CD8<64, CD8VF>;

defm VCVTQQ2PS : avx512_cvtqq2ps<0x5B, "vcvtqq2ps", sint_to_fp, X86VSintToFP,
                            X86VSintToFpRnd>, VEX_W, PS, EVEX_CD8<64, CD8VF>;

defm VCVTUQQ2PS : avx512_cvtqq2ps<0x7A, "vcvtuqq2ps", uint_to_fp, X86VUintToFP,
                            X86VUintToFpRnd>, VEX_W, XD, EVEX_CD8<64, CD8VF>;

let Predicates = [HasAVX512, NoVLX] in {
def : Pat<(v8i32 (fp_to_uint (v8f32 VR256X:$src1))),
          (EXTRACT_SUBREG (v16i32 (VCVTTPS2UDQZrr
           (v16f32 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR256X:$src1, sub_ymm)))), sub_ymm)>;

def : Pat<(v4i32 (fp_to_uint (v4f32 VR128X:$src1))),
          (EXTRACT_SUBREG (v16i32 (VCVTTPS2UDQZrr
           (v16f32 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR128X:$src1, sub_xmm)))), sub_xmm)>;

def : Pat<(v4i32 (fp_to_uint (v4f64 VR256X:$src1))),
          (EXTRACT_SUBREG (v8i32 (VCVTTPD2UDQZrr
           (v8f64 (INSERT_SUBREG (IMPLICIT_DEF),
                                 VR256X:$src1, sub_ymm)))), sub_xmm)>;

def : Pat<(v4i32 (X86cvttp2ui (v2f64 VR128X:$src))),
          (EXTRACT_SUBREG (v8i32 (VCVTTPD2UDQZrr
           (v8f64 (INSERT_SUBREG (IMPLICIT_DEF),
                                 VR128X:$src, sub_xmm)))), sub_xmm)>;

def : Pat<(v8f32 (uint_to_fp (v8i32 VR256X:$src1))),
          (EXTRACT_SUBREG (v16f32 (VCVTUDQ2PSZrr
           (v16i32 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR256X:$src1, sub_ymm)))), sub_ymm)>;

def : Pat<(v4f32 (uint_to_fp (v4i32 VR128X:$src1))),
          (EXTRACT_SUBREG (v16f32 (VCVTUDQ2PSZrr
           (v16i32 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR128X:$src1, sub_xmm)))), sub_xmm)>;

def : Pat<(v4f64 (uint_to_fp (v4i32 VR128X:$src1))),
          (EXTRACT_SUBREG (v8f64 (VCVTUDQ2PDZrr
           (v8i32 (INSERT_SUBREG (IMPLICIT_DEF),
                                 VR128X:$src1, sub_xmm)))), sub_ymm)>;

def : Pat<(v2f64 (X86VUintToFP (v4i32 VR128X:$src1))),
          (EXTRACT_SUBREG (v8f64 (VCVTUDQ2PDZrr
           (v8i32 (INSERT_SUBREG (IMPLICIT_DEF),
                                 VR128X:$src1, sub_xmm)))), sub_xmm)>;
}

let Predicates = [HasAVX512, HasVLX] in {
  let AddedComplexity = 15 in {
    def : Pat<(X86vzmovl (v2i64 (bitconvert
                                (v4i32 (X86cvtp2Int (v2f64 VR128X:$src)))))),
              (VCVTPD2DQZ128rr VR128X:$src)>;
    def : Pat<(v4i32 (bitconvert (X86vzmovl (v2i64 (bitconvert
                                 (v4i32 (X86cvtp2UInt (v2f64 VR128X:$src)))))))),
              (VCVTPD2UDQZ128rr VR128X:$src)>;
    def : Pat<(X86vzmovl (v2i64 (bitconvert
                                (v4i32 (X86cvttp2si (v2f64 VR128X:$src)))))),
              (VCVTTPD2DQZ128rr VR128X:$src)>;
    def : Pat<(v4i32 (bitconvert (X86vzmovl (v2i64 (bitconvert
                                 (v4i32 (X86cvttp2ui (v2f64 VR128X:$src)))))))),
              (VCVTTPD2UDQZ128rr VR128X:$src)>;
  }
}

let Predicates = [HasAVX512] in {
  def : Pat<(v8f32 (fpround (loadv8f64 addr:$src))),
            (VCVTPD2PSZrm addr:$src)>;
  def : Pat<(v8f64 (extloadv8f32 addr:$src)),
            (VCVTPS2PDZrm addr:$src)>;
}

let Predicates = [HasDQI, HasVLX] in {
  let AddedComplexity = 15 in {
    def : Pat<(X86vzmovl (v2f64 (bitconvert
                                (v4f32 (X86VSintToFP (v2i64 VR128X:$src)))))),
              (VCVTQQ2PSZ128rr VR128X:$src)>;
    def : Pat<(X86vzmovl (v2f64 (bitconvert
                                (v4f32 (X86VUintToFP (v2i64 VR128X:$src)))))),
              (VCVTUQQ2PSZ128rr VR128X:$src)>;
  }
}

let Predicates = [HasDQI, NoVLX] in {
def : Pat<(v2i64 (fp_to_sint (v2f64 VR128X:$src1))),
          (EXTRACT_SUBREG (v8i64 (VCVTTPD2QQZrr
           (v8f64 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR128X:$src1, sub_xmm)))), sub_xmm)>;

def : Pat<(v4i64 (fp_to_sint (v4f32 VR128X:$src1))),
          (EXTRACT_SUBREG (v8i64 (VCVTTPS2QQZrr
           (v8f32 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR128X:$src1, sub_xmm)))), sub_ymm)>;

def : Pat<(v4i64 (fp_to_sint (v4f64 VR256X:$src1))),
          (EXTRACT_SUBREG (v8i64 (VCVTTPD2QQZrr
           (v8f64 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR256X:$src1, sub_ymm)))), sub_ymm)>;

def : Pat<(v2i64 (fp_to_uint (v2f64 VR128X:$src1))),
          (EXTRACT_SUBREG (v8i64 (VCVTTPD2UQQZrr
           (v8f64 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR128X:$src1, sub_xmm)))), sub_xmm)>;

def : Pat<(v4i64 (fp_to_uint (v4f32 VR128X:$src1))),
          (EXTRACT_SUBREG (v8i64 (VCVTTPS2UQQZrr
           (v8f32 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR128X:$src1, sub_xmm)))), sub_ymm)>;

def : Pat<(v4i64 (fp_to_uint (v4f64 VR256X:$src1))),
          (EXTRACT_SUBREG (v8i64 (VCVTTPD2UQQZrr
           (v8f64 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR256X:$src1, sub_ymm)))), sub_ymm)>;

def : Pat<(v4f32 (sint_to_fp (v4i64 VR256X:$src1))),
          (EXTRACT_SUBREG (v8f32 (VCVTQQ2PSZrr
           (v8i64 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR256X:$src1, sub_ymm)))), sub_xmm)>;

def : Pat<(v2f64 (sint_to_fp (v2i64 VR128X:$src1))),
          (EXTRACT_SUBREG (v8f64 (VCVTQQ2PDZrr
           (v8i64 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR128X:$src1, sub_xmm)))), sub_xmm)>;

def : Pat<(v4f64 (sint_to_fp (v4i64 VR256X:$src1))),
          (EXTRACT_SUBREG (v8f64 (VCVTQQ2PDZrr
           (v8i64 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR256X:$src1, sub_ymm)))), sub_ymm)>;

def : Pat<(v4f32 (uint_to_fp (v4i64 VR256X:$src1))),
          (EXTRACT_SUBREG (v8f32 (VCVTUQQ2PSZrr
           (v8i64 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR256X:$src1, sub_ymm)))), sub_xmm)>;

def : Pat<(v2f64 (uint_to_fp (v2i64 VR128X:$src1))),
          (EXTRACT_SUBREG (v8f64 (VCVTUQQ2PDZrr
           (v8i64 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR128X:$src1, sub_xmm)))), sub_xmm)>;

def : Pat<(v4f64 (uint_to_fp (v4i64 VR256X:$src1))),
          (EXTRACT_SUBREG (v8f64 (VCVTUQQ2PDZrr
           (v8i64 (INSERT_SUBREG (IMPLICIT_DEF),
                                  VR256X:$src1, sub_ymm)))), sub_ymm)>;
}

//===----------------------------------------------------------------------===//
// Half precision conversion instructions
//===----------------------------------------------------------------------===//
multiclass avx512_cvtph2ps<X86VectorVTInfo _dest, X86VectorVTInfo _src,
                           X86MemOperand x86memop, PatFrag ld_frag> {
  defm rr : AVX512_maskable<0x13, MRMSrcReg, _dest ,(outs _dest.RC:$dst), (ins _src.RC:$src),
                    "vcvtph2ps", "$src", "$src",
                   (X86cvtph2ps (_src.VT _src.RC:$src),
                                                (i32 FROUND_CURRENT))>, T8PD;
  defm rm : AVX512_maskable<0x13, MRMSrcMem, _dest, (outs _dest.RC:$dst), (ins x86memop:$src),
                    "vcvtph2ps", "$src", "$src",
                    (X86cvtph2ps (_src.VT (bitconvert (ld_frag addr:$src))),
                                     (i32 FROUND_CURRENT))>, T8PD;
}

multiclass avx512_cvtph2ps_sae<X86VectorVTInfo _dest, X86VectorVTInfo _src> {
  defm rb : AVX512_maskable<0x13, MRMSrcReg, _dest ,(outs _dest.RC:$dst), (ins _src.RC:$src),
                    "vcvtph2ps", "{sae}, $src", "$src, {sae}",
                   (X86cvtph2ps (_src.VT _src.RC:$src),
                                                (i32 FROUND_NO_EXC))>, T8PD, EVEX_B;

}

let Predicates = [HasAVX512] in {
  defm VCVTPH2PSZ : avx512_cvtph2ps<v16f32_info, v16i16x_info, f256mem, loadv4i64>,
                    avx512_cvtph2ps_sae<v16f32_info, v16i16x_info>,
                    EVEX, EVEX_V512, EVEX_CD8<32, CD8VH>;
  let Predicates = [HasVLX] in {
    defm VCVTPH2PSZ256 : avx512_cvtph2ps<v8f32x_info, v8i16x_info, f128mem,
                         loadv2i64>,EVEX, EVEX_V256, EVEX_CD8<32, CD8VH>;
    defm VCVTPH2PSZ128 : avx512_cvtph2ps<v4f32x_info, v8i16x_info, f64mem,
                         loadv2i64>, EVEX, EVEX_V128, EVEX_CD8<32, CD8VH>;
  }
}

multiclass avx512_cvtps2ph<X86VectorVTInfo _dest, X86VectorVTInfo _src,
                           X86MemOperand x86memop> {
  defm rr : AVX512_maskable<0x1D, MRMDestReg, _dest ,(outs _dest.RC:$dst),
                   (ins _src.RC:$src1, i32u8imm:$src2),
                   "vcvtps2ph", "$src2, $src1", "$src1, $src2",
                   (X86cvtps2ph (_src.VT _src.RC:$src1),
                                (i32 imm:$src2)),
                   NoItinerary, 0, 0, X86select>, AVX512AIi8Base;
  def mr : AVX512AIi8<0x1D, MRMDestMem, (outs),
             (ins x86memop:$dst, _src.RC:$src1, i32u8imm:$src2),
             "vcvtps2ph\t{$src2, $src1, $dst|$dst, $src1, $src2}",
             [(store (_dest.VT (X86cvtps2ph (_src.VT _src.RC:$src1),
                                     (i32 imm:$src2))),
                                     addr:$dst)]>;
  let hasSideEffects = 0, mayStore = 1 in
  def mrk : AVX512AIi8<0x1D, MRMDestMem, (outs),
             (ins x86memop:$dst, _dest.KRCWM:$mask, _src.RC:$src1, i32u8imm:$src2),
             "vcvtps2ph\t{$src2, $src1, $dst {${mask}}|$dst {${mask}}, $src1, $src2}",
              []>, EVEX_K;
}
multiclass avx512_cvtps2ph_sae<X86VectorVTInfo _dest, X86VectorVTInfo _src> {
  let hasSideEffects = 0 in
  defm rb : AVX512_maskable_in_asm<0x1D, MRMDestReg, _dest,
                   (outs _dest.RC:$dst),
                   (ins _src.RC:$src1, i32u8imm:$src2),
                   "vcvtps2ph", "$src2, {sae}, $src1", "$src1, {sae}, $src2",
                   []>, EVEX_B, AVX512AIi8Base;
}
let Predicates = [HasAVX512] in {
  defm VCVTPS2PHZ : avx512_cvtps2ph<v16i16x_info, v16f32_info, f256mem>,
                    avx512_cvtps2ph_sae<v16i16x_info, v16f32_info>,
                      EVEX, EVEX_V512, EVEX_CD8<32, CD8VH>;
  let Predicates = [HasVLX] in {
    defm VCVTPS2PHZ256 : avx512_cvtps2ph<v8i16x_info, v8f32x_info, f128mem>,
                        EVEX, EVEX_V256, EVEX_CD8<32, CD8VH>;
    defm VCVTPS2PHZ128 : avx512_cvtps2ph<v8i16x_info, v4f32x_info, f64mem>,
                        EVEX, EVEX_V128, EVEX_CD8<32, CD8VH>;
  }
}

// Patterns for matching conversions from float to half-float and vice versa.
let Predicates = [HasVLX] in {
  // Use MXCSR.RC for rounding instead of explicitly specifying the default
  // rounding mode (Nearest-Even, encoded as 0). Both are equivalent in the
  // configurations we support (the default). However, falling back to MXCSR is
  // more consistent with other instructions, which are always controlled by it.
  // It's encoded as 0b100.
  def : Pat<(fp_to_f16 FR32X:$src),
            (i16 (EXTRACT_SUBREG (VMOVPDI2DIZrr (VCVTPS2PHZ128rr
              (COPY_TO_REGCLASS FR32X:$src, VR128X), 4)), sub_16bit))>;

  def : Pat<(f16_to_fp GR16:$src),
            (f32 (COPY_TO_REGCLASS (VCVTPH2PSZ128rr
              (COPY_TO_REGCLASS (MOVSX32rr16 GR16:$src), VR128X)), FR32X)) >;

  def : Pat<(f16_to_fp (i16 (fp_to_f16 FR32X:$src))),
            (f32 (COPY_TO_REGCLASS (VCVTPH2PSZ128rr
              (VCVTPS2PHZ128rr (COPY_TO_REGCLASS FR32X:$src, VR128X), 4)), FR32X)) >;
}

// Patterns for matching float to half-float conversion when AVX512 is supported
// but F16C isn't. In that case we have to use 512-bit vectors.
let Predicates = [HasAVX512, NoVLX, NoF16C] in {
  def : Pat<(fp_to_f16 FR32X:$src),
            (i16 (EXTRACT_SUBREG
                  (VMOVPDI2DIZrr
                   (v8i16 (EXTRACT_SUBREG
                    (VCVTPS2PHZrr
                     (INSERT_SUBREG (v16f32 (IMPLICIT_DEF)),
                      (v4f32 (COPY_TO_REGCLASS FR32X:$src, VR128X)),
                      sub_xmm), 4), sub_xmm))), sub_16bit))>;

  def : Pat<(f16_to_fp GR16:$src),
            (f32 (COPY_TO_REGCLASS
                  (v4f32 (EXTRACT_SUBREG
                   (VCVTPH2PSZrr
                    (INSERT_SUBREG (v16i16 (IMPLICIT_DEF)),
                     (v8i16 (COPY_TO_REGCLASS (MOVSX32rr16 GR16:$src), VR128X)),
                     sub_xmm)), sub_xmm)), FR32X))>;

  def : Pat<(f16_to_fp (i16 (fp_to_f16 FR32X:$src))),
            (f32 (COPY_TO_REGCLASS
                  (v4f32 (EXTRACT_SUBREG
                          (VCVTPH2PSZrr
                           (VCVTPS2PHZrr (INSERT_SUBREG (v16f32 (IMPLICIT_DEF)),
                            (v4f32 (COPY_TO_REGCLASS FR32X:$src, VR128X)),
                            sub_xmm), 4)), sub_xmm)), FR32X))>;
}

//  Unordered/Ordered scalar fp compare with Sea and set EFLAGS
multiclass avx512_ord_cmp_sae<bits<8> opc, X86VectorVTInfo _,
                            string OpcodeStr> {
  def rb: AVX512<opc, MRMSrcReg, (outs), (ins _.RC:$src1, _.RC:$src2),
                 !strconcat(OpcodeStr, "\t{{sae}, $src2, $src1|$src1, $src2, {sae}}"),
                 [], IIC_SSE_COMIS_RR>, EVEX, EVEX_B, VEX_LIG, EVEX_V128,
                 Sched<[WriteFAdd]>;
}

let Defs = [EFLAGS], Predicates = [HasAVX512] in {
  defm VUCOMISSZ : avx512_ord_cmp_sae<0x2E, v4f32x_info, "vucomiss">,
                                   AVX512PSIi8Base, EVEX_CD8<32, CD8VT1>;
  defm VUCOMISDZ : avx512_ord_cmp_sae<0x2E, v2f64x_info, "vucomisd">,
                                   AVX512PDIi8Base, VEX_W, EVEX_CD8<64, CD8VT1>;
  defm VCOMISSZ : avx512_ord_cmp_sae<0x2F, v4f32x_info, "vcomiss">,
                                   AVX512PSIi8Base, EVEX_CD8<32, CD8VT1>;
  defm VCOMISDZ : avx512_ord_cmp_sae<0x2F, v2f64x_info, "vcomisd">,
                                   AVX512PDIi8Base, VEX_W, EVEX_CD8<64, CD8VT1>;
}

let Defs = [EFLAGS], Predicates = [HasAVX512] in {
  defm VUCOMISSZ : sse12_ord_cmp<0x2E, FR32X, X86cmp, f32, f32mem, loadf32,
                                 "ucomiss">, PS, EVEX, VEX_LIG,
                                 EVEX_CD8<32, CD8VT1>;
  defm VUCOMISDZ : sse12_ord_cmp<0x2E, FR64X, X86cmp, f64, f64mem, loadf64,
                                  "ucomisd">, PD, EVEX,
                                  VEX_LIG, VEX_W, EVEX_CD8<64, CD8VT1>;
  let Pattern = []<dag> in {
    defm VCOMISSZ  : sse12_ord_cmp<0x2F, FR32X, undef, f32, f32mem, loadf32,
                                   "comiss">, PS, EVEX, VEX_LIG,
                                   EVEX_CD8<32, CD8VT1>;
    defm VCOMISDZ  : sse12_ord_cmp<0x2F, FR64X, undef, f64, f64mem, loadf64,
                                   "comisd">, PD, EVEX,
                                    VEX_LIG, VEX_W, EVEX_CD8<64, CD8VT1>;
  }
  let isCodeGenOnly = 1 in {
    defm Int_VUCOMISSZ  : sse12_ord_cmp_int<0x2E, VR128X, X86ucomi, v4f32, ssmem,
                              sse_load_f32, "ucomiss">, PS, EVEX, VEX_LIG,
                              EVEX_CD8<32, CD8VT1>;
    defm Int_VUCOMISDZ  : sse12_ord_cmp_int<0x2E, VR128X, X86ucomi, v2f64, sdmem,
                              sse_load_f64, "ucomisd">, PD, EVEX,
                              VEX_LIG, VEX_W, EVEX_CD8<64, CD8VT1>;

    defm Int_VCOMISSZ  : sse12_ord_cmp_int<0x2F, VR128X, X86comi, v4f32, ssmem,
                              sse_load_f32, "comiss">, PS, EVEX, VEX_LIG,
                              EVEX_CD8<32, CD8VT1>;
    defm Int_VCOMISDZ  : sse12_ord_cmp_int<0x2F, VR128X, X86comi, v2f64, sdmem,
                              sse_load_f64, "comisd">, PD, EVEX,
                              VEX_LIG, VEX_W, EVEX_CD8<64, CD8VT1>;
  }
}

/// avx512_fp14_s rcp14ss, rcp14sd, rsqrt14ss, rsqrt14sd
multiclass avx512_fp14_s<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _> {
  let Predicates = [HasAVX512], ExeDomain = _.ExeDomain in {
  defm rr : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                           (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                           "$src2, $src1", "$src1, $src2",
                           (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2))>, EVEX_4V;
  defm rm : AVX512_maskable_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr,
                         "$src2, $src1", "$src1, $src2",
                         (OpNode (_.VT _.RC:$src1),
                          (_.VT (scalar_to_vector (_.ScalarLdFrag addr:$src2))))>, EVEX_4V;
}
}

defm VRCP14SS   : avx512_fp14_s<0x4D, "vrcp14ss", X86frcp14s, f32x_info>,
                  EVEX_CD8<32, CD8VT1>, T8PD;
defm VRCP14SD   : avx512_fp14_s<0x4D, "vrcp14sd", X86frcp14s, f64x_info>,
                  VEX_W, EVEX_CD8<64, CD8VT1>, T8PD;
defm VRSQRT14SS   : avx512_fp14_s<0x4F, "vrsqrt14ss", X86frsqrt14s, f32x_info>,
                  EVEX_CD8<32, CD8VT1>, T8PD;
defm VRSQRT14SD   : avx512_fp14_s<0x4F, "vrsqrt14sd", X86frsqrt14s, f64x_info>,
                  VEX_W, EVEX_CD8<64, CD8VT1>, T8PD;

/// avx512_fp14_p rcp14ps, rcp14pd, rsqrt14ps, rsqrt14pd
multiclass avx512_fp14_p<bits<8> opc, string OpcodeStr, SDNode OpNode,
                         X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
  defm r: AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _.RC:$src), OpcodeStr, "$src", "$src",
                         (_.FloatVT (OpNode _.RC:$src))>, EVEX, T8PD;
  defm m: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.MemOp:$src), OpcodeStr, "$src", "$src",
                         (OpNode (_.FloatVT
                           (bitconvert (_.LdFrag addr:$src))))>, EVEX, T8PD;
  defm mb: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                          (ins _.ScalarMemOp:$src), OpcodeStr,
                          "${src}"##_.BroadcastStr, "${src}"##_.BroadcastStr,
                          (OpNode (_.FloatVT
                            (X86VBroadcast (_.ScalarLdFrag addr:$src))))>,
                          EVEX, T8PD, EVEX_B;
  }
}

multiclass avx512_fp14_p_vl_all<bits<8> opc, string OpcodeStr, SDNode OpNode> {
  defm PSZ : avx512_fp14_p<opc, !strconcat(OpcodeStr, "ps"), OpNode, v16f32_info>,
                          EVEX_V512, EVEX_CD8<32, CD8VF>;
  defm PDZ : avx512_fp14_p<opc, !strconcat(OpcodeStr, "pd"), OpNode, v8f64_info>,
                          EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;

  // Define only if AVX512VL feature is present.
  let Predicates = [HasVLX] in {
    defm PSZ128 : avx512_fp14_p<opc, !strconcat(OpcodeStr, "ps"),
                                OpNode, v4f32x_info>,
                               EVEX_V128, EVEX_CD8<32, CD8VF>;
    defm PSZ256 : avx512_fp14_p<opc, !strconcat(OpcodeStr, "ps"),
                                OpNode, v8f32x_info>,
                               EVEX_V256, EVEX_CD8<32, CD8VF>;
    defm PDZ128 : avx512_fp14_p<opc, !strconcat(OpcodeStr, "pd"),
                                OpNode, v2f64x_info>,
                               EVEX_V128, VEX_W, EVEX_CD8<64, CD8VF>;
    defm PDZ256 : avx512_fp14_p<opc, !strconcat(OpcodeStr, "pd"),
                                OpNode, v4f64x_info>,
                               EVEX_V256, VEX_W, EVEX_CD8<64, CD8VF>;
  }
}

defm VRSQRT14 : avx512_fp14_p_vl_all<0x4E, "vrsqrt14", X86frsqrt>;
defm VRCP14 : avx512_fp14_p_vl_all<0x4C, "vrcp14", X86frcp>;

/// avx512_fp28_s rcp28ss, rcp28sd, rsqrt28ss, rsqrt28sd
multiclass avx512_fp28_s<bits<8> opc, string OpcodeStr,X86VectorVTInfo _,
                         SDNode OpNode> {
  let ExeDomain = _.ExeDomain in {
  defm r : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                           (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                           "$src2, $src1", "$src1, $src2",
                           (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                           (i32 FROUND_CURRENT))>;

  defm rb : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                            (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                            "{sae}, $src2, $src1", "$src1, $src2, {sae}",
                            (OpNode (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                            (i32 FROUND_NO_EXC))>, EVEX_B;

  defm m : AVX512_maskable_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr,
                         "$src2, $src1", "$src1, $src2",
                         (OpNode (_.VT _.RC:$src1),
                          (_.VT (scalar_to_vector (_.ScalarLdFrag addr:$src2))),
                         (i32 FROUND_CURRENT))>;
  }
}

multiclass avx512_eri_s<bits<8> opc, string OpcodeStr, SDNode OpNode> {
  defm SS : avx512_fp28_s<opc, OpcodeStr#"ss", f32x_info, OpNode>,
              EVEX_CD8<32, CD8VT1>;
  defm SD : avx512_fp28_s<opc, OpcodeStr#"sd", f64x_info, OpNode>,
              EVEX_CD8<64, CD8VT1>, VEX_W;
}

let Predicates = [HasERI] in {
  defm VRCP28   : avx512_eri_s<0xCB, "vrcp28",   X86rcp28s>,   T8PD, EVEX_4V;
  defm VRSQRT28 : avx512_eri_s<0xCD, "vrsqrt28", X86rsqrt28s>, T8PD, EVEX_4V;
}

defm VGETEXP   : avx512_eri_s<0x43, "vgetexp", X86fgetexpRnds>, T8PD, EVEX_4V;
/// avx512_fp28_p rcp28ps, rcp28pd, rsqrt28ps, rsqrt28pd

multiclass avx512_fp28_p<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                         SDNode OpNode> {
  let ExeDomain = _.ExeDomain in {
  defm r : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _.RC:$src), OpcodeStr, "$src", "$src",
                         (OpNode (_.VT _.RC:$src), (i32 FROUND_CURRENT))>;

  defm m : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.MemOp:$src), OpcodeStr, "$src", "$src",
                         (OpNode (_.FloatVT
                             (bitconvert (_.LdFrag addr:$src))),
                          (i32 FROUND_CURRENT))>;

  defm mb : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.ScalarMemOp:$src), OpcodeStr,
                         "${src}"##_.BroadcastStr, "${src}"##_.BroadcastStr,
                         (OpNode (_.FloatVT
                                  (X86VBroadcast (_.ScalarLdFrag addr:$src))),
                                 (i32 FROUND_CURRENT))>, EVEX_B;
  }
}
multiclass avx512_fp28_p_round<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                         SDNode OpNode> {
  let ExeDomain = _.ExeDomain in
  defm rb : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                        (ins _.RC:$src), OpcodeStr,
                        "{sae}, $src", "$src, {sae}",
                        (OpNode (_.VT _.RC:$src), (i32 FROUND_NO_EXC))>, EVEX_B;
}

multiclass  avx512_eri<bits<8> opc, string OpcodeStr, SDNode OpNode> {
   defm PS : avx512_fp28_p<opc, OpcodeStr#"ps", v16f32_info, OpNode>,
             avx512_fp28_p_round<opc, OpcodeStr#"ps", v16f32_info, OpNode>,
             T8PD, EVEX_V512, EVEX_CD8<32, CD8VF>;
   defm PD : avx512_fp28_p<opc, OpcodeStr#"pd", v8f64_info, OpNode>,
             avx512_fp28_p_round<opc, OpcodeStr#"pd", v8f64_info, OpNode>,
             T8PD, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VF>;
}

multiclass avx512_fp_unaryop_packed<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode> {
  // Define only if AVX512VL feature is present.
  let Predicates = [HasVLX] in {
    defm PSZ128 : avx512_fp28_p<opc, OpcodeStr#"ps", v4f32x_info, OpNode>,
                                     EVEX_V128, T8PD, EVEX_CD8<32, CD8VF>;
    defm PSZ256 : avx512_fp28_p<opc, OpcodeStr#"ps", v8f32x_info, OpNode>,
                                     EVEX_V256, T8PD, EVEX_CD8<32, CD8VF>;
    defm PDZ128 : avx512_fp28_p<opc, OpcodeStr#"pd", v2f64x_info, OpNode>,
                                     EVEX_V128, VEX_W, T8PD, EVEX_CD8<64, CD8VF>;
    defm PDZ256 : avx512_fp28_p<opc, OpcodeStr#"pd", v4f64x_info, OpNode>,
                                     EVEX_V256, VEX_W, T8PD, EVEX_CD8<64, CD8VF>;
  }
}
let Predicates = [HasERI] in {

 defm VRSQRT28 : avx512_eri<0xCC, "vrsqrt28", X86rsqrt28>, EVEX;
 defm VRCP28   : avx512_eri<0xCA, "vrcp28",   X86rcp28>,   EVEX;
 defm VEXP2    : avx512_eri<0xC8, "vexp2",    X86exp2>,    EVEX;
}
defm VGETEXP   : avx512_eri<0x42, "vgetexp", X86fgetexpRnd>,
                 avx512_fp_unaryop_packed<0x42, "vgetexp", X86fgetexpRnd> , EVEX;

multiclass avx512_sqrt_packed_round<bits<8> opc, string OpcodeStr,
                              SDNode OpNodeRnd, X86VectorVTInfo _>{
  let ExeDomain = _.ExeDomain in
  defm rb: AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _.RC:$src, AVX512RC:$rc), OpcodeStr, "$rc, $src", "$src, $rc",
                         (_.VT (OpNodeRnd _.RC:$src, (i32 imm:$rc)))>,
                         EVEX, EVEX_B, EVEX_RC;
}

multiclass avx512_sqrt_packed<bits<8> opc, string OpcodeStr,
                              SDNode OpNode, X86VectorVTInfo _>{
  let ExeDomain = _.ExeDomain in {
  defm r: AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _.RC:$src), OpcodeStr, "$src", "$src",
                         (_.FloatVT (OpNode _.RC:$src))>, EVEX;
  defm m: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.MemOp:$src), OpcodeStr, "$src", "$src",
                         (OpNode (_.FloatVT
                           (bitconvert (_.LdFrag addr:$src))))>, EVEX;

  defm mb: AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                          (ins _.ScalarMemOp:$src), OpcodeStr,
                          "${src}"##_.BroadcastStr, "${src}"##_.BroadcastStr,
                          (OpNode (_.FloatVT
                            (X86VBroadcast (_.ScalarLdFrag addr:$src))))>,
                          EVEX, EVEX_B;
  }
}

multiclass avx512_sqrt_packed_all<bits<8> opc, string OpcodeStr,
                                  SDNode OpNode> {
  defm PSZ : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "ps"), OpNode,
                                v16f32_info>,
                                EVEX_V512, PS, EVEX_CD8<32, CD8VF>;
  defm PDZ : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "pd"), OpNode,
                                v8f64_info>,
                                EVEX_V512, VEX_W, PD, EVEX_CD8<64, CD8VF>;
  // Define only if AVX512VL feature is present.
  let Predicates = [HasVLX] in {
    defm PSZ128 : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "ps"),
                                     OpNode, v4f32x_info>,
                                     EVEX_V128, PS, EVEX_CD8<32, CD8VF>;
    defm PSZ256 : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "ps"),
                                     OpNode, v8f32x_info>,
                                     EVEX_V256, PS, EVEX_CD8<32, CD8VF>;
    defm PDZ128 : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "pd"),
                                     OpNode, v2f64x_info>,
                                     EVEX_V128, VEX_W, PD, EVEX_CD8<64, CD8VF>;
    defm PDZ256 : avx512_sqrt_packed<opc, !strconcat(OpcodeStr, "pd"),
                                     OpNode, v4f64x_info>,
                                     EVEX_V256, VEX_W, PD, EVEX_CD8<64, CD8VF>;
  }
}

multiclass avx512_sqrt_packed_all_round<bits<8> opc, string OpcodeStr,
                                          SDNode OpNodeRnd> {
  defm PSZ : avx512_sqrt_packed_round<opc, !strconcat(OpcodeStr, "ps"), OpNodeRnd,
                                v16f32_info>, EVEX_V512, PS, EVEX_CD8<32, CD8VF>;
  defm PDZ : avx512_sqrt_packed_round<opc, !strconcat(OpcodeStr, "pd"), OpNodeRnd,
                                v8f64_info>, EVEX_V512, VEX_W, PD, EVEX_CD8<64, CD8VF>;
}

multiclass avx512_sqrt_scalar<bits<8> opc, string OpcodeStr,X86VectorVTInfo _,
                              string SUFF, SDNode OpNode, SDNode OpNodeRnd> {
  let ExeDomain = _.ExeDomain in {
  defm r_Int : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _.RC:$src2), OpcodeStr,
                         "$src2, $src1", "$src1, $src2",
                         (OpNodeRnd (_.VT _.RC:$src1),
                                    (_.VT _.RC:$src2),
                                    (i32 FROUND_CURRENT))>;
  defm m_Int : AVX512_maskable_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                       (ins _.RC:$src1, _.ScalarMemOp:$src2), OpcodeStr,
                       "$src2, $src1", "$src1, $src2",
                       (OpNodeRnd (_.VT _.RC:$src1),
                                  (_.VT (scalar_to_vector
                                            (_.ScalarLdFrag addr:$src2))),
                                  (i32 FROUND_CURRENT))>;

  defm rb_Int : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _.RC:$src2, AVX512RC:$rc), OpcodeStr,
                         "$rc, $src2, $src1", "$src1, $src2, $rc",
                         (OpNodeRnd (_.VT _.RC:$src1),
                                     (_.VT _.RC:$src2),
                                     (i32 imm:$rc))>,
                         EVEX_B, EVEX_RC;

  let isCodeGenOnly = 1, hasSideEffects = 0 in {
    def r : I<opc, MRMSrcReg, (outs _.FRC:$dst),
               (ins _.FRC:$src1, _.FRC:$src2),
               OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>;

    let mayLoad = 1 in
      def m : I<opc, MRMSrcMem, (outs _.FRC:$dst),
                 (ins _.FRC:$src1, _.ScalarMemOp:$src2),
                 OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>;
  }
  }

  def : Pat<(_.EltVT (OpNode _.FRC:$src)),
            (!cast<Instruction>(NAME#SUFF#Zr)
                (_.EltVT (IMPLICIT_DEF)), _.FRC:$src)>;

  def : Pat<(_.EltVT (OpNode (load addr:$src))),
            (!cast<Instruction>(NAME#SUFF#Zm)
                (_.EltVT (IMPLICIT_DEF)), addr:$src)>, Requires<[HasAVX512, OptForSize]>;
}

multiclass avx512_sqrt_scalar_all<bits<8> opc, string OpcodeStr> {
  defm SSZ : avx512_sqrt_scalar<opc, OpcodeStr#"ss", f32x_info, "SS", fsqrt,
                        X86fsqrtRnds>, EVEX_CD8<32, CD8VT1>, EVEX_4V, XS;
  defm SDZ : avx512_sqrt_scalar<opc, OpcodeStr#"sd", f64x_info, "SD", fsqrt,
                        X86fsqrtRnds>, EVEX_CD8<64, CD8VT1>, EVEX_4V, XD, VEX_W;
}

defm VSQRT   : avx512_sqrt_packed_all<0x51, "vsqrt", fsqrt>,
               avx512_sqrt_packed_all_round<0x51, "vsqrt", X86fsqrtRnd>;

defm VSQRT   : avx512_sqrt_scalar_all<0x51, "vsqrt">, VEX_LIG;

let Predicates = [HasAVX512] in {
  def : Pat<(f32 (X86frsqrt FR32X:$src)),
            (COPY_TO_REGCLASS (VRSQRT14SSrr (v4f32 (IMPLICIT_DEF)), (COPY_TO_REGCLASS FR32X:$src, VR128X)), VR128X)>;
  def : Pat<(f32 (X86frsqrt (load addr:$src))),
            (COPY_TO_REGCLASS (VRSQRT14SSrm (v4f32 (IMPLICIT_DEF)), addr:$src), VR128X)>,
            Requires<[OptForSize]>;
  def : Pat<(f32 (X86frcp FR32X:$src)),
            (COPY_TO_REGCLASS (VRCP14SSrr (v4f32 (IMPLICIT_DEF)), (COPY_TO_REGCLASS FR32X:$src, VR128X)), VR128X )>;
  def : Pat<(f32 (X86frcp (load addr:$src))),
            (COPY_TO_REGCLASS (VRCP14SSrm (v4f32 (IMPLICIT_DEF)), addr:$src), VR128X)>,
            Requires<[OptForSize]>;
}

multiclass
avx512_rndscale_scalar<bits<8> opc, string OpcodeStr, X86VectorVTInfo _> {

  let ExeDomain = _.ExeDomain in {
  defm r : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                           (ins _.RC:$src1, _.RC:$src2, i32u8imm:$src3), OpcodeStr,
                           "$src3, $src2, $src1", "$src1, $src2, $src3",
                           (_.VT (X86RndScales (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                            (i32 imm:$src3), (i32 FROUND_CURRENT)))>;

  defm rb : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _.RC:$src2, i32u8imm:$src3), OpcodeStr,
                         "$src3, {sae}, $src2, $src1", "$src1, $src2, {sae}, $src3",
                         (_.VT (X86RndScales (_.VT _.RC:$src1), (_.VT _.RC:$src2),
                         (i32 imm:$src3), (i32 FROUND_NO_EXC)))>, EVEX_B;

  defm m : AVX512_maskable_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                         (ins _.RC:$src1, _.ScalarMemOp:$src2, i32u8imm:$src3),
                         OpcodeStr,
                         "$src3, $src2, $src1", "$src1, $src2, $src3",
                         (_.VT (X86RndScales (_.VT _.RC:$src1),
                          (_.VT (scalar_to_vector (_.ScalarLdFrag addr:$src2))),
                          (i32 imm:$src3), (i32 FROUND_CURRENT)))>;
  }
  let Predicates = [HasAVX512] in {
  def : Pat<(ffloor _.FRC:$src), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##r) (_.VT (IMPLICIT_DEF)),
             (_.VT (COPY_TO_REGCLASS _.FRC:$src, _.RC)), (i32 0x1))), _.FRC)>;
  def : Pat<(fceil _.FRC:$src), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##r) (_.VT (IMPLICIT_DEF)),
             (_.VT (COPY_TO_REGCLASS _.FRC:$src, _.RC)), (i32 0x2))), _.FRC)>;
  def : Pat<(ftrunc _.FRC:$src), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##r) (_.VT (IMPLICIT_DEF)),
             (_.VT (COPY_TO_REGCLASS _.FRC:$src, _.RC)), (i32 0x3))), _.FRC)>;
  def : Pat<(frint _.FRC:$src), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##r) (_.VT (IMPLICIT_DEF)),
             (_.VT (COPY_TO_REGCLASS _.FRC:$src, _.RC)), (i32 0x4))), _.FRC)>;
  def : Pat<(fnearbyint _.FRC:$src), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##r) (_.VT (IMPLICIT_DEF)),
             (_.VT (COPY_TO_REGCLASS _.FRC:$src, _.RC)), (i32 0xc))), _.FRC)>;

  def : Pat<(ffloor (_.ScalarLdFrag addr:$src)), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##m) (_.VT (IMPLICIT_DEF)),
             addr:$src, (i32 0x1))), _.FRC)>;
  def : Pat<(fceil (_.ScalarLdFrag addr:$src)), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##m) (_.VT (IMPLICIT_DEF)),
             addr:$src, (i32 0x2))), _.FRC)>;
  def : Pat<(ftrunc (_.ScalarLdFrag addr:$src)), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##m) (_.VT (IMPLICIT_DEF)),
             addr:$src, (i32 0x3))), _.FRC)>;
  def : Pat<(frint (_.ScalarLdFrag addr:$src)), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##m) (_.VT (IMPLICIT_DEF)),
             addr:$src, (i32 0x4))), _.FRC)>;
  def : Pat<(fnearbyint (_.ScalarLdFrag addr:$src)), (COPY_TO_REGCLASS
             (_.VT (!cast<Instruction>(NAME##m) (_.VT (IMPLICIT_DEF)),
             addr:$src, (i32 0xc))), _.FRC)>;
  }
}

defm VRNDSCALESS : avx512_rndscale_scalar<0x0A, "vrndscaless", f32x_info>,
                                AVX512AIi8Base, EVEX_4V, EVEX_CD8<32, CD8VT1>;

defm VRNDSCALESD : avx512_rndscale_scalar<0x0B, "vrndscalesd", f64x_info>, VEX_W,
                                AVX512AIi8Base, EVEX_4V, EVEX_CD8<64, CD8VT1>;

//-------------------------------------------------
// Integer truncate and extend operations
//-------------------------------------------------

multiclass avx512_trunc_common<bits<8> opc, string OpcodeStr, SDNode OpNode,
                              X86VectorVTInfo SrcInfo, X86VectorVTInfo DestInfo,
                              X86MemOperand x86memop> {
  let ExeDomain = DestInfo.ExeDomain in
  defm rr  : AVX512_maskable<opc, MRMDestReg, DestInfo, (outs DestInfo.RC:$dst),
                      (ins SrcInfo.RC:$src1), OpcodeStr ,"$src1", "$src1",
                      (DestInfo.VT (OpNode (SrcInfo.VT SrcInfo.RC:$src1)))>,
                       EVEX, T8XS;

  // for intrinsic patter match
  def : Pat<(DestInfo.VT (X86select DestInfo.KRCWM:$mask,
                           (DestInfo.VT (OpNode (SrcInfo.VT SrcInfo.RC:$src1))),
                           undef)),
            (!cast<Instruction>(NAME#SrcInfo.ZSuffix##rrkz) DestInfo.KRCWM:$mask ,
                                      SrcInfo.RC:$src1)>;

  def : Pat<(DestInfo.VT (X86select DestInfo.KRCWM:$mask,
                           (DestInfo.VT (OpNode (SrcInfo.VT SrcInfo.RC:$src1))),
                           DestInfo.ImmAllZerosV)),
            (!cast<Instruction>(NAME#SrcInfo.ZSuffix##rrkz) DestInfo.KRCWM:$mask ,
                                      SrcInfo.RC:$src1)>;

  def : Pat<(DestInfo.VT (X86select DestInfo.KRCWM:$mask,
                           (DestInfo.VT (OpNode (SrcInfo.VT SrcInfo.RC:$src1))),
                           DestInfo.RC:$src0)),
            (!cast<Instruction>(NAME#SrcInfo.ZSuffix##rrk) DestInfo.RC:$src0,
                                      DestInfo.KRCWM:$mask ,
                                      SrcInfo.RC:$src1)>;

  let mayStore = 1, mayLoad = 1, hasSideEffects = 0,
      ExeDomain = DestInfo.ExeDomain in {
    def mr : AVX512XS8I<opc, MRMDestMem, (outs),
               (ins x86memop:$dst, SrcInfo.RC:$src),
               OpcodeStr # "\t{$src, $dst|$dst, $src}",
               []>, EVEX;

    def mrk : AVX512XS8I<opc, MRMDestMem, (outs),
               (ins x86memop:$dst, SrcInfo.KRCWM:$mask, SrcInfo.RC:$src),
               OpcodeStr # "\t{$src, $dst {${mask}}|$dst {${mask}}, $src}",
               []>, EVEX, EVEX_K;
  }//mayStore = 1, mayLoad = 1, hasSideEffects = 0
}

multiclass avx512_trunc_mr_lowering<X86VectorVTInfo SrcInfo,
                                    X86VectorVTInfo DestInfo,
                                    PatFrag truncFrag, PatFrag mtruncFrag > {

  def : Pat<(truncFrag (SrcInfo.VT SrcInfo.RC:$src), addr:$dst),
            (!cast<Instruction>(NAME#SrcInfo.ZSuffix##mr)
                                    addr:$dst, SrcInfo.RC:$src)>;

  def : Pat<(mtruncFrag addr:$dst, SrcInfo.KRCWM:$mask,
                                               (SrcInfo.VT SrcInfo.RC:$src)),
            (!cast<Instruction>(NAME#SrcInfo.ZSuffix##mrk)
                            addr:$dst, SrcInfo.KRCWM:$mask, SrcInfo.RC:$src)>;
}

multiclass avx512_trunc<bits<8> opc, string OpcodeStr, SDNode OpNode,
         AVX512VLVectorVTInfo VTSrcInfo, X86VectorVTInfo DestInfoZ128,
         X86VectorVTInfo DestInfoZ256, X86VectorVTInfo DestInfoZ,
         X86MemOperand x86memopZ128, X86MemOperand x86memopZ256,
         X86MemOperand x86memopZ, PatFrag truncFrag, PatFrag mtruncFrag,
                                                     Predicate prd = HasAVX512>{

  let Predicates = [HasVLX, prd] in {
    defm Z128:  avx512_trunc_common<opc, OpcodeStr, OpNode, VTSrcInfo.info128,
                             DestInfoZ128, x86memopZ128>,
                avx512_trunc_mr_lowering<VTSrcInfo.info128, DestInfoZ128,
                             truncFrag, mtruncFrag>, EVEX_V128;

    defm Z256:  avx512_trunc_common<opc, OpcodeStr, OpNode, VTSrcInfo.info256,
                             DestInfoZ256, x86memopZ256>,
                avx512_trunc_mr_lowering<VTSrcInfo.info256, DestInfoZ256,
                             truncFrag, mtruncFrag>, EVEX_V256;
  }
  let Predicates = [prd] in
    defm Z:     avx512_trunc_common<opc, OpcodeStr, OpNode, VTSrcInfo.info512,
                             DestInfoZ, x86memopZ>,
                avx512_trunc_mr_lowering<VTSrcInfo.info512, DestInfoZ,
                             truncFrag, mtruncFrag>, EVEX_V512;
}

multiclass avx512_trunc_qb<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           PatFrag StoreNode, PatFrag MaskedStoreNode> {
  defm NAME: avx512_trunc<opc, OpcodeStr, OpNode, avx512vl_i64_info,
               v16i8x_info, v16i8x_info, v16i8x_info, i16mem, i32mem, i64mem,
               StoreNode, MaskedStoreNode>, EVEX_CD8<8, CD8VO>;
}

multiclass avx512_trunc_qw<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           PatFrag StoreNode, PatFrag MaskedStoreNode> {
  defm NAME: avx512_trunc<opc, OpcodeStr, OpNode, avx512vl_i64_info,
               v8i16x_info, v8i16x_info, v8i16x_info, i32mem, i64mem, i128mem,
               StoreNode, MaskedStoreNode>, EVEX_CD8<16, CD8VQ>;
}

multiclass avx512_trunc_qd<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           PatFrag StoreNode, PatFrag MaskedStoreNode> {
  defm NAME: avx512_trunc<opc, OpcodeStr, OpNode, avx512vl_i64_info,
               v4i32x_info, v4i32x_info, v8i32x_info, i64mem, i128mem, i256mem,
               StoreNode, MaskedStoreNode>, EVEX_CD8<32, CD8VH>;
}

multiclass avx512_trunc_db<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           PatFrag StoreNode, PatFrag MaskedStoreNode> {
  defm NAME: avx512_trunc<opc, OpcodeStr, OpNode, avx512vl_i32_info,
               v16i8x_info, v16i8x_info, v16i8x_info, i32mem, i64mem, i128mem,
               StoreNode, MaskedStoreNode>, EVEX_CD8<8, CD8VQ>;
}

multiclass avx512_trunc_dw<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           PatFrag StoreNode, PatFrag MaskedStoreNode> {
  defm NAME: avx512_trunc<opc, OpcodeStr, OpNode, avx512vl_i32_info,
              v8i16x_info, v8i16x_info, v16i16x_info, i64mem, i128mem, i256mem,
              StoreNode, MaskedStoreNode>, EVEX_CD8<16, CD8VH>;
}

multiclass avx512_trunc_wb<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           PatFrag StoreNode, PatFrag MaskedStoreNode> {
  defm NAME: avx512_trunc<opc, OpcodeStr, OpNode, avx512vl_i16_info,
              v16i8x_info, v16i8x_info, v32i8x_info, i64mem, i128mem, i256mem,
              StoreNode, MaskedStoreNode, HasBWI>, EVEX_CD8<16, CD8VH>;
}

defm VPMOVQB    : avx512_trunc_qb<0x32, "vpmovqb",   X86vtrunc,
                                  truncstorevi8, masked_truncstorevi8>;
defm VPMOVSQB   : avx512_trunc_qb<0x22, "vpmovsqb",  X86vtruncs,
                                  truncstore_s_vi8, masked_truncstore_s_vi8>;
defm VPMOVUSQB  : avx512_trunc_qb<0x12, "vpmovusqb", X86vtruncus,
                                  truncstore_us_vi8, masked_truncstore_us_vi8>;

defm VPMOVQW    : avx512_trunc_qw<0x34, "vpmovqw",   X86vtrunc,
                                  truncstorevi16, masked_truncstorevi16>;
defm VPMOVSQW   : avx512_trunc_qw<0x24, "vpmovsqw",  X86vtruncs,
                                  truncstore_s_vi16, masked_truncstore_s_vi16>;
defm VPMOVUSQW  : avx512_trunc_qw<0x14, "vpmovusqw", X86vtruncus,
                                  truncstore_us_vi16, masked_truncstore_us_vi16>;

defm VPMOVQD    : avx512_trunc_qd<0x35, "vpmovqd",   X86vtrunc,
                                  truncstorevi32, masked_truncstorevi32>;
defm VPMOVSQD   : avx512_trunc_qd<0x25, "vpmovsqd",  X86vtruncs,
                                  truncstore_s_vi32, masked_truncstore_s_vi32>;
defm VPMOVUSQD  : avx512_trunc_qd<0x15, "vpmovusqd", X86vtruncus,
                                  truncstore_us_vi32, masked_truncstore_us_vi32>;

defm VPMOVDB    : avx512_trunc_db<0x31, "vpmovdb", X86vtrunc,
                                  truncstorevi8, masked_truncstorevi8>;
defm VPMOVSDB   : avx512_trunc_db<0x21, "vpmovsdb",   X86vtruncs,
                                  truncstore_s_vi8, masked_truncstore_s_vi8>;
defm VPMOVUSDB  : avx512_trunc_db<0x11, "vpmovusdb",  X86vtruncus,
                                  truncstore_us_vi8, masked_truncstore_us_vi8>;

defm VPMOVDW    : avx512_trunc_dw<0x33, "vpmovdw", X86vtrunc,
                                  truncstorevi16, masked_truncstorevi16>;
defm VPMOVSDW   : avx512_trunc_dw<0x23, "vpmovsdw",   X86vtruncs,
                                  truncstore_s_vi16, masked_truncstore_s_vi16>;
defm VPMOVUSDW  : avx512_trunc_dw<0x13, "vpmovusdw",  X86vtruncus,
                                  truncstore_us_vi16, masked_truncstore_us_vi16>;

defm VPMOVWB    : avx512_trunc_wb<0x30, "vpmovwb", X86vtrunc,
                                  truncstorevi8, masked_truncstorevi8>;
defm VPMOVSWB   : avx512_trunc_wb<0x20, "vpmovswb",   X86vtruncs,
                                  truncstore_s_vi8, masked_truncstore_s_vi8>;
defm VPMOVUSWB  : avx512_trunc_wb<0x10, "vpmovuswb",  X86vtruncus,
                                  truncstore_us_vi8, masked_truncstore_us_vi8>;

let Predicates = [HasAVX512, NoVLX] in {
def: Pat<(v8i16 (X86vtrunc (v8i32 VR256X:$src))),
         (v8i16 (EXTRACT_SUBREG
                 (v16i16 (VPMOVDWZrr (v16i32 (INSERT_SUBREG (IMPLICIT_DEF),
                                          VR256X:$src, sub_ymm)))), sub_xmm))>;
def: Pat<(v4i32 (X86vtrunc (v4i64 VR256X:$src))),
         (v4i32 (EXTRACT_SUBREG
                 (v8i32 (VPMOVQDZrr (v8i64 (INSERT_SUBREG (IMPLICIT_DEF),
                                           VR256X:$src, sub_ymm)))), sub_xmm))>;
}

let Predicates = [HasBWI, NoVLX] in {
def: Pat<(v16i8 (X86vtrunc (v16i16 VR256X:$src))),
         (v16i8 (EXTRACT_SUBREG (VPMOVWBZrr (v32i16 (INSERT_SUBREG (IMPLICIT_DEF),
                                            VR256X:$src, sub_ymm))), sub_xmm))>;
}

multiclass avx512_extend_common<bits<8> opc, string OpcodeStr,
              X86VectorVTInfo DestInfo, X86VectorVTInfo SrcInfo,
              X86MemOperand x86memop, PatFrag LdFrag, SDPatternOperator OpNode>{
  let ExeDomain = DestInfo.ExeDomain in {
  defm rr   : AVX512_maskable<opc, MRMSrcReg, DestInfo, (outs DestInfo.RC:$dst),
                    (ins SrcInfo.RC:$src), OpcodeStr ,"$src", "$src",
                    (DestInfo.VT (OpNode (SrcInfo.VT SrcInfo.RC:$src)))>,
                  EVEX;

  defm rm : AVX512_maskable<opc, MRMSrcMem, DestInfo, (outs DestInfo.RC:$dst),
                  (ins x86memop:$src), OpcodeStr ,"$src", "$src",
                  (DestInfo.VT (LdFrag addr:$src))>,
                EVEX;
  }
}

multiclass avx512_extend_BW<bits<8> opc, string OpcodeStr,
          SDPatternOperator OpNode, SDPatternOperator InVecNode,
          string ExtTy,PatFrag LdFrag = !cast<PatFrag>(ExtTy#"extloadvi8")> {
  let Predicates = [HasVLX, HasBWI] in {
    defm Z128:  avx512_extend_common<opc, OpcodeStr, v8i16x_info,
                    v16i8x_info, i64mem, LdFrag, InVecNode>,
                     EVEX_CD8<8, CD8VH>, T8PD, EVEX_V128;

    defm Z256:  avx512_extend_common<opc, OpcodeStr, v16i16x_info,
                    v16i8x_info, i128mem, LdFrag, OpNode>,
                     EVEX_CD8<8, CD8VH>, T8PD, EVEX_V256;
  }
  let Predicates = [HasBWI] in {
    defm Z   :  avx512_extend_common<opc, OpcodeStr, v32i16_info,
                    v32i8x_info, i256mem, LdFrag, OpNode>,
                     EVEX_CD8<8, CD8VH>, T8PD, EVEX_V512;
  }
}

multiclass avx512_extend_BD<bits<8> opc, string OpcodeStr,
          SDPatternOperator OpNode, SDPatternOperator InVecNode,
          string ExtTy,PatFrag LdFrag = !cast<PatFrag>(ExtTy#"extloadvi8")> {
  let Predicates = [HasVLX, HasAVX512] in {
    defm Z128:  avx512_extend_common<opc, OpcodeStr, v4i32x_info,
                   v16i8x_info, i32mem, LdFrag, InVecNode>,
                         EVEX_CD8<8, CD8VQ>, T8PD, EVEX_V128;

    defm Z256:  avx512_extend_common<opc, OpcodeStr, v8i32x_info,
                   v16i8x_info, i64mem, LdFrag, OpNode>,
                         EVEX_CD8<8, CD8VQ>, T8PD, EVEX_V256;
  }
  let Predicates = [HasAVX512] in {
    defm Z   :  avx512_extend_common<opc, OpcodeStr, v16i32_info,
                   v16i8x_info, i128mem, LdFrag, OpNode>,
                         EVEX_CD8<8, CD8VQ>, T8PD, EVEX_V512;
  }
}

multiclass avx512_extend_BQ<bits<8> opc, string OpcodeStr,
          SDPatternOperator OpNode, SDPatternOperator InVecNode,
          string ExtTy,PatFrag LdFrag = !cast<PatFrag>(ExtTy#"extloadvi8")> {
  let Predicates = [HasVLX, HasAVX512] in {
    defm Z128:  avx512_extend_common<opc, OpcodeStr, v2i64x_info,
                   v16i8x_info, i16mem, LdFrag, InVecNode>,
                     EVEX_CD8<8, CD8VO>, T8PD, EVEX_V128;

    defm Z256:  avx512_extend_common<opc, OpcodeStr, v4i64x_info,
                   v16i8x_info, i32mem, LdFrag, OpNode>,
                     EVEX_CD8<8, CD8VO>, T8PD, EVEX_V256;
  }
  let Predicates = [HasAVX512] in {
    defm Z   :  avx512_extend_common<opc, OpcodeStr, v8i64_info,
                   v16i8x_info, i64mem, LdFrag, OpNode>,
                     EVEX_CD8<8, CD8VO>, T8PD, EVEX_V512;
  }
}

multiclass avx512_extend_WD<bits<8> opc, string OpcodeStr,
         SDPatternOperator OpNode, SDPatternOperator InVecNode,
         string ExtTy,PatFrag LdFrag = !cast<PatFrag>(ExtTy#"extloadvi16")> {
  let Predicates = [HasVLX, HasAVX512] in {
    defm Z128:  avx512_extend_common<opc, OpcodeStr, v4i32x_info,
                   v8i16x_info, i64mem, LdFrag, InVecNode>,
                     EVEX_CD8<16, CD8VH>, T8PD, EVEX_V128;

    defm Z256:  avx512_extend_common<opc, OpcodeStr, v8i32x_info,
                   v8i16x_info, i128mem, LdFrag, OpNode>,
                     EVEX_CD8<16, CD8VH>, T8PD, EVEX_V256;
  }
  let Predicates = [HasAVX512] in {
    defm Z   :  avx512_extend_common<opc, OpcodeStr, v16i32_info,
                   v16i16x_info, i256mem, LdFrag, OpNode>,
                     EVEX_CD8<16, CD8VH>, T8PD, EVEX_V512;
  }
}

multiclass avx512_extend_WQ<bits<8> opc, string OpcodeStr,
         SDPatternOperator OpNode, SDPatternOperator InVecNode,
         string ExtTy,PatFrag LdFrag = !cast<PatFrag>(ExtTy#"extloadvi16")> {
  let Predicates = [HasVLX, HasAVX512] in {
    defm Z128:  avx512_extend_common<opc, OpcodeStr, v2i64x_info,
                   v8i16x_info, i32mem, LdFrag, InVecNode>,
                     EVEX_CD8<16, CD8VQ>, T8PD, EVEX_V128;

    defm Z256:  avx512_extend_common<opc, OpcodeStr, v4i64x_info,
                   v8i16x_info, i64mem, LdFrag, OpNode>,
                     EVEX_CD8<16, CD8VQ>, T8PD, EVEX_V256;
  }
  let Predicates = [HasAVX512] in {
    defm Z   :  avx512_extend_common<opc, OpcodeStr, v8i64_info,
                   v8i16x_info, i128mem, LdFrag, OpNode>,
                     EVEX_CD8<16, CD8VQ>, T8PD, EVEX_V512;
  }
}

multiclass avx512_extend_DQ<bits<8> opc, string OpcodeStr,
         SDPatternOperator OpNode, SDPatternOperator InVecNode,
         string ExtTy,PatFrag LdFrag = !cast<PatFrag>(ExtTy#"extloadvi32")> {

  let Predicates = [HasVLX, HasAVX512] in {
    defm Z128:  avx512_extend_common<opc, OpcodeStr, v2i64x_info,
                   v4i32x_info, i64mem, LdFrag, InVecNode>,
                     EVEX_CD8<32, CD8VH>, T8PD, EVEX_V128;

    defm Z256:  avx512_extend_common<opc, OpcodeStr, v4i64x_info,
                   v4i32x_info, i128mem, LdFrag, OpNode>,
                     EVEX_CD8<32, CD8VH>, T8PD, EVEX_V256;
  }
  let Predicates = [HasAVX512] in {
    defm Z   :  avx512_extend_common<opc, OpcodeStr, v8i64_info,
                   v8i32x_info, i256mem, LdFrag, OpNode>,
                     EVEX_CD8<32, CD8VH>, T8PD, EVEX_V512;
  }
}

defm VPMOVZXBW : avx512_extend_BW<0x30, "vpmovzxbw", X86vzext, zext_invec, "z">;
defm VPMOVZXBD : avx512_extend_BD<0x31, "vpmovzxbd", X86vzext, zext_invec, "z">;
defm VPMOVZXBQ : avx512_extend_BQ<0x32, "vpmovzxbq", X86vzext, zext_invec, "z">;
defm VPMOVZXWD : avx512_extend_WD<0x33, "vpmovzxwd", X86vzext, zext_invec, "z">;
defm VPMOVZXWQ : avx512_extend_WQ<0x34, "vpmovzxwq", X86vzext, zext_invec, "z">;
defm VPMOVZXDQ : avx512_extend_DQ<0x35, "vpmovzxdq", X86vzext, zext_invec, "z">;

defm VPMOVSXBW: avx512_extend_BW<0x20, "vpmovsxbw", X86vsext, sext_invec, "s">;
defm VPMOVSXBD: avx512_extend_BD<0x21, "vpmovsxbd", X86vsext, sext_invec, "s">;
defm VPMOVSXBQ: avx512_extend_BQ<0x22, "vpmovsxbq", X86vsext, sext_invec, "s">;
defm VPMOVSXWD: avx512_extend_WD<0x23, "vpmovsxwd", X86vsext, sext_invec, "s">;
defm VPMOVSXWQ: avx512_extend_WQ<0x24, "vpmovsxwq", X86vsext, sext_invec, "s">;
defm VPMOVSXDQ: avx512_extend_DQ<0x25, "vpmovsxdq", X86vsext, sext_invec, "s">;

// EXTLOAD patterns, implemented using vpmovz
multiclass avx512_ext_lowering<string InstrStr, X86VectorVTInfo To,
                               X86VectorVTInfo From, PatFrag LdFrag> {
  def : Pat<(To.VT (LdFrag addr:$src)),
            (!cast<Instruction>("VPMOVZX"#InstrStr#"rm") addr:$src)>;
  def : Pat<(To.VT (vselect To.KRCWM:$mask, (LdFrag addr:$src), To.RC:$src0)),
            (!cast<Instruction>("VPMOVZX"#InstrStr#"rmk") To.RC:$src0,
             To.KRC:$mask, addr:$src)>;
  def : Pat<(To.VT (vselect To.KRCWM:$mask, (LdFrag addr:$src),
                    To.ImmAllZerosV)),
            (!cast<Instruction>("VPMOVZX"#InstrStr#"rmkz") To.KRC:$mask,
             addr:$src)>;
}

let Predicates = [HasVLX, HasBWI] in {
  defm : avx512_ext_lowering<"BWZ128", v8i16x_info,  v16i8x_info,  extloadvi8>;
  defm : avx512_ext_lowering<"BWZ256", v16i16x_info, v16i8x_info,  extloadvi8>;
}
let Predicates = [HasBWI] in {
  defm : avx512_ext_lowering<"BWZ",    v32i16_info,  v32i8x_info,  extloadvi8>;
}
let Predicates = [HasVLX, HasAVX512] in {
  defm : avx512_ext_lowering<"BDZ128", v4i32x_info,  v16i8x_info,  extloadvi8>;
  defm : avx512_ext_lowering<"BDZ256", v8i32x_info,  v16i8x_info,  extloadvi8>;
  defm : avx512_ext_lowering<"BQZ128", v2i64x_info,  v16i8x_info,  extloadvi8>;
  defm : avx512_ext_lowering<"BQZ256", v4i64x_info,  v16i8x_info,  extloadvi8>;
  defm : avx512_ext_lowering<"WDZ128", v4i32x_info,  v8i16x_info,  extloadvi16>;
  defm : avx512_ext_lowering<"WDZ256", v8i32x_info,  v8i16x_info,  extloadvi16>;
  defm : avx512_ext_lowering<"WQZ128", v2i64x_info,  v8i16x_info,  extloadvi16>;
  defm : avx512_ext_lowering<"WQZ256", v4i64x_info,  v8i16x_info,  extloadvi16>;
  defm : avx512_ext_lowering<"DQZ128", v2i64x_info,  v4i32x_info,  extloadvi32>;
  defm : avx512_ext_lowering<"DQZ256", v4i64x_info,  v4i32x_info,  extloadvi32>;
}
let Predicates = [HasAVX512] in {
  defm : avx512_ext_lowering<"BDZ",    v16i32_info,  v16i8x_info,  extloadvi8>;
  defm : avx512_ext_lowering<"BQZ",    v8i64_info,   v16i8x_info,  extloadvi8>;
  defm : avx512_ext_lowering<"WDZ",    v16i32_info,  v16i16x_info, extloadvi16>;
  defm : avx512_ext_lowering<"WQZ",    v8i64_info,   v8i16x_info,  extloadvi16>;
  defm : avx512_ext_lowering<"DQZ",    v8i64_info,   v8i32x_info,  extloadvi32>;
}

multiclass AVX512_pmovx_patterns<string OpcPrefix, SDNode ExtOp,
                                 SDNode InVecOp, PatFrag ExtLoad16> {
  // 128-bit patterns
  let Predicates = [HasVLX, HasBWI] in {
  def : Pat<(v8i16 (InVecOp (bc_v16i8 (v2i64 (scalar_to_vector (loadi64 addr:$src)))))),
            (!cast<I>(OpcPrefix#BWZ128rm) addr:$src)>;
  def : Pat<(v8i16 (InVecOp (bc_v16i8 (v2f64 (scalar_to_vector (loadf64 addr:$src)))))),
            (!cast<I>(OpcPrefix#BWZ128rm) addr:$src)>;
  def : Pat<(v8i16 (InVecOp (v16i8 (vzmovl_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BWZ128rm) addr:$src)>;
  def : Pat<(v8i16 (InVecOp (v16i8 (vzload_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BWZ128rm) addr:$src)>;
  def : Pat<(v8i16 (InVecOp (bc_v16i8 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BWZ128rm) addr:$src)>;
  }
  let Predicates = [HasVLX] in {
  def : Pat<(v4i32 (InVecOp (bc_v16i8 (v4i32 (scalar_to_vector (loadi32 addr:$src)))))),
            (!cast<I>(OpcPrefix#BDZ128rm) addr:$src)>;
  def : Pat<(v4i32 (InVecOp (v16i8 (vzmovl_v4i32 addr:$src)))),
            (!cast<I>(OpcPrefix#BDZ128rm) addr:$src)>;
  def : Pat<(v4i32 (InVecOp (v16i8 (vzload_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BDZ128rm) addr:$src)>;
  def : Pat<(v4i32 (InVecOp (bc_v16i8 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BDZ128rm) addr:$src)>;

  def : Pat<(v2i64 (InVecOp (bc_v16i8 (v4i32 (scalar_to_vector (ExtLoad16 addr:$src)))))),
            (!cast<I>(OpcPrefix#BQZ128rm) addr:$src)>;
  def : Pat<(v2i64 (InVecOp (v16i8 (vzmovl_v4i32 addr:$src)))),
            (!cast<I>(OpcPrefix#BQZ128rm) addr:$src)>;
  def : Pat<(v2i64 (InVecOp (v16i8 (vzload_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BQZ128rm) addr:$src)>;
  def : Pat<(v2i64 (InVecOp (bc_v16i8 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BQZ128rm) addr:$src)>;

  def : Pat<(v4i32 (InVecOp (bc_v8i16 (v2i64 (scalar_to_vector (loadi64 addr:$src)))))),
            (!cast<I>(OpcPrefix#WDZ128rm) addr:$src)>;
  def : Pat<(v4i32 (InVecOp (bc_v8i16 (v2f64 (scalar_to_vector (loadf64 addr:$src)))))),
            (!cast<I>(OpcPrefix#WDZ128rm) addr:$src)>;
  def : Pat<(v4i32 (InVecOp (v8i16 (vzmovl_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WDZ128rm) addr:$src)>;
  def : Pat<(v4i32 (InVecOp (v8i16 (vzload_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WDZ128rm) addr:$src)>;
  def : Pat<(v4i32 (InVecOp (bc_v8i16 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WDZ128rm) addr:$src)>;

  def : Pat<(v2i64 (InVecOp (bc_v8i16 (v4i32 (scalar_to_vector (loadi32 addr:$src)))))),
            (!cast<I>(OpcPrefix#WQZ128rm) addr:$src)>;
  def : Pat<(v2i64 (InVecOp (v8i16 (vzmovl_v4i32 addr:$src)))),
            (!cast<I>(OpcPrefix#WQZ128rm) addr:$src)>;
  def : Pat<(v2i64 (InVecOp (v8i16 (vzload_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WQZ128rm) addr:$src)>;
  def : Pat<(v2i64 (InVecOp (bc_v8i16 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WQZ128rm) addr:$src)>;

  def : Pat<(v2i64 (InVecOp (bc_v4i32 (v2i64 (scalar_to_vector (loadi64 addr:$src)))))),
            (!cast<I>(OpcPrefix#DQZ128rm) addr:$src)>;
  def : Pat<(v2i64 (InVecOp (bc_v4i32 (v2f64 (scalar_to_vector (loadf64 addr:$src)))))),
            (!cast<I>(OpcPrefix#DQZ128rm) addr:$src)>;
  def : Pat<(v2i64 (InVecOp (v4i32 (vzmovl_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#DQZ128rm) addr:$src)>;
  def : Pat<(v2i64 (InVecOp (v4i32 (vzload_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#DQZ128rm) addr:$src)>;
  def : Pat<(v2i64 (InVecOp (bc_v4i32 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#DQZ128rm) addr:$src)>;
  }
  // 256-bit patterns
  let Predicates = [HasVLX, HasBWI] in {
  def : Pat<(v16i16 (ExtOp (bc_v16i8 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BWZ256rm) addr:$src)>;
  def : Pat<(v16i16 (ExtOp (v16i8 (vzmovl_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BWZ256rm) addr:$src)>;
  def : Pat<(v16i16 (ExtOp (v16i8 (vzload_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BWZ256rm) addr:$src)>;
  }
  let Predicates = [HasVLX] in {
  def : Pat<(v8i32 (ExtOp (bc_v16i8 (v2i64 (scalar_to_vector (loadi64 addr:$src)))))),
            (!cast<I>(OpcPrefix#BDZ256rm) addr:$src)>;
  def : Pat<(v8i32 (ExtOp (v16i8 (vzmovl_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BDZ256rm) addr:$src)>;
  def : Pat<(v8i32 (ExtOp (v16i8 (vzload_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BDZ256rm) addr:$src)>;
  def : Pat<(v8i32 (ExtOp (bc_v16i8 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BDZ256rm) addr:$src)>;

  def : Pat<(v4i64 (ExtOp (bc_v16i8 (v4i32 (scalar_to_vector (loadi32 addr:$src)))))),
            (!cast<I>(OpcPrefix#BQZ256rm) addr:$src)>;
  def : Pat<(v4i64 (ExtOp (v16i8 (vzmovl_v4i32 addr:$src)))),
            (!cast<I>(OpcPrefix#BQZ256rm) addr:$src)>;
  def : Pat<(v4i64 (ExtOp (v16i8 (vzload_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BQZ256rm) addr:$src)>;
  def : Pat<(v4i64 (ExtOp (bc_v16i8 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BQZ256rm) addr:$src)>;

  def : Pat<(v8i32 (ExtOp (bc_v8i16 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WDZ256rm) addr:$src)>;
  def : Pat<(v8i32 (ExtOp (v8i16 (vzmovl_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WDZ256rm) addr:$src)>;
  def : Pat<(v8i32 (ExtOp (v8i16 (vzload_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WDZ256rm) addr:$src)>;

  def : Pat<(v4i64 (ExtOp (bc_v8i16 (v2i64 (scalar_to_vector (loadi64 addr:$src)))))),
            (!cast<I>(OpcPrefix#WQZ256rm) addr:$src)>;
  def : Pat<(v4i64 (ExtOp (v8i16 (vzmovl_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WQZ256rm) addr:$src)>;
  def : Pat<(v4i64 (ExtOp (v8i16 (vzload_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WQZ256rm) addr:$src)>;
  def : Pat<(v4i64 (ExtOp (bc_v8i16 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WQZ256rm) addr:$src)>;

  def : Pat<(v4i64 (ExtOp (bc_v4i32 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#DQZ256rm) addr:$src)>;
  def : Pat<(v4i64 (ExtOp (v4i32 (vzmovl_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#DQZ256rm) addr:$src)>;
  def : Pat<(v4i64 (ExtOp (v4i32 (vzload_v2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#DQZ256rm) addr:$src)>;
  }
  // 512-bit patterns
  let Predicates = [HasBWI] in {
  def : Pat<(v32i16 (ExtOp (bc_v32i8 (loadv4i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BWZrm) addr:$src)>;
  }
  let Predicates = [HasAVX512] in {
  def : Pat<(v16i32 (ExtOp (bc_v16i8 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BDZrm) addr:$src)>;

  def : Pat<(v8i64 (ExtOp (bc_v16i8 (v2i64 (scalar_to_vector (loadi64 addr:$src)))))),
            (!cast<I>(OpcPrefix#BQZrm) addr:$src)>;
  def : Pat<(v8i64 (ExtOp (bc_v16i8 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#BQZrm) addr:$src)>;

  def : Pat<(v16i32 (ExtOp (bc_v16i16 (loadv4i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WDZrm) addr:$src)>;

  def : Pat<(v8i64 (ExtOp (bc_v8i16 (loadv2i64 addr:$src)))),
            (!cast<I>(OpcPrefix#WQZrm) addr:$src)>;

  def : Pat<(v8i64 (ExtOp (bc_v8i32 (loadv4i64 addr:$src)))),
            (!cast<I>(OpcPrefix#DQZrm) addr:$src)>;
  }
}

defm : AVX512_pmovx_patterns<"VPMOVSX", X86vsext, sext_invec, extloadi32i16>;
defm : AVX512_pmovx_patterns<"VPMOVZX", X86vzext, zext_invec, loadi16_anyext>;

//===----------------------------------------------------------------------===//
// GATHER - SCATTER Operations

multiclass avx512_gather<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                         X86MemOperand memop, PatFrag GatherNode> {
  let Constraints = "@earlyclobber $dst, $src1 = $dst, $mask = $mask_wb",
      ExeDomain = _.ExeDomain in
  def rm  : AVX5128I<opc, MRMSrcMem, (outs _.RC:$dst, _.KRCWM:$mask_wb),
            (ins _.RC:$src1, _.KRCWM:$mask, memop:$src2),
            !strconcat(OpcodeStr#_.Suffix,
            "\t{$src2, ${dst} {${mask}}|${dst} {${mask}}, $src2}"),
            [(set _.RC:$dst, _.KRCWM:$mask_wb,
              (GatherNode  (_.VT _.RC:$src1), _.KRCWM:$mask,
                     vectoraddr:$src2))]>, EVEX, EVEX_K,
             EVEX_CD8<_.EltSize, CD8VT1>;
}

multiclass avx512_gather_q_pd<bits<8> dopc, bits<8> qopc,
                        AVX512VLVectorVTInfo _, string OpcodeStr, string SUFF> {
  defm NAME##D##SUFF##Z: avx512_gather<dopc, OpcodeStr##"d", _.info512,
                                      vy512mem, mgatherv8i32>, EVEX_V512, VEX_W;
  defm NAME##Q##SUFF##Z: avx512_gather<qopc, OpcodeStr##"q", _.info512,
                                      vz512mem,  mgatherv8i64>, EVEX_V512, VEX_W;
let Predicates = [HasVLX] in {
  defm NAME##D##SUFF##Z256: avx512_gather<dopc, OpcodeStr##"d", _.info256,
                              vx256xmem, mgatherv4i32>, EVEX_V256, VEX_W;
  defm NAME##Q##SUFF##Z256: avx512_gather<qopc, OpcodeStr##"q", _.info256,
                              vy256xmem, mgatherv4i64>, EVEX_V256, VEX_W;
  defm NAME##D##SUFF##Z128: avx512_gather<dopc, OpcodeStr##"d", _.info128,
                              vx128xmem, mgatherv4i32>, EVEX_V128, VEX_W;
  defm NAME##Q##SUFF##Z128: avx512_gather<qopc, OpcodeStr##"q", _.info128,
                              vx128xmem, mgatherv2i64>, EVEX_V128, VEX_W;
}
}

multiclass avx512_gather_d_ps<bits<8> dopc, bits<8> qopc,
                       AVX512VLVectorVTInfo _, string OpcodeStr, string SUFF> {
  defm NAME##D##SUFF##Z: avx512_gather<dopc, OpcodeStr##"d", _.info512, vz512mem,
                                       mgatherv16i32>, EVEX_V512;
  defm NAME##Q##SUFF##Z: avx512_gather<qopc, OpcodeStr##"q", _.info256, vz256xmem,
                                       mgatherv8i64>, EVEX_V512;
let Predicates = [HasVLX] in {
  defm NAME##D##SUFF##Z256: avx512_gather<dopc, OpcodeStr##"d", _.info256,
                                          vy256xmem, mgatherv8i32>, EVEX_V256;
  defm NAME##Q##SUFF##Z256: avx512_gather<qopc, OpcodeStr##"q", _.info128,
                                          vy128xmem, mgatherv4i64>, EVEX_V256;
  defm NAME##D##SUFF##Z128: avx512_gather<dopc, OpcodeStr##"d", _.info128,
                                          vx128xmem, mgatherv4i32>, EVEX_V128;
  defm NAME##Q##SUFF##Z128: avx512_gather<qopc, OpcodeStr##"q", _.info128,
                                          vx64xmem, mgatherv2i64>, EVEX_V128;
}
}


defm VGATHER : avx512_gather_q_pd<0x92, 0x93, avx512vl_f64_info, "vgather", "PD">,
               avx512_gather_d_ps<0x92, 0x93, avx512vl_f32_info, "vgather", "PS">;

defm VPGATHER : avx512_gather_q_pd<0x90, 0x91, avx512vl_i64_info, "vpgather", "Q">,
                avx512_gather_d_ps<0x90, 0x91, avx512vl_i32_info, "vpgather", "D">;

multiclass avx512_scatter<bits<8> opc, string OpcodeStr, X86VectorVTInfo _,
                          X86MemOperand memop, PatFrag ScatterNode> {

let mayStore = 1, Constraints = "$mask = $mask_wb", ExeDomain = _.ExeDomain in

  def mr  : AVX5128I<opc, MRMDestMem, (outs _.KRCWM:$mask_wb),
            (ins memop:$dst, _.KRCWM:$mask, _.RC:$src),
            !strconcat(OpcodeStr#_.Suffix,
            "\t{$src, ${dst} {${mask}}|${dst} {${mask}}, $src}"),
            [(set _.KRCWM:$mask_wb, (ScatterNode (_.VT _.RC:$src),
                                     _.KRCWM:$mask,  vectoraddr:$dst))]>,
            EVEX, EVEX_K, EVEX_CD8<_.EltSize, CD8VT1>;
}

multiclass avx512_scatter_q_pd<bits<8> dopc, bits<8> qopc,
                        AVX512VLVectorVTInfo _, string OpcodeStr, string SUFF> {
  defm NAME##D##SUFF##Z: avx512_scatter<dopc, OpcodeStr##"d", _.info512,
                                      vy512mem, mscatterv8i32>, EVEX_V512, VEX_W;
  defm NAME##Q##SUFF##Z: avx512_scatter<qopc, OpcodeStr##"q", _.info512,
                                      vz512mem,  mscatterv8i64>, EVEX_V512, VEX_W;
let Predicates = [HasVLX] in {
  defm NAME##D##SUFF##Z256: avx512_scatter<dopc, OpcodeStr##"d", _.info256,
                              vx256xmem, mscatterv4i32>, EVEX_V256, VEX_W;
  defm NAME##Q##SUFF##Z256: avx512_scatter<qopc, OpcodeStr##"q", _.info256,
                              vy256xmem, mscatterv4i64>, EVEX_V256, VEX_W;
  defm NAME##D##SUFF##Z128: avx512_scatter<dopc, OpcodeStr##"d", _.info128,
                              vx128xmem, mscatterv4i32>, EVEX_V128, VEX_W;
  defm NAME##Q##SUFF##Z128: avx512_scatter<qopc, OpcodeStr##"q", _.info128,
                              vx128xmem, mscatterv2i64>, EVEX_V128, VEX_W;
}
}

multiclass avx512_scatter_d_ps<bits<8> dopc, bits<8> qopc,
                       AVX512VLVectorVTInfo _, string OpcodeStr, string SUFF> {
  defm NAME##D##SUFF##Z: avx512_scatter<dopc, OpcodeStr##"d", _.info512, vz512mem,
                                       mscatterv16i32>, EVEX_V512;
  defm NAME##Q##SUFF##Z: avx512_scatter<qopc, OpcodeStr##"q", _.info256, vz256xmem,
                                       mscatterv8i64>, EVEX_V512;
let Predicates = [HasVLX] in {
  defm NAME##D##SUFF##Z256: avx512_scatter<dopc, OpcodeStr##"d", _.info256,
                                          vy256xmem, mscatterv8i32>, EVEX_V256;
  defm NAME##Q##SUFF##Z256: avx512_scatter<qopc, OpcodeStr##"q", _.info128,
                                          vy128xmem, mscatterv4i64>, EVEX_V256;
  defm NAME##D##SUFF##Z128: avx512_scatter<dopc, OpcodeStr##"d", _.info128,
                                          vx128xmem, mscatterv4i32>, EVEX_V128;
  defm NAME##Q##SUFF##Z128: avx512_scatter<qopc, OpcodeStr##"q", _.info128,
                                          vx64xmem, mscatterv2i64>, EVEX_V128;
}
}

defm VSCATTER : avx512_scatter_q_pd<0xA2, 0xA3, avx512vl_f64_info, "vscatter", "PD">,
               avx512_scatter_d_ps<0xA2, 0xA3, avx512vl_f32_info, "vscatter", "PS">;

defm VPSCATTER : avx512_scatter_q_pd<0xA0, 0xA1, avx512vl_i64_info, "vpscatter", "Q">,
                avx512_scatter_d_ps<0xA0, 0xA1, avx512vl_i32_info, "vpscatter", "D">;

// prefetch
multiclass avx512_gather_scatter_prefetch<bits<8> opc, Format F, string OpcodeStr,
                       RegisterClass KRC, X86MemOperand memop> {
  let Predicates = [HasPFI], hasSideEffects = 1 in
  def m  : AVX5128I<opc, F, (outs), (ins KRC:$mask, memop:$src),
            !strconcat(OpcodeStr, "\t{$src {${mask}}|{${mask}}, $src}"),
            []>, EVEX, EVEX_K;
}

defm VGATHERPF0DPS: avx512_gather_scatter_prefetch<0xC6, MRM1m, "vgatherpf0dps",
                     VK16WM, vz512mem>, EVEX_V512, EVEX_CD8<32, CD8VT1>;

defm VGATHERPF0QPS: avx512_gather_scatter_prefetch<0xC7, MRM1m, "vgatherpf0qps",
                     VK8WM, vz256xmem>, EVEX_V512, EVEX_CD8<64, CD8VT1>;

defm VGATHERPF0DPD: avx512_gather_scatter_prefetch<0xC6, MRM1m, "vgatherpf0dpd",
                     VK8WM, vy512mem>, EVEX_V512, VEX_W, EVEX_CD8<32, CD8VT1>;

defm VGATHERPF0QPD: avx512_gather_scatter_prefetch<0xC7, MRM1m, "vgatherpf0qpd",
                     VK8WM, vz512mem>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;

defm VGATHERPF1DPS: avx512_gather_scatter_prefetch<0xC6, MRM2m, "vgatherpf1dps",
                     VK16WM, vz512mem>, EVEX_V512, EVEX_CD8<32, CD8VT1>;

defm VGATHERPF1QPS: avx512_gather_scatter_prefetch<0xC7, MRM2m, "vgatherpf1qps",
                     VK8WM, vz256xmem>, EVEX_V512, EVEX_CD8<64, CD8VT1>;

defm VGATHERPF1DPD: avx512_gather_scatter_prefetch<0xC6, MRM2m, "vgatherpf1dpd",
                     VK8WM, vy512mem>, EVEX_V512, VEX_W, EVEX_CD8<32, CD8VT1>;

defm VGATHERPF1QPD: avx512_gather_scatter_prefetch<0xC7, MRM2m, "vgatherpf1qpd",
                     VK8WM, vz512mem>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;

defm VSCATTERPF0DPS: avx512_gather_scatter_prefetch<0xC6, MRM5m, "vscatterpf0dps",
                     VK16WM, vz512mem>, EVEX_V512, EVEX_CD8<32, CD8VT1>;

defm VSCATTERPF0QPS: avx512_gather_scatter_prefetch<0xC7, MRM5m, "vscatterpf0qps",
                     VK8WM, vz256xmem>, EVEX_V512, EVEX_CD8<64, CD8VT1>;

defm VSCATTERPF0DPD: avx512_gather_scatter_prefetch<0xC6, MRM5m, "vscatterpf0dpd",
                     VK8WM, vy512mem>, EVEX_V512, VEX_W, EVEX_CD8<32, CD8VT1>;

defm VSCATTERPF0QPD: avx512_gather_scatter_prefetch<0xC7, MRM5m, "vscatterpf0qpd",
                     VK8WM, vz512mem>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;

defm VSCATTERPF1DPS: avx512_gather_scatter_prefetch<0xC6, MRM6m, "vscatterpf1dps",
                     VK16WM, vz512mem>, EVEX_V512, EVEX_CD8<32, CD8VT1>;

defm VSCATTERPF1QPS: avx512_gather_scatter_prefetch<0xC7, MRM6m, "vscatterpf1qps",
                     VK8WM, vz256xmem>, EVEX_V512, EVEX_CD8<64, CD8VT1>;

defm VSCATTERPF1DPD: avx512_gather_scatter_prefetch<0xC6, MRM6m, "vscatterpf1dpd",
                     VK8WM, vy512mem>, EVEX_V512, VEX_W, EVEX_CD8<32, CD8VT1>;

defm VSCATTERPF1QPD: avx512_gather_scatter_prefetch<0xC7, MRM6m, "vscatterpf1qpd",
                     VK8WM, vz512mem>, EVEX_V512, VEX_W, EVEX_CD8<64, CD8VT1>;

// Helper fragments to match sext vXi1 to vXiY.
def v64i1sextv64i8 : PatLeaf<(v64i8
                              (X86vsext
                               (v64i1 (X86pcmpgtm
                                (bc_v64i8 (v16i32 immAllZerosV)),
                                VR512:$src))))>;
def v32i1sextv32i16 : PatLeaf<(v32i16 (X86vsrai VR512:$src, (i8 15)))>;
def v16i1sextv16i32 : PatLeaf<(v16i32 (X86vsrai VR512:$src, (i8 31)))>;
def v8i1sextv8i64   : PatLeaf<(v8i64 (X86vsrai VR512:$src, (i8 63)))>;

multiclass cvt_by_vec_width<bits<8> opc, X86VectorVTInfo Vec, string OpcodeStr > {
def rr : AVX512XS8I<opc, MRMSrcReg, (outs Vec.RC:$dst), (ins Vec.KRC:$src),
                  !strconcat(OpcodeStr##Vec.Suffix, "\t{$src, $dst|$dst, $src}"),
                  [(set Vec.RC:$dst, (Vec.VT (X86vsext Vec.KRC:$src)))]>, EVEX;
}

multiclass cvt_mask_by_elt_width<bits<8> opc, AVX512VLVectorVTInfo VTInfo,
                                 string OpcodeStr, Predicate prd> {
let Predicates = [prd] in
  defm Z : cvt_by_vec_width<opc, VTInfo.info512, OpcodeStr>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : cvt_by_vec_width<opc, VTInfo.info256, OpcodeStr>, EVEX_V256;
    defm Z128 : cvt_by_vec_width<opc, VTInfo.info128, OpcodeStr>, EVEX_V128;
  }
}

multiclass avx512_convert_mask_to_vector<string OpcodeStr> {
  defm NAME##B : cvt_mask_by_elt_width<0x28, avx512vl_i8_info,  OpcodeStr,
                                       HasBWI>;
  defm NAME##W : cvt_mask_by_elt_width<0x28, avx512vl_i16_info, OpcodeStr,
                                       HasBWI>, VEX_W;
  defm NAME##D : cvt_mask_by_elt_width<0x38, avx512vl_i32_info, OpcodeStr,
                                       HasDQI>;
  defm NAME##Q : cvt_mask_by_elt_width<0x38, avx512vl_i64_info, OpcodeStr,
                                       HasDQI>, VEX_W;
}

defm VPMOVM2 : avx512_convert_mask_to_vector<"vpmovm2">;

multiclass convert_vector_to_mask_common<bits<8> opc, X86VectorVTInfo _, string OpcodeStr > {
    def rr : AVX512XS8I<opc, MRMSrcReg, (outs _.KRC:$dst), (ins _.RC:$src),
                        !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                        [(set _.KRC:$dst, (X86cvt2mask (_.VT _.RC:$src)))]>, EVEX;
}

// Use 512bit version to implement 128/256 bit in case NoVLX.
multiclass convert_vector_to_mask_lowering<X86VectorVTInfo ExtendInfo,
                                                            X86VectorVTInfo _> {

  def : Pat<(_.KVT (X86cvt2mask (_.VT _.RC:$src))),
            (_.KVT (COPY_TO_REGCLASS
                     (!cast<Instruction>(NAME#"Zrr")
                       (INSERT_SUBREG (ExtendInfo.VT (IMPLICIT_DEF)),
                                      _.RC:$src, _.SubRegIdx)),
                   _.KRC))>;
}

multiclass avx512_convert_vector_to_mask<bits<8> opc, string OpcodeStr,
                                   AVX512VLVectorVTInfo VTInfo, Predicate prd> {
  let Predicates = [prd] in
    defm Z : convert_vector_to_mask_common <opc, VTInfo.info512, OpcodeStr>,
                                            EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : convert_vector_to_mask_common<opc, VTInfo.info256, OpcodeStr>,
                                              EVEX_V256;
    defm Z128 : convert_vector_to_mask_common<opc, VTInfo.info128, OpcodeStr>,
                                               EVEX_V128;
  }
  let Predicates = [prd, NoVLX] in {
    defm Z256_Alt : convert_vector_to_mask_lowering<VTInfo.info512, VTInfo.info256>;
    defm Z128_Alt : convert_vector_to_mask_lowering<VTInfo.info512, VTInfo.info128>;
  }
}

defm VPMOVB2M : avx512_convert_vector_to_mask<0x29, "vpmovb2m",
                                              avx512vl_i8_info, HasBWI>;
defm VPMOVW2M : avx512_convert_vector_to_mask<0x29, "vpmovw2m",
                                              avx512vl_i16_info, HasBWI>, VEX_W;
defm VPMOVD2M : avx512_convert_vector_to_mask<0x39, "vpmovd2m",
                                              avx512vl_i32_info, HasDQI>;
defm VPMOVQ2M : avx512_convert_vector_to_mask<0x39, "vpmovq2m",
                                              avx512vl_i64_info, HasDQI>, VEX_W;

//===----------------------------------------------------------------------===//
// AVX-512 - COMPRESS and EXPAND
//

multiclass compress_by_vec_width_common<bits<8> opc, X86VectorVTInfo _,
                                 string OpcodeStr> {
  defm rr : AVX512_maskable<opc, MRMDestReg, _, (outs _.RC:$dst),
              (ins _.RC:$src1), OpcodeStr, "$src1", "$src1",
              (_.VT (X86compress _.RC:$src1))>, AVX5128IBase;

  let mayStore = 1, hasSideEffects = 0 in
  def mr : AVX5128I<opc, MRMDestMem, (outs),
              (ins _.MemOp:$dst, _.RC:$src),
              OpcodeStr # "\t{$src, $dst|$dst, $src}",
              []>, EVEX_CD8<_.EltSize, CD8VT1>;

  def mrk : AVX5128I<opc, MRMDestMem, (outs),
              (ins _.MemOp:$dst, _.KRCWM:$mask, _.RC:$src),
              OpcodeStr # "\t{$src, $dst {${mask}}|$dst {${mask}}, $src}",
              []>,
              EVEX_K, EVEX_CD8<_.EltSize, CD8VT1>;
}

multiclass compress_by_vec_width_lowering<X86VectorVTInfo _ > {

  def : Pat<(X86mCompressingStore addr:$dst, _.KRCWM:$mask,
                                               (_.VT _.RC:$src)),
            (!cast<Instruction>(NAME#_.ZSuffix##mrk)
                            addr:$dst, _.KRCWM:$mask, _.RC:$src)>;
}

multiclass compress_by_elt_width<bits<8> opc, string OpcodeStr,
                                 AVX512VLVectorVTInfo VTInfo> {
  defm Z : compress_by_vec_width_common<opc, VTInfo.info512, OpcodeStr>,
           compress_by_vec_width_lowering<VTInfo.info512>, EVEX_V512;

  let Predicates = [HasVLX] in {
    defm Z256 : compress_by_vec_width_common<opc, VTInfo.info256, OpcodeStr>,
                compress_by_vec_width_lowering<VTInfo.info256>, EVEX_V256;
    defm Z128 : compress_by_vec_width_common<opc, VTInfo.info128, OpcodeStr>,
                compress_by_vec_width_lowering<VTInfo.info128>, EVEX_V128;
  }
}

defm VPCOMPRESSD : compress_by_elt_width <0x8B, "vpcompressd", avx512vl_i32_info>,
                                         EVEX;
defm VPCOMPRESSQ : compress_by_elt_width <0x8B, "vpcompressq", avx512vl_i64_info>,
                                         EVEX, VEX_W;
defm VCOMPRESSPS : compress_by_elt_width <0x8A, "vcompressps", avx512vl_f32_info>,
                                         EVEX;
defm VCOMPRESSPD : compress_by_elt_width <0x8A, "vcompresspd", avx512vl_f64_info>,
                                         EVEX, VEX_W;

// expand
multiclass expand_by_vec_width<bits<8> opc, X86VectorVTInfo _,
                                 string OpcodeStr> {
  defm rr : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
              (ins _.RC:$src1), OpcodeStr, "$src1", "$src1",
              (_.VT (X86expand _.RC:$src1))>, AVX5128IBase;

  defm rm : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
              (ins _.MemOp:$src1), OpcodeStr, "$src1", "$src1",
              (_.VT (X86expand (_.VT (bitconvert
                                      (_.LdFrag addr:$src1)))))>,
            AVX5128IBase, EVEX_CD8<_.EltSize, CD8VT1>;
}

multiclass expand_by_vec_width_lowering<X86VectorVTInfo _ > {

  def : Pat<(_.VT (X86mExpandingLoad addr:$src, _.KRCWM:$mask, undef)),
            (!cast<Instruction>(NAME#_.ZSuffix##rmkz)
                                        _.KRCWM:$mask, addr:$src)>;

  def : Pat<(_.VT (X86mExpandingLoad addr:$src, _.KRCWM:$mask,
                                               (_.VT _.RC:$src0))),
            (!cast<Instruction>(NAME#_.ZSuffix##rmk)
                            _.RC:$src0, _.KRCWM:$mask, addr:$src)>;
}

multiclass expand_by_elt_width<bits<8> opc, string OpcodeStr,
                                 AVX512VLVectorVTInfo VTInfo> {
  defm Z : expand_by_vec_width<opc, VTInfo.info512, OpcodeStr>,
           expand_by_vec_width_lowering<VTInfo.info512>, EVEX_V512;

  let Predicates = [HasVLX] in {
    defm Z256 : expand_by_vec_width<opc, VTInfo.info256, OpcodeStr>,
                expand_by_vec_width_lowering<VTInfo.info256>, EVEX_V256;
    defm Z128 : expand_by_vec_width<opc, VTInfo.info128, OpcodeStr>,
                expand_by_vec_width_lowering<VTInfo.info128>, EVEX_V128;
  }
}

defm VPEXPANDD : expand_by_elt_width <0x89, "vpexpandd", avx512vl_i32_info>,
                                         EVEX;
defm VPEXPANDQ : expand_by_elt_width <0x89, "vpexpandq", avx512vl_i64_info>,
                                         EVEX, VEX_W;
defm VEXPANDPS : expand_by_elt_width <0x88, "vexpandps", avx512vl_f32_info>,
                                         EVEX;
defm VEXPANDPD : expand_by_elt_width <0x88, "vexpandpd", avx512vl_f64_info>,
                                         EVEX, VEX_W;

//handle instruction  reg_vec1 = op(reg_vec,imm)
//                               op(mem_vec,imm)
//                               op(broadcast(eltVt),imm)
//all instruction created with FROUND_CURRENT
multiclass avx512_unary_fp_packed_imm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                      X86VectorVTInfo _>{
  let ExeDomain = _.ExeDomain in {
  defm rri : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                      (ins _.RC:$src1, i32u8imm:$src2),
                      OpcodeStr##_.Suffix, "$src2, $src1", "$src1, $src2",
                      (OpNode (_.VT _.RC:$src1),
                              (i32 imm:$src2),
                              (i32 FROUND_CURRENT))>;
  defm rmi : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.MemOp:$src1, i32u8imm:$src2),
                    OpcodeStr##_.Suffix, "$src2, $src1", "$src1, $src2",
                    (OpNode (_.VT (bitconvert (_.LdFrag addr:$src1))),
                            (i32 imm:$src2),
                            (i32 FROUND_CURRENT))>;
  defm rmbi : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.ScalarMemOp:$src1, i32u8imm:$src2),
                    OpcodeStr##_.Suffix, "$src2, ${src1}"##_.BroadcastStr,
                    "${src1}"##_.BroadcastStr##", $src2",
                    (OpNode (_.VT (X86VBroadcast(_.ScalarLdFrag addr:$src1))),
                            (i32 imm:$src2),
                            (i32 FROUND_CURRENT))>, EVEX_B;
  }
}

//handle instruction  reg_vec1 = op(reg_vec2,reg_vec3,imm),{sae}
multiclass avx512_unary_fp_sae_packed_imm<bits<8> opc, string OpcodeStr,
                                             SDNode OpNode, X86VectorVTInfo _>{
  let ExeDomain = _.ExeDomain in
  defm rrib : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                      (ins _.RC:$src1, i32u8imm:$src2),
                      OpcodeStr##_.Suffix, "$src2, {sae}, $src1",
                      "$src1, {sae}, $src2",
                      (OpNode (_.VT _.RC:$src1),
                              (i32 imm:$src2),
                              (i32 FROUND_NO_EXC))>, EVEX_B;
}

multiclass avx512_common_unary_fp_sae_packed_imm<string OpcodeStr,
            AVX512VLVectorVTInfo _, bits<8> opc, SDNode OpNode, Predicate prd>{
  let Predicates = [prd] in {
    defm Z    : avx512_unary_fp_packed_imm<opc, OpcodeStr, OpNode, _.info512>,
                avx512_unary_fp_sae_packed_imm<opc, OpcodeStr, OpNode, _.info512>,
                                  EVEX_V512;
  }
  let Predicates = [prd, HasVLX] in {
    defm Z128 : avx512_unary_fp_packed_imm<opc, OpcodeStr, OpNode, _.info128>,
                                  EVEX_V128;
    defm Z256 : avx512_unary_fp_packed_imm<opc, OpcodeStr, OpNode, _.info256>,
                                  EVEX_V256;
  }
}

//handle instruction  reg_vec1 = op(reg_vec2,reg_vec3,imm)
//                               op(reg_vec2,mem_vec,imm)
//                               op(reg_vec2,broadcast(eltVt),imm)
//all instruction created with FROUND_CURRENT
multiclass avx512_fp_packed_imm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                X86VectorVTInfo _>{
  let ExeDomain = _.ExeDomain in {
  defm rri : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                      (ins _.RC:$src1, _.RC:$src2, i32u8imm:$src3),
                      OpcodeStr, "$src3, $src2, $src1", "$src1, $src2, $src3",
                      (OpNode (_.VT _.RC:$src1),
                              (_.VT _.RC:$src2),
                              (i32 imm:$src3),
                              (i32 FROUND_CURRENT))>;
  defm rmi : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.RC:$src1, _.MemOp:$src2, i32u8imm:$src3),
                    OpcodeStr, "$src3, $src2, $src1", "$src1, $src2, $src3",
                    (OpNode (_.VT _.RC:$src1),
                            (_.VT (bitconvert (_.LdFrag addr:$src2))),
                            (i32 imm:$src3),
                            (i32 FROUND_CURRENT))>;
  defm rmbi : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.RC:$src1, _.ScalarMemOp:$src2, i32u8imm:$src3),
                    OpcodeStr, "$src3, ${src2}"##_.BroadcastStr##", $src1",
                    "$src1, ${src2}"##_.BroadcastStr##", $src3",
                    (OpNode (_.VT _.RC:$src1),
                            (_.VT (X86VBroadcast(_.ScalarLdFrag addr:$src2))),
                            (i32 imm:$src3),
                            (i32 FROUND_CURRENT))>, EVEX_B;
  }
}

//handle instruction  reg_vec1 = op(reg_vec2,reg_vec3,imm)
//                               op(reg_vec2,mem_vec,imm)
multiclass avx512_3Op_rm_imm8<bits<8> opc, string OpcodeStr, SDNode OpNode,
                             X86VectorVTInfo DestInfo, X86VectorVTInfo SrcInfo>{
  let ExeDomain = DestInfo.ExeDomain in {
  defm rri : AVX512_maskable<opc, MRMSrcReg, DestInfo, (outs DestInfo.RC:$dst),
                  (ins SrcInfo.RC:$src1, SrcInfo.RC:$src2, u8imm:$src3),
                  OpcodeStr, "$src3, $src2, $src1", "$src1, $src2, $src3",
                  (DestInfo.VT (OpNode (SrcInfo.VT SrcInfo.RC:$src1),
                               (SrcInfo.VT SrcInfo.RC:$src2),
                               (i8 imm:$src3)))>;
  defm rmi : AVX512_maskable<opc, MRMSrcMem, DestInfo, (outs DestInfo.RC:$dst),
                (ins SrcInfo.RC:$src1, SrcInfo.MemOp:$src2, u8imm:$src3),
                OpcodeStr, "$src3, $src2, $src1", "$src1, $src2, $src3",
                (DestInfo.VT (OpNode (SrcInfo.VT SrcInfo.RC:$src1),
                             (SrcInfo.VT (bitconvert
                                                (SrcInfo.LdFrag addr:$src2))),
                             (i8 imm:$src3)))>;
  }
}

//handle instruction  reg_vec1 = op(reg_vec2,reg_vec3,imm)
//                               op(reg_vec2,mem_vec,imm)
//                               op(reg_vec2,broadcast(eltVt),imm)
multiclass avx512_3Op_imm8<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           X86VectorVTInfo _>:
  avx512_3Op_rm_imm8<opc, OpcodeStr, OpNode, _, _>{

  let ExeDomain = _.ExeDomain in
  defm rmbi : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.RC:$src1, _.ScalarMemOp:$src2, u8imm:$src3),
                    OpcodeStr, "$src3, ${src2}"##_.BroadcastStr##", $src1",
                    "$src1, ${src2}"##_.BroadcastStr##", $src3",
                    (OpNode (_.VT _.RC:$src1),
                            (_.VT (X86VBroadcast(_.ScalarLdFrag addr:$src2))),
                            (i8 imm:$src3))>, EVEX_B;
}

//handle scalar instruction  reg_vec1 = op(reg_vec2,reg_vec3,imm)
//                                      op(reg_vec2,mem_scalar,imm)
//all instruction created with FROUND_CURRENT
multiclass avx512_fp_scalar_imm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
  defm rri : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                      (ins _.RC:$src1, _.RC:$src2, i32u8imm:$src3),
                      OpcodeStr, "$src3, $src2, $src1", "$src1, $src2, $src3",
                      (OpNode (_.VT _.RC:$src1),
                              (_.VT _.RC:$src2),
                              (i32 imm:$src3),
                              (i32 FROUND_CURRENT))>;
  defm rmi : AVX512_maskable_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.RC:$src1, _.ScalarMemOp:$src2, i32u8imm:$src3),
                    OpcodeStr, "$src3, $src2, $src1", "$src1, $src2, $src3",
                    (OpNode (_.VT _.RC:$src1),
                            (_.VT (scalar_to_vector
                                      (_.ScalarLdFrag addr:$src2))),
                            (i32 imm:$src3),
                            (i32 FROUND_CURRENT))>;
  }
}

//handle instruction  reg_vec1 = op(reg_vec2,reg_vec3,imm),{sae}
multiclass avx512_fp_sae_packed_imm<bits<8> opc, string OpcodeStr,
                                             SDNode OpNode, X86VectorVTInfo _>{
  let ExeDomain = _.ExeDomain in
  defm rrib : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                      (ins _.RC:$src1, _.RC:$src2, i32u8imm:$src3),
                      OpcodeStr, "$src3, {sae}, $src2, $src1",
                      "$src1, $src2, {sae}, $src3",
                      (OpNode (_.VT _.RC:$src1),
                              (_.VT _.RC:$src2),
                              (i32 imm:$src3),
                              (i32 FROUND_NO_EXC))>, EVEX_B;
}
//handle scalar instruction  reg_vec1 = op(reg_vec2,reg_vec3,imm),{sae}
multiclass avx512_fp_sae_scalar_imm<bits<8> opc, string OpcodeStr,
                                             SDNode OpNode, X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in
  defm NAME#rrib : AVX512_maskable_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                      (ins _.RC:$src1, _.RC:$src2, i32u8imm:$src3),
                      OpcodeStr, "$src3, {sae}, $src2, $src1",
                      "$src1, $src2, {sae}, $src3",
                      (OpNode (_.VT _.RC:$src1),
                              (_.VT _.RC:$src2),
                              (i32 imm:$src3),
                              (i32 FROUND_NO_EXC))>, EVEX_B;
}

multiclass avx512_common_fp_sae_packed_imm<string OpcodeStr,
            AVX512VLVectorVTInfo _, bits<8> opc, SDNode OpNode, Predicate prd>{
  let Predicates = [prd] in {
    defm Z    : avx512_fp_packed_imm<opc, OpcodeStr, OpNode, _.info512>,
                avx512_fp_sae_packed_imm<opc, OpcodeStr, OpNode, _.info512>,
                                  EVEX_V512;

  }
  let Predicates = [prd, HasVLX] in {
    defm Z128 : avx512_fp_packed_imm<opc, OpcodeStr, OpNode, _.info128>,
                                  EVEX_V128;
    defm Z256 : avx512_fp_packed_imm<opc, OpcodeStr, OpNode, _.info256>,
                                  EVEX_V256;
  }
}

multiclass avx512_common_3Op_rm_imm8<bits<8> opc, SDNode OpNode, string OpStr,
                   AVX512VLVectorVTInfo DestInfo, AVX512VLVectorVTInfo SrcInfo>{
  let Predicates = [HasBWI] in {
    defm Z    : avx512_3Op_rm_imm8<opc, OpStr, OpNode, DestInfo.info512,
                           SrcInfo.info512>, EVEX_V512, AVX512AIi8Base, EVEX_4V;
  }
  let Predicates = [HasBWI, HasVLX] in {
    defm Z128 : avx512_3Op_rm_imm8<opc, OpStr, OpNode, DestInfo.info128,
                           SrcInfo.info128>, EVEX_V128, AVX512AIi8Base, EVEX_4V;
    defm Z256 : avx512_3Op_rm_imm8<opc, OpStr, OpNode,  DestInfo.info256,
                           SrcInfo.info256>, EVEX_V256, AVX512AIi8Base, EVEX_4V;
  }
}

multiclass avx512_common_3Op_imm8<string OpcodeStr, AVX512VLVectorVTInfo _,
                                bits<8> opc, SDNode OpNode>{
  let Predicates = [HasAVX512] in {
    defm Z    : avx512_3Op_imm8<opc, OpcodeStr, OpNode, _.info512>, EVEX_V512;
  }
  let Predicates = [HasAVX512, HasVLX] in {
    defm Z128 : avx512_3Op_imm8<opc, OpcodeStr, OpNode, _.info128>, EVEX_V128;
    defm Z256 : avx512_3Op_imm8<opc, OpcodeStr, OpNode, _.info256>, EVEX_V256;
  }
}

multiclass avx512_common_fp_sae_scalar_imm<string OpcodeStr,
                  X86VectorVTInfo _, bits<8> opc, SDNode OpNode, Predicate prd>{
  let Predicates = [prd] in {
     defm Z128 : avx512_fp_scalar_imm<opc, OpcodeStr, OpNode, _>,
                 avx512_fp_sae_scalar_imm<opc, OpcodeStr, OpNode, _>;
  }
}

multiclass avx512_common_unary_fp_sae_packed_imm_all<string OpcodeStr,
                    bits<8> opcPs, bits<8> opcPd, SDNode OpNode, Predicate prd>{
  defm PS : avx512_common_unary_fp_sae_packed_imm<OpcodeStr, avx512vl_f32_info,
                            opcPs, OpNode, prd>, EVEX_CD8<32, CD8VF>;
  defm PD : avx512_common_unary_fp_sae_packed_imm<OpcodeStr, avx512vl_f64_info,
                            opcPd, OpNode, prd>, EVEX_CD8<64, CD8VF>, VEX_W;
}


defm VREDUCE   : avx512_common_unary_fp_sae_packed_imm_all<"vreduce", 0x56, 0x56,
                              X86VReduce, HasDQI>, AVX512AIi8Base, EVEX;
defm VRNDSCALE : avx512_common_unary_fp_sae_packed_imm_all<"vrndscale", 0x08, 0x09,
                              X86VRndScale, HasAVX512>, AVX512AIi8Base, EVEX;
defm VGETMANT : avx512_common_unary_fp_sae_packed_imm_all<"vgetmant", 0x26, 0x26,
                              X86VGetMant, HasAVX512>, AVX512AIi8Base, EVEX;


defm VRANGEPD : avx512_common_fp_sae_packed_imm<"vrangepd", avx512vl_f64_info,
                                                       0x50, X86VRange, HasDQI>,
      AVX512AIi8Base, EVEX_4V, EVEX_CD8<64, CD8VF>, VEX_W;
defm VRANGEPS : avx512_common_fp_sae_packed_imm<"vrangeps", avx512vl_f32_info,
                                                       0x50, X86VRange, HasDQI>,
      AVX512AIi8Base, EVEX_4V, EVEX_CD8<32, CD8VF>;

defm VRANGESD: avx512_common_fp_sae_scalar_imm<"vrangesd", f64x_info,
                                                 0x51, X86VRange, HasDQI>,
      AVX512AIi8Base, VEX_LIG, EVEX_4V, EVEX_CD8<64, CD8VT1>, VEX_W;
defm VRANGESS: avx512_common_fp_sae_scalar_imm<"vrangess", f32x_info,
                                                 0x51, X86VRange, HasDQI>,
      AVX512AIi8Base, VEX_LIG, EVEX_4V, EVEX_CD8<32, CD8VT1>;

defm VREDUCESD: avx512_common_fp_sae_scalar_imm<"vreducesd", f64x_info,
                                                 0x57, X86Reduces, HasDQI>,
      AVX512AIi8Base, VEX_LIG, EVEX_4V, EVEX_CD8<64, CD8VT1>, VEX_W;
defm VREDUCESS: avx512_common_fp_sae_scalar_imm<"vreducess", f32x_info,
                                                 0x57, X86Reduces, HasDQI>,
      AVX512AIi8Base, VEX_LIG, EVEX_4V, EVEX_CD8<32, CD8VT1>;

defm VGETMANTSD: avx512_common_fp_sae_scalar_imm<"vgetmantsd", f64x_info,
                                                 0x27, X86GetMants, HasAVX512>,
      AVX512AIi8Base, VEX_LIG, EVEX_4V, EVEX_CD8<64, CD8VT1>, VEX_W;
defm VGETMANTSS: avx512_common_fp_sae_scalar_imm<"vgetmantss", f32x_info,
                                                 0x27, X86GetMants, HasAVX512>,
      AVX512AIi8Base, VEX_LIG, EVEX_4V, EVEX_CD8<32, CD8VT1>;

multiclass avx512_shuff_packed_128<string OpcodeStr, AVX512VLVectorVTInfo _,
                                       bits<8> opc, SDNode OpNode = X86Shuf128>{
  let Predicates = [HasAVX512] in {
    defm Z    : avx512_3Op_imm8<opc, OpcodeStr, OpNode, _.info512>, EVEX_V512;

  }
  let Predicates = [HasAVX512, HasVLX] in {
     defm Z256 : avx512_3Op_imm8<opc, OpcodeStr, OpNode, _.info256>, EVEX_V256;
  }
}
let Predicates = [HasAVX512] in {
def : Pat<(v16f32 (ffloor VR512:$src)),
          (VRNDSCALEPSZrri VR512:$src, (i32 0x1))>;
def : Pat<(v16f32 (fnearbyint VR512:$src)),
          (VRNDSCALEPSZrri VR512:$src, (i32 0xC))>;
def : Pat<(v16f32 (fceil VR512:$src)),
          (VRNDSCALEPSZrri VR512:$src, (i32 0x2))>;
def : Pat<(v16f32 (frint VR512:$src)),
          (VRNDSCALEPSZrri VR512:$src, (i32 0x4))>;
def : Pat<(v16f32 (ftrunc VR512:$src)),
          (VRNDSCALEPSZrri VR512:$src, (i32 0x3))>;

def : Pat<(v8f64 (ffloor VR512:$src)),
          (VRNDSCALEPDZrri VR512:$src, (i32 0x1))>;
def : Pat<(v8f64 (fnearbyint VR512:$src)),
          (VRNDSCALEPDZrri VR512:$src, (i32 0xC))>;
def : Pat<(v8f64 (fceil VR512:$src)),
          (VRNDSCALEPDZrri VR512:$src, (i32 0x2))>;
def : Pat<(v8f64 (frint VR512:$src)),
          (VRNDSCALEPDZrri VR512:$src, (i32 0x4))>;
def : Pat<(v8f64 (ftrunc VR512:$src)),
          (VRNDSCALEPDZrri VR512:$src, (i32 0x3))>;
}

defm VSHUFF32X4 : avx512_shuff_packed_128<"vshuff32x4",avx512vl_f32_info, 0x23>,
      AVX512AIi8Base, EVEX_4V, EVEX_CD8<32, CD8VF>;
defm VSHUFF64X2 : avx512_shuff_packed_128<"vshuff64x2",avx512vl_f64_info, 0x23>,
      AVX512AIi8Base, EVEX_4V, EVEX_CD8<64, CD8VF>, VEX_W;
defm VSHUFI32X4 : avx512_shuff_packed_128<"vshufi32x4",avx512vl_i32_info, 0x43>,
      AVX512AIi8Base, EVEX_4V, EVEX_CD8<32, CD8VF>;
defm VSHUFI64X2 : avx512_shuff_packed_128<"vshufi64x2",avx512vl_i64_info, 0x43>,
      AVX512AIi8Base, EVEX_4V, EVEX_CD8<64, CD8VF>, VEX_W;

let Predicates = [HasAVX512] in {
// Provide fallback in case the load node that is used in the broadcast
// patterns above is used by additional users, which prevents the pattern
// selection.
def : Pat<(v8f64 (X86SubVBroadcast (v2f64 VR128X:$src))),
          (VSHUFF64X2Zrri (INSERT_SUBREG (v8f64 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                          (INSERT_SUBREG (v8f64 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                          0)>;
def : Pat<(v8i64 (X86SubVBroadcast (v2i64 VR128X:$src))),
          (VSHUFI64X2Zrri (INSERT_SUBREG (v8i64 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                          (INSERT_SUBREG (v8i64 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                          0)>;

def : Pat<(v16f32 (X86SubVBroadcast (v4f32 VR128X:$src))),
          (VSHUFF32X4Zrri (INSERT_SUBREG (v16f32 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                          (INSERT_SUBREG (v16f32 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                          0)>;
def : Pat<(v16i32 (X86SubVBroadcast (v4i32 VR128X:$src))),
          (VSHUFI32X4Zrri (INSERT_SUBREG (v16i32 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                          (INSERT_SUBREG (v16i32 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                          0)>;

def : Pat<(v32i16 (X86SubVBroadcast (v8i16 VR128X:$src))),
          (VSHUFI32X4Zrri (INSERT_SUBREG (v32i16 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                          (INSERT_SUBREG (v32i16 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                          0)>;

def : Pat<(v64i8 (X86SubVBroadcast (v16i8 VR128X:$src))),
          (VSHUFI32X4Zrri (INSERT_SUBREG (v64i8 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                          (INSERT_SUBREG (v64i8 (IMPLICIT_DEF)), VR128X:$src, sub_xmm),
                          0)>;
}

multiclass avx512_valign<string OpcodeStr, AVX512VLVectorVTInfo VTInfo_I> {
  defm NAME:       avx512_common_3Op_imm8<OpcodeStr, VTInfo_I, 0x03, X86VAlign>,
                           AVX512AIi8Base, EVEX_4V;
}

defm VALIGND: avx512_valign<"valignd", avx512vl_i32_info>,
                                                  EVEX_CD8<32, CD8VF>;
defm VALIGNQ: avx512_valign<"valignq", avx512vl_i64_info>,
                                                  EVEX_CD8<64, CD8VF>, VEX_W;

multiclass avx512_vpalignr_lowering<X86VectorVTInfo _ , list<Predicate> p>{
  let Predicates = p in
    def NAME#_.VTName#rri:
          Pat<(_.VT (X86PAlignr _.RC:$src1, _.RC:$src2, (i8 imm:$imm))),
              (!cast<Instruction>(NAME#_.ZSuffix#rri)
                    _.RC:$src1, _.RC:$src2, imm:$imm)>;
}

multiclass avx512_vpalignr_lowering_common<AVX512VLVectorVTInfo _>:
      avx512_vpalignr_lowering<_.info512, [HasBWI]>,
      avx512_vpalignr_lowering<_.info128, [HasBWI, HasVLX]>,
      avx512_vpalignr_lowering<_.info256, [HasBWI, HasVLX]>;

defm VPALIGNR:   avx512_common_3Op_rm_imm8<0x0F, X86PAlignr, "vpalignr" ,
                                          avx512vl_i8_info, avx512vl_i8_info>,
                avx512_vpalignr_lowering_common<avx512vl_i16_info>,
                avx512_vpalignr_lowering_common<avx512vl_i32_info>,
                avx512_vpalignr_lowering_common<avx512vl_f32_info>,
                avx512_vpalignr_lowering_common<avx512vl_i64_info>,
                avx512_vpalignr_lowering_common<avx512vl_f64_info>,
                EVEX_CD8<8, CD8VF>;

defm VDBPSADBW: avx512_common_3Op_rm_imm8<0x42, X86dbpsadbw, "vdbpsadbw" ,
                    avx512vl_i16_info, avx512vl_i8_info>, EVEX_CD8<8, CD8VF>;

multiclass avx512_unary_rm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                           X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
  defm rr : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                    (ins _.RC:$src1), OpcodeStr,
                    "$src1", "$src1",
                    (_.VT (OpNode _.RC:$src1))>, EVEX, AVX5128IBase;

  defm rm : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                  (ins _.MemOp:$src1), OpcodeStr,
                  "$src1", "$src1",
                  (_.VT (OpNode (bitconvert (_.LdFrag addr:$src1))))>,
            EVEX, AVX5128IBase, EVEX_CD8<_.EltSize, CD8VF>;
  }
}

multiclass avx512_unary_rmb<bits<8> opc, string OpcodeStr, SDNode OpNode,
                            X86VectorVTInfo _> :
           avx512_unary_rm<opc, OpcodeStr, OpNode, _> {
  defm rmb : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                  (ins _.ScalarMemOp:$src1), OpcodeStr,
                  "${src1}"##_.BroadcastStr,
                  "${src1}"##_.BroadcastStr,
                  (_.VT (OpNode (X86VBroadcast
                                    (_.ScalarLdFrag addr:$src1))))>,
             EVEX, AVX5128IBase, EVEX_B, EVEX_CD8<_.EltSize, CD8VF>;
}

multiclass avx512_unary_rm_vl<bits<8> opc, string OpcodeStr, SDNode OpNode,
                              AVX512VLVectorVTInfo VTInfo, Predicate prd> {
  let Predicates = [prd] in
    defm Z : avx512_unary_rm<opc, OpcodeStr, OpNode, VTInfo.info512>, EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_unary_rm<opc, OpcodeStr, OpNode, VTInfo.info256>,
                              EVEX_V256;
    defm Z128 : avx512_unary_rm<opc, OpcodeStr, OpNode, VTInfo.info128>,
                              EVEX_V128;
  }
}

multiclass avx512_unary_rmb_vl<bits<8> opc, string OpcodeStr, SDNode OpNode,
                               AVX512VLVectorVTInfo VTInfo, Predicate prd> {
  let Predicates = [prd] in
    defm Z : avx512_unary_rmb<opc, OpcodeStr, OpNode, VTInfo.info512>,
                              EVEX_V512;

  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_unary_rmb<opc, OpcodeStr, OpNode, VTInfo.info256>,
                                 EVEX_V256;
    defm Z128 : avx512_unary_rmb<opc, OpcodeStr, OpNode, VTInfo.info128>,
                                 EVEX_V128;
  }
}

multiclass avx512_unary_rm_vl_dq<bits<8> opc_d, bits<8> opc_q, string OpcodeStr,
                                 SDNode OpNode, Predicate prd> {
  defm Q : avx512_unary_rmb_vl<opc_q, OpcodeStr#"q", OpNode, avx512vl_i64_info,
                               prd>, VEX_W;
  defm D : avx512_unary_rmb_vl<opc_d, OpcodeStr#"d", OpNode, avx512vl_i32_info,
                               prd>;
}

multiclass avx512_unary_rm_vl_bw<bits<8> opc_b, bits<8> opc_w, string OpcodeStr,
                                 SDNode OpNode, Predicate prd> {
  defm W : avx512_unary_rm_vl<opc_w, OpcodeStr#"w", OpNode, avx512vl_i16_info, prd>;
  defm B : avx512_unary_rm_vl<opc_b, OpcodeStr#"b", OpNode, avx512vl_i8_info, prd>;
}

multiclass avx512_unary_rm_vl_all<bits<8> opc_b, bits<8> opc_w,
                                  bits<8> opc_d, bits<8> opc_q,
                                  string OpcodeStr, SDNode OpNode> {
  defm NAME : avx512_unary_rm_vl_dq<opc_d, opc_q, OpcodeStr, OpNode,
                                    HasAVX512>,
              avx512_unary_rm_vl_bw<opc_b, opc_w, OpcodeStr, OpNode,
                                    HasBWI>;
}

defm VPABS : avx512_unary_rm_vl_all<0x1C, 0x1D, 0x1E, 0x1F, "vpabs", abs>;

multiclass avx512_ctlz<bits<8> opc, string OpcodeStr, Predicate prd>{

  defm NAME :          avx512_unary_rm_vl_dq<opc, opc, OpcodeStr, ctlz, prd>;
}

defm VPLZCNT    : avx512_ctlz<0x44, "vplzcnt", HasCDI>;
defm VPCONFLICT : avx512_unary_rm_vl_dq<0xC4, 0xC4, "vpconflict", X86Conflict, HasCDI>;

//===---------------------------------------------------------------------===//
// Replicate Single FP - MOVSHDUP and MOVSLDUP
//===---------------------------------------------------------------------===//
multiclass avx512_replicate<bits<8> opc, string OpcodeStr, SDNode OpNode>{
  defm NAME:       avx512_unary_rm_vl<opc, OpcodeStr, OpNode, avx512vl_f32_info,
                                      HasAVX512>, XS;
}

defm VMOVSHDUP : avx512_replicate<0x16, "vmovshdup", X86Movshdup>;
defm VMOVSLDUP : avx512_replicate<0x12, "vmovsldup", X86Movsldup>;

//===----------------------------------------------------------------------===//
// AVX-512 - MOVDDUP
//===----------------------------------------------------------------------===//

multiclass avx512_movddup_128<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                                            X86VectorVTInfo _> {
  let ExeDomain = _.ExeDomain in {
  defm rr : AVX512_maskable<opc, MRMSrcReg, _, (outs _.RC:$dst),
                   (ins _.RC:$src), OpcodeStr, "$src", "$src",
                   (_.VT (OpNode (_.VT _.RC:$src)))>, EVEX;
  defm rm : AVX512_maskable<opc, MRMSrcMem, _, (outs _.RC:$dst),
                 (ins _.ScalarMemOp:$src), OpcodeStr, "$src", "$src",
                 (_.VT (OpNode (_.VT (scalar_to_vector
                                       (_.ScalarLdFrag addr:$src)))))>,
                 EVEX, EVEX_CD8<_.EltSize, CD8VH>;
  }
}

multiclass avx512_movddup_common<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                                  AVX512VLVectorVTInfo VTInfo> {

  defm Z : avx512_unary_rm<opc, OpcodeStr, OpNode, VTInfo.info512>, EVEX_V512;

  let Predicates = [HasAVX512, HasVLX] in {
    defm Z256 : avx512_unary_rm<opc, OpcodeStr, OpNode, VTInfo.info256>,
                               EVEX_V256;
    defm Z128 : avx512_movddup_128<opc, OpcodeStr, OpNode, VTInfo.info128>,
                               EVEX_V128;
  }
}

multiclass avx512_movddup<bits<8> opc, string OpcodeStr, SDNode OpNode>{
  defm NAME:      avx512_movddup_common<opc, OpcodeStr, OpNode,
                                        avx512vl_f64_info>, XD, VEX_W;
}

defm VMOVDDUP : avx512_movddup<0x12, "vmovddup", X86Movddup>;

let Predicates = [HasVLX] in {
def : Pat<(X86Movddup (loadv2f64 addr:$src)),
          (VMOVDDUPZ128rm addr:$src)>;
def : Pat<(v2f64 (X86VBroadcast (loadf64 addr:$src))),
          (VMOVDDUPZ128rm addr:$src)>;
def : Pat<(v2f64 (X86VBroadcast f64:$src)),
          (VMOVDDUPZ128rr (COPY_TO_REGCLASS FR64X:$src, VR128X))>;

def : Pat<(vselect (v2i1 VK2WM:$mask), (X86Movddup (loadv2f64 addr:$src)),
                   (v2f64 VR128X:$src0)),
          (VMOVDDUPZ128rmk VR128X:$src0, VK2WM:$mask, addr:$src)>;
def : Pat<(vselect (v2i1 VK2WM:$mask), (X86Movddup (loadv2f64 addr:$src)),
                   (bitconvert (v4i32 immAllZerosV))),
          (VMOVDDUPZ128rmkz VK2WM:$mask, addr:$src)>;

def : Pat<(vselect (v2i1 VK2WM:$mask), (v2f64 (X86VBroadcast f64:$src)),
                   (v2f64 VR128X:$src0)),
          (VMOVDDUPZ128rrk VR128X:$src0, VK2WM:$mask,
                           (COPY_TO_REGCLASS FR64X:$src, VR128X))>;
def : Pat<(vselect (v2i1 VK2WM:$mask), (v2f64 (X86VBroadcast f64:$src)),
                   (bitconvert (v4i32 immAllZerosV))),
          (VMOVDDUPZ128rrkz VK2WM:$mask, (COPY_TO_REGCLASS FR64X:$src, VR128X))>;

def : Pat<(vselect (v2i1 VK2WM:$mask), (v2f64 (X86VBroadcast (loadf64 addr:$src))),
                   (v2f64 VR128X:$src0)),
          (VMOVDDUPZ128rmk VR128X:$src0, VK2WM:$mask, addr:$src)>;
def : Pat<(vselect (v2i1 VK2WM:$mask), (v2f64 (X86VBroadcast (loadf64 addr:$src))),
                   (bitconvert (v4i32 immAllZerosV))),
          (VMOVDDUPZ128rmkz VK2WM:$mask, addr:$src)>;
}

//===----------------------------------------------------------------------===//
// AVX-512 - Unpack Instructions
//===----------------------------------------------------------------------===//
defm VUNPCKH : avx512_fp_binop_p<0x15, "vunpckh", X86Unpckh, HasAVX512,
                                 SSE_ALU_ITINS_S>;
defm VUNPCKL : avx512_fp_binop_p<0x14, "vunpckl", X86Unpckl, HasAVX512,
                                 SSE_ALU_ITINS_S>;

defm VPUNPCKLBW : avx512_binop_rm_vl_b<0x60, "vpunpcklbw", X86Unpckl,
                                       SSE_INTALU_ITINS_P, HasBWI>;
defm VPUNPCKHBW : avx512_binop_rm_vl_b<0x68, "vpunpckhbw", X86Unpckh,
                                       SSE_INTALU_ITINS_P, HasBWI>;
defm VPUNPCKLWD : avx512_binop_rm_vl_w<0x61, "vpunpcklwd", X86Unpckl,
                                       SSE_INTALU_ITINS_P, HasBWI>;
defm VPUNPCKHWD : avx512_binop_rm_vl_w<0x69, "vpunpckhwd", X86Unpckh,
                                       SSE_INTALU_ITINS_P, HasBWI>;

defm VPUNPCKLDQ : avx512_binop_rm_vl_d<0x62, "vpunpckldq", X86Unpckl,
                                       SSE_INTALU_ITINS_P, HasAVX512>;
defm VPUNPCKHDQ : avx512_binop_rm_vl_d<0x6A, "vpunpckhdq", X86Unpckh,
                                       SSE_INTALU_ITINS_P, HasAVX512>;
defm VPUNPCKLQDQ : avx512_binop_rm_vl_q<0x6C, "vpunpcklqdq", X86Unpckl,
                                       SSE_INTALU_ITINS_P, HasAVX512>;
defm VPUNPCKHQDQ : avx512_binop_rm_vl_q<0x6D, "vpunpckhqdq", X86Unpckh,
                                       SSE_INTALU_ITINS_P, HasAVX512>;

//===----------------------------------------------------------------------===//
// AVX-512 - Extract & Insert Integer Instructions
//===----------------------------------------------------------------------===//

multiclass avx512_extract_elt_bw_m<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                                            X86VectorVTInfo _> {
  def mr : AVX512Ii8<opc, MRMDestMem, (outs),
              (ins _.ScalarMemOp:$dst, _.RC:$src1, u8imm:$src2),
              OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
              [(store (_.EltVT (trunc (assertzext (OpNode (_.VT _.RC:$src1),
                                                          imm:$src2)))),
                      addr:$dst)]>,
              EVEX, EVEX_CD8<_.EltSize, CD8VT1>;
}

multiclass avx512_extract_elt_b<string OpcodeStr, X86VectorVTInfo _> {
  let Predicates = [HasBWI] in {
    def rr : AVX512Ii8<0x14, MRMDestReg, (outs GR32orGR64:$dst),
                  (ins _.RC:$src1, u8imm:$src2),
                  OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                  [(set GR32orGR64:$dst,
                        (X86pextrb (_.VT _.RC:$src1), imm:$src2))]>,
                  EVEX, TAPD;

    defm NAME : avx512_extract_elt_bw_m<0x14, OpcodeStr, X86pextrb, _>, TAPD;
  }
}

multiclass avx512_extract_elt_w<string OpcodeStr, X86VectorVTInfo _> {
  let Predicates = [HasBWI] in {
    def rr : AVX512Ii8<0xC5, MRMSrcReg, (outs GR32orGR64:$dst),
                  (ins _.RC:$src1, u8imm:$src2),
                  OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                  [(set GR32orGR64:$dst,
                        (X86pextrw (_.VT _.RC:$src1), imm:$src2))]>,
                  EVEX, PD;

    let hasSideEffects = 0 in
    def rr_REV : AVX512Ii8<0x15, MRMDestReg, (outs GR32orGR64:$dst),
                   (ins _.RC:$src1, u8imm:$src2),
                   OpcodeStr#".s\t{$src2, $src1, $dst|$dst, $src1, $src2}", []>,
                   EVEX, TAPD;

    defm NAME : avx512_extract_elt_bw_m<0x15, OpcodeStr, X86pextrw, _>, TAPD;
  }
}

multiclass avx512_extract_elt_dq<string OpcodeStr, X86VectorVTInfo _,
                                                            RegisterClass GRC> {
  let Predicates = [HasDQI] in {
    def rr : AVX512Ii8<0x16, MRMDestReg, (outs GRC:$dst),
                  (ins _.RC:$src1, u8imm:$src2),
                  OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                  [(set GRC:$dst,
                      (extractelt (_.VT _.RC:$src1), imm:$src2))]>,
                  EVEX, TAPD;

    def mr : AVX512Ii8<0x16, MRMDestMem, (outs),
                (ins _.ScalarMemOp:$dst, _.RC:$src1, u8imm:$src2),
                OpcodeStr#"\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                [(store (extractelt (_.VT _.RC:$src1),
                                    imm:$src2),addr:$dst)]>,
                EVEX, EVEX_CD8<_.EltSize, CD8VT1>, TAPD;
  }
}

defm VPEXTRBZ : avx512_extract_elt_b<"vpextrb", v16i8x_info>;
defm VPEXTRWZ : avx512_extract_elt_w<"vpextrw", v8i16x_info>;
defm VPEXTRDZ : avx512_extract_elt_dq<"vpextrd", v4i32x_info, GR32>;
defm VPEXTRQZ : avx512_extract_elt_dq<"vpextrq", v2i64x_info, GR64>, VEX_W;

multiclass avx512_insert_elt_m<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                            X86VectorVTInfo _, PatFrag LdFrag> {
  def rm : AVX512Ii8<opc, MRMSrcMem, (outs _.RC:$dst),
      (ins _.RC:$src1,  _.ScalarMemOp:$src2, u8imm:$src3),
      OpcodeStr#"\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
      [(set _.RC:$dst,
          (_.VT (OpNode _.RC:$src1, (LdFrag addr:$src2), imm:$src3)))]>,
      EVEX_4V, EVEX_CD8<_.EltSize, CD8VT1>;
}

multiclass avx512_insert_elt_bw<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                            X86VectorVTInfo _, PatFrag LdFrag> {
  let Predicates = [HasBWI] in {
    def rr : AVX512Ii8<opc, MRMSrcReg, (outs _.RC:$dst),
        (ins _.RC:$src1, GR32orGR64:$src2, u8imm:$src3),
        OpcodeStr#"\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
        [(set _.RC:$dst,
            (OpNode _.RC:$src1, GR32orGR64:$src2, imm:$src3))]>, EVEX_4V;

    defm NAME : avx512_insert_elt_m<opc, OpcodeStr, OpNode, _, LdFrag>;
  }
}

multiclass avx512_insert_elt_dq<bits<8> opc, string OpcodeStr,
                                         X86VectorVTInfo _, RegisterClass GRC> {
  let Predicates = [HasDQI] in {
    def rr : AVX512Ii8<opc, MRMSrcReg, (outs _.RC:$dst),
        (ins _.RC:$src1, GRC:$src2, u8imm:$src3),
        OpcodeStr#"\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}",
        [(set _.RC:$dst,
            (_.VT (insertelt _.RC:$src1, GRC:$src2, imm:$src3)))]>,
        EVEX_4V, TAPD;

    defm NAME : avx512_insert_elt_m<opc, OpcodeStr, insertelt, _,
                                    _.ScalarLdFrag>, TAPD;
  }
}

defm VPINSRBZ : avx512_insert_elt_bw<0x20, "vpinsrb", X86pinsrb, v16i8x_info,
                                     extloadi8>, TAPD;
defm VPINSRWZ : avx512_insert_elt_bw<0xC4, "vpinsrw", X86pinsrw, v8i16x_info,
                                     extloadi16>, PD;
defm VPINSRDZ : avx512_insert_elt_dq<0x22, "vpinsrd", v4i32x_info, GR32>;
defm VPINSRQZ : avx512_insert_elt_dq<0x22, "vpinsrq", v2i64x_info, GR64>, VEX_W;
//===----------------------------------------------------------------------===//
// VSHUFPS - VSHUFPD Operations
//===----------------------------------------------------------------------===//
multiclass avx512_shufp<string OpcodeStr, AVX512VLVectorVTInfo VTInfo_I,
                                                AVX512VLVectorVTInfo VTInfo_FP>{
  defm NAME:     avx512_common_3Op_imm8<OpcodeStr, VTInfo_FP, 0xC6, X86Shufp>,
                                   EVEX_CD8<VTInfo_FP.info512.EltSize, CD8VF>,
                                   AVX512AIi8Base, EVEX_4V;
}

defm VSHUFPS: avx512_shufp<"vshufps", avx512vl_i32_info, avx512vl_f32_info>, PS;
defm VSHUFPD: avx512_shufp<"vshufpd", avx512vl_i64_info, avx512vl_f64_info>, PD, VEX_W;
//===----------------------------------------------------------------------===//
// AVX-512 - Byte shift Left/Right
//===----------------------------------------------------------------------===//

multiclass avx512_shift_packed<bits<8> opc, SDNode OpNode, Format MRMr,
                             Format MRMm, string OpcodeStr, X86VectorVTInfo _>{
  def rr : AVX512<opc, MRMr,
             (outs _.RC:$dst), (ins _.RC:$src1, u8imm:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set _.RC:$dst,(_.VT (OpNode _.RC:$src1, (i8 imm:$src2))))]>;
  def rm : AVX512<opc, MRMm,
           (outs _.RC:$dst), (ins _.MemOp:$src1, u8imm:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           [(set _.RC:$dst,(_.VT (OpNode
                                 (_.VT (bitconvert (_.LdFrag addr:$src1))),
                                 (i8 imm:$src2))))]>;
}

multiclass avx512_shift_packed_all<bits<8> opc, SDNode OpNode, Format MRMr,
                                 Format MRMm, string OpcodeStr, Predicate prd>{
  let Predicates = [prd] in
    defm Z512 : avx512_shift_packed<opc, OpNode, MRMr, MRMm,
                                    OpcodeStr, v64i8_info>, EVEX_V512;
  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_shift_packed<opc, OpNode, MRMr, MRMm,
                                    OpcodeStr, v32i8x_info>, EVEX_V256;
    defm Z128 : avx512_shift_packed<opc, OpNode, MRMr, MRMm,
                                    OpcodeStr, v16i8x_info>, EVEX_V128;
  }
}
defm VPSLLDQ : avx512_shift_packed_all<0x73, X86vshldq, MRM7r, MRM7m, "vpslldq",
                                       HasBWI>, AVX512PDIi8Base, EVEX_4V;
defm VPSRLDQ : avx512_shift_packed_all<0x73, X86vshrdq, MRM3r, MRM3m, "vpsrldq",
                                       HasBWI>, AVX512PDIi8Base, EVEX_4V;


multiclass avx512_psadbw_packed<bits<8> opc, SDNode OpNode,
                                string OpcodeStr, X86VectorVTInfo _dst,
                                X86VectorVTInfo _src>{
  def rr : AVX512BI<opc, MRMSrcReg,
             (outs _dst.RC:$dst), (ins _src.RC:$src1, _src.RC:$src2),
             !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set _dst.RC:$dst,(_dst.VT
                                (OpNode (_src.VT _src.RC:$src1),
                                        (_src.VT _src.RC:$src2))))]>;
  def rm : AVX512BI<opc, MRMSrcMem,
           (outs _dst.RC:$dst), (ins _src.RC:$src1, _src.MemOp:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           [(set _dst.RC:$dst,(_dst.VT
                              (OpNode (_src.VT _src.RC:$src1),
                              (_src.VT (bitconvert
                                        (_src.LdFrag addr:$src2))))))]>;
}

multiclass avx512_psadbw_packed_all<bits<8> opc, SDNode OpNode,
                                    string OpcodeStr, Predicate prd> {
  let Predicates = [prd] in
    defm Z512 : avx512_psadbw_packed<opc, OpNode, OpcodeStr, v8i64_info,
                                    v64i8_info>, EVEX_V512;
  let Predicates = [prd, HasVLX] in {
    defm Z256 : avx512_psadbw_packed<opc, OpNode, OpcodeStr, v4i64x_info,
                                    v32i8x_info>, EVEX_V256;
    defm Z128 : avx512_psadbw_packed<opc, OpNode, OpcodeStr, v2i64x_info,
                                    v16i8x_info>, EVEX_V128;
  }
}

defm VPSADBW : avx512_psadbw_packed_all<0xf6, X86psadbw, "vpsadbw",
                                       HasBWI>, EVEX_4V;

// Transforms to swizzle an immediate to enable better matching when
// memory operand isn't in the right place.
def VPTERNLOG321_imm8 : SDNodeXForm<imm, [{
  // Convert a VPTERNLOG immediate by swapping operand 0 and operand 2.
  uint8_t Imm = N->getZExtValue();
  // Swap bits 1/4 and 3/6.
  uint8_t NewImm = Imm & 0xa5;
  if (Imm & 0x02) NewImm |= 0x10;
  if (Imm & 0x10) NewImm |= 0x02;
  if (Imm & 0x08) NewImm |= 0x40;
  if (Imm & 0x40) NewImm |= 0x08;
  return getI8Imm(NewImm, SDLoc(N));
}]>;
def VPTERNLOG213_imm8 : SDNodeXForm<imm, [{
  // Convert a VPTERNLOG immediate by swapping operand 1 and operand 2.
  uint8_t Imm = N->getZExtValue();
  // Swap bits 2/4 and 3/5.
  uint8_t NewImm = Imm & 0xc3;
  if (Imm & 0x04) NewImm |= 0x10;
  if (Imm & 0x10) NewImm |= 0x04;
  if (Imm & 0x08) NewImm |= 0x20;
  if (Imm & 0x20) NewImm |= 0x08;
  return getI8Imm(NewImm, SDLoc(N));
}]>;
def VPTERNLOG132_imm8 : SDNodeXForm<imm, [{
  // Convert a VPTERNLOG immediate by swapping operand 1 and operand 2.
  uint8_t Imm = N->getZExtValue();
  // Swap bits 1/2 and 5/6.
  uint8_t NewImm = Imm & 0x99;
  if (Imm & 0x02) NewImm |= 0x04;
  if (Imm & 0x04) NewImm |= 0x02;
  if (Imm & 0x20) NewImm |= 0x40;
  if (Imm & 0x40) NewImm |= 0x20;
  return getI8Imm(NewImm, SDLoc(N));
}]>;
def VPTERNLOG231_imm8 : SDNodeXForm<imm, [{
  // Convert a VPTERNLOG immediate by moving operand 1 to the end.
  uint8_t Imm = N->getZExtValue();
  // Move bits 1->2, 2->4, 3->6, 4->1, 5->3, 6->5
  uint8_t NewImm = Imm & 0x81;
  if (Imm & 0x02) NewImm |= 0x04;
  if (Imm & 0x04) NewImm |= 0x10;
  if (Imm & 0x08) NewImm |= 0x40;
  if (Imm & 0x10) NewImm |= 0x02;
  if (Imm & 0x20) NewImm |= 0x08;
  if (Imm & 0x40) NewImm |= 0x20;
  return getI8Imm(NewImm, SDLoc(N));
}]>;
def VPTERNLOG312_imm8 : SDNodeXForm<imm, [{
  // Convert a VPTERNLOG immediate by moving operand 2 to the beginning.
  uint8_t Imm = N->getZExtValue();
  // Move bits 1->4, 2->1, 3->5, 4->2, 5->6, 6->3
  uint8_t NewImm = Imm & 0x81;
  if (Imm & 0x02) NewImm |= 0x10;
  if (Imm & 0x04) NewImm |= 0x02;
  if (Imm & 0x08) NewImm |= 0x20;
  if (Imm & 0x10) NewImm |= 0x04;
  if (Imm & 0x20) NewImm |= 0x40;
  if (Imm & 0x40) NewImm |= 0x08;
  return getI8Imm(NewImm, SDLoc(N));
}]>;

multiclass avx512_ternlog<bits<8> opc, string OpcodeStr, SDNode OpNode,
                          X86VectorVTInfo _>{
  let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in {
  defm rri : AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
                      (ins _.RC:$src2, _.RC:$src3, u8imm:$src4),
                      OpcodeStr, "$src4, $src3, $src2", "$src2, $src3, $src4",
                      (OpNode (_.VT _.RC:$src1),
                              (_.VT _.RC:$src2),
                              (_.VT _.RC:$src3),
                              (i8 imm:$src4)), 1, 1>, AVX512AIi8Base, EVEX_4V;
  defm rmi : AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.RC:$src2, _.MemOp:$src3, u8imm:$src4),
                    OpcodeStr, "$src4, $src3, $src2", "$src2, $src3, $src4",
                    (OpNode (_.VT _.RC:$src1),
                            (_.VT _.RC:$src2),
                            (_.VT (bitconvert (_.LdFrag addr:$src3))),
                            (i8 imm:$src4)), 1, 0>,
                    AVX512AIi8Base, EVEX_4V, EVEX_CD8<_.EltSize, CD8VF>;
  defm rmbi : AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
                    (ins _.RC:$src2, _.ScalarMemOp:$src3, u8imm:$src4),
                    OpcodeStr, "$src4, ${src3}"##_.BroadcastStr##", $src2",
                    "$src2, ${src3}"##_.BroadcastStr##", $src4",
                    (OpNode (_.VT _.RC:$src1),
                            (_.VT _.RC:$src2),
                            (_.VT (X86VBroadcast(_.ScalarLdFrag addr:$src3))),
                            (i8 imm:$src4)), 1, 0>, EVEX_B,
                    AVX512AIi8Base, EVEX_4V, EVEX_CD8<_.EltSize, CD8VF>;
  }// Constraints = "$src1 = $dst"

  // Additional patterns for matching passthru operand in other positions.
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode _.RC:$src3, _.RC:$src2, _.RC:$src1, (i8 imm:$src4)),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#_.ZSuffix#rrik) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, _.RC:$src3, (VPTERNLOG321_imm8 imm:$src4))>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode _.RC:$src2, _.RC:$src1, _.RC:$src3, (i8 imm:$src4)),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#_.ZSuffix#rrik) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, _.RC:$src3, (VPTERNLOG213_imm8 imm:$src4))>;

  // Additional patterns for matching loads in other positions.
  def : Pat<(_.VT (OpNode (bitconvert (_.LdFrag addr:$src3)),
                          _.RC:$src2, _.RC:$src1, (i8 imm:$src4))),
            (!cast<Instruction>(NAME#_.ZSuffix#rmi) _.RC:$src1, _.RC:$src2,
                                   addr:$src3, (VPTERNLOG321_imm8 imm:$src4))>;
  def : Pat<(_.VT (OpNode _.RC:$src1,
                          (bitconvert (_.LdFrag addr:$src3)),
                          _.RC:$src2, (i8 imm:$src4))),
            (!cast<Instruction>(NAME#_.ZSuffix#rmi) _.RC:$src1, _.RC:$src2,
                                   addr:$src3, (VPTERNLOG132_imm8 imm:$src4))>;

  // Additional patterns for matching zero masking with loads in other
  // positions.
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode (bitconvert (_.LdFrag addr:$src3)),
                    _.RC:$src2, _.RC:$src1, (i8 imm:$src4)),
                   _.ImmAllZerosV)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmikz) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, addr:$src3, (VPTERNLOG321_imm8 imm:$src4))>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode _.RC:$src1, (bitconvert (_.LdFrag addr:$src3)),
                    _.RC:$src2, (i8 imm:$src4)),
                   _.ImmAllZerosV)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmikz) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, addr:$src3, (VPTERNLOG132_imm8 imm:$src4))>;

  // Additional patterns for matching masked loads with different
  // operand orders.
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode _.RC:$src1, (bitconvert (_.LdFrag addr:$src3)),
                    _.RC:$src2, (i8 imm:$src4)),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmik) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, addr:$src3, (VPTERNLOG132_imm8 imm:$src4))>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode (bitconvert (_.LdFrag addr:$src3)),
                    _.RC:$src2, _.RC:$src1, (i8 imm:$src4)),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmik) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, addr:$src3, (VPTERNLOG321_imm8 imm:$src4))>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode _.RC:$src2, _.RC:$src1,
                    (bitconvert (_.LdFrag addr:$src3)), (i8 imm:$src4)),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmik) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, addr:$src3, (VPTERNLOG213_imm8 imm:$src4))>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode _.RC:$src2, (bitconvert (_.LdFrag addr:$src3)),
                    _.RC:$src1, (i8 imm:$src4)),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmik) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, addr:$src3, (VPTERNLOG231_imm8 imm:$src4))>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode (bitconvert (_.LdFrag addr:$src3)),
                    _.RC:$src1, _.RC:$src2, (i8 imm:$src4)),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmik) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, addr:$src3, (VPTERNLOG312_imm8 imm:$src4))>;

  // Additional patterns for matching broadcasts in other positions.
  def : Pat<(_.VT (OpNode (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                          _.RC:$src2, _.RC:$src1, (i8 imm:$src4))),
            (!cast<Instruction>(NAME#_.ZSuffix#rmbi) _.RC:$src1, _.RC:$src2,
                                   addr:$src3, (VPTERNLOG321_imm8 imm:$src4))>;
  def : Pat<(_.VT (OpNode _.RC:$src1,
                          (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                          _.RC:$src2, (i8 imm:$src4))),
            (!cast<Instruction>(NAME#_.ZSuffix#rmbi) _.RC:$src1, _.RC:$src2,
                                   addr:$src3, (VPTERNLOG132_imm8 imm:$src4))>;

  // Additional patterns for matching zero masking with broadcasts in other
  // positions.
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                    _.RC:$src2, _.RC:$src1, (i8 imm:$src4)),
                   _.ImmAllZerosV)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmbikz) _.RC:$src1,
             _.KRCWM:$mask, _.RC:$src2, addr:$src3,
             (VPTERNLOG321_imm8 imm:$src4))>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode _.RC:$src1,
                    (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                    _.RC:$src2, (i8 imm:$src4)),
                   _.ImmAllZerosV)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmbikz) _.RC:$src1,
             _.KRCWM:$mask, _.RC:$src2, addr:$src3,
             (VPTERNLOG132_imm8 imm:$src4))>;

  // Additional patterns for matching masked broadcasts with different
  // operand orders.
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode _.RC:$src1,
                    (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                    _.RC:$src2, (i8 imm:$src4)),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmbik) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, addr:$src3, (VPTERNLOG132_imm8 imm:$src4))>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                    _.RC:$src2, _.RC:$src1, (i8 imm:$src4)),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmik) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, addr:$src3, (VPTERNLOG321_imm8 imm:$src4))>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode _.RC:$src2, _.RC:$src1,
                    (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                    (i8 imm:$src4)), _.RC:$src1)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmik) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, addr:$src3, (VPTERNLOG213_imm8 imm:$src4))>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode _.RC:$src2,
                    (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                    _.RC:$src1, (i8 imm:$src4)),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmik) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, addr:$src3, (VPTERNLOG231_imm8 imm:$src4))>;
  def : Pat<(_.VT (vselect _.KRCWM:$mask,
                   (OpNode (X86VBroadcast (_.ScalarLdFrag addr:$src3)),
                    _.RC:$src1, _.RC:$src2, (i8 imm:$src4)),
                   _.RC:$src1)),
            (!cast<Instruction>(NAME#_.ZSuffix#rmik) _.RC:$src1, _.KRCWM:$mask,
             _.RC:$src2, addr:$src3, (VPTERNLOG312_imm8 imm:$src4))>;
}

multiclass avx512_common_ternlog<string OpcodeStr, AVX512VLVectorVTInfo _>{
  let Predicates = [HasAVX512] in
    defm Z    : avx512_ternlog<0x25, OpcodeStr, X86vpternlog, _.info512>, EVEX_V512;
  let Predicates = [HasAVX512, HasVLX] in {
    defm Z128 : avx512_ternlog<0x25, OpcodeStr, X86vpternlog, _.info128>, EVEX_V128;
    defm Z256 : avx512_ternlog<0x25, OpcodeStr, X86vpternlog, _.info256>, EVEX_V256;
  }
}

defm VPTERNLOGD : avx512_common_ternlog<"vpternlogd", avx512vl_i32_info>;
defm VPTERNLOGQ : avx512_common_ternlog<"vpternlogq", avx512vl_i64_info>, VEX_W;

//===----------------------------------------------------------------------===//
// AVX-512 - FixupImm
//===----------------------------------------------------------------------===//

multiclass avx512_fixupimm_packed<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  X86VectorVTInfo _>{
  let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in {
    defm rri : AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
                        (ins _.RC:$src2, _.RC:$src3, i32u8imm:$src4),
                         OpcodeStr##_.Suffix, "$src4, $src3, $src2", "$src2, $src3, $src4",
                        (OpNode (_.VT _.RC:$src1),
                                (_.VT _.RC:$src2),
                                (_.IntVT _.RC:$src3),
                                (i32 imm:$src4),
                                (i32 FROUND_CURRENT))>;
    defm rmi : AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
                      (ins _.RC:$src2, _.MemOp:$src3, i32u8imm:$src4),
                      OpcodeStr##_.Suffix, "$src4, $src3, $src2", "$src2, $src3, $src4",
                      (OpNode (_.VT _.RC:$src1),
                              (_.VT _.RC:$src2),
                              (_.IntVT (bitconvert (_.LdFrag addr:$src3))),
                              (i32 imm:$src4),
                              (i32 FROUND_CURRENT))>;
    defm rmbi : AVX512_maskable_3src<opc, MRMSrcMem, _, (outs _.RC:$dst),
                      (ins _.RC:$src2, _.ScalarMemOp:$src3, i32u8imm:$src4),
                    OpcodeStr##_.Suffix, "$src4, ${src3}"##_.BroadcastStr##", $src2",
                    "$src2, ${src3}"##_.BroadcastStr##", $src4",
                      (OpNode (_.VT _.RC:$src1),
                              (_.VT _.RC:$src2),
                              (_.IntVT (X86VBroadcast(_.ScalarLdFrag addr:$src3))),
                              (i32 imm:$src4),
                              (i32 FROUND_CURRENT))>, EVEX_B;
  } // Constraints = "$src1 = $dst"
}

multiclass avx512_fixupimm_packed_sae<bits<8> opc, string OpcodeStr,
                                      SDNode OpNode, X86VectorVTInfo _>{
let Constraints = "$src1 = $dst", ExeDomain = _.ExeDomain in {
  defm rrib : AVX512_maskable_3src<opc, MRMSrcReg, _, (outs _.RC:$dst),
                      (ins _.RC:$src2, _.RC:$src3, i32u8imm:$src4),
                      OpcodeStr##_.Suffix, "$src4, {sae}, $src3, $src2",
                      "$src2, $src3, {sae}, $src4",
                      (OpNode (_.VT _.RC:$src1),
                                (_.VT _.RC:$src2),
                                (_.IntVT _.RC:$src3),
                                (i32 imm:$src4),
                                (i32 FROUND_NO_EXC))>, EVEX_B;
  }
}

multiclass avx512_fixupimm_scalar<bits<8> opc, string OpcodeStr, SDNode OpNode,
                                  X86VectorVTInfo _, X86VectorVTInfo _src3VT> {
  let Constraints = "$src1 = $dst" , Predicates = [HasAVX512],
      ExeDomain = _.ExeDomain in {
    defm rri : AVX512_maskable_3src_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                      (ins _.RC:$src2, _.RC:$src3, i32u8imm:$src4),
                      OpcodeStr##_.Suffix, "$src4, $src3, $src2", "$src2, $src3, $src4",
                      (OpNode (_.VT _.RC:$src1),
                              (_.VT _.RC:$src2),
                              (_src3VT.VT _src3VT.RC:$src3),
                              (i32 imm:$src4),
                              (i32 FROUND_CURRENT))>;

    defm rrib : AVX512_maskable_3src_scalar<opc, MRMSrcReg, _, (outs _.RC:$dst),
                      (ins _.RC:$src2, _.RC:$src3, i32u8imm:$src4),
                      OpcodeStr##_.Suffix, "$src4, {sae}, $src3, $src2",
                      "$src2, $src3, {sae}, $src4",
                      (OpNode (_.VT _.RC:$src1),
                              (_.VT _.RC:$src2),
                              (_src3VT.VT _src3VT.RC:$src3),
                              (i32 imm:$src4),
                              (i32 FROUND_NO_EXC))>, EVEX_B;
    defm rmi : AVX512_maskable_3src_scalar<opc, MRMSrcMem, _, (outs _.RC:$dst),
                     (ins _.RC:$src2, _.ScalarMemOp:$src3, i32u8imm:$src4),
                     OpcodeStr##_.Suffix, "$src4, $src3, $src2", "$src2, $src3, $src4",
                     (OpNode (_.VT _.RC:$src1),
                             (_.VT _.RC:$src2),
                             (_src3VT.VT (scalar_to_vector
                                       (_src3VT.ScalarLdFrag addr:$src3))),
                             (i32 imm:$src4),
                             (i32 FROUND_CURRENT))>;
  }
}

multiclass avx512_fixupimm_packed_all<AVX512VLVectorVTInfo _Vec>{
  let Predicates = [HasAVX512] in
    defm Z    : avx512_fixupimm_packed<0x54, "vfixupimm", X86VFixupimm, _Vec.info512>,
                avx512_fixupimm_packed_sae<0x54, "vfixupimm", X86VFixupimm, _Vec.info512>,
                                  AVX512AIi8Base, EVEX_4V, EVEX_V512;
  let Predicates = [HasAVX512, HasVLX] in {
    defm Z128 : avx512_fixupimm_packed<0x54, "vfixupimm", X86VFixupimm, _Vec.info128>,
                                  AVX512AIi8Base, EVEX_4V, EVEX_V128;
    defm Z256 : avx512_fixupimm_packed<0x54, "vfixupimm", X86VFixupimm, _Vec.info256>,
                                  AVX512AIi8Base, EVEX_4V, EVEX_V256;
  }
}

defm VFIXUPIMMSS : avx512_fixupimm_scalar<0x55, "vfixupimm", X86VFixupimmScalar,
                                          f32x_info, v4i32x_info>,
                         AVX512AIi8Base, VEX_LIG, EVEX_4V, EVEX_CD8<32, CD8VT1>;
defm VFIXUPIMMSD : avx512_fixupimm_scalar<0x55, "vfixupimm", X86VFixupimmScalar,
                                          f64x_info, v2i64x_info>,
                         AVX512AIi8Base, VEX_LIG, EVEX_4V, EVEX_CD8<64, CD8VT1>, VEX_W;
defm VFIXUPIMMPS : avx512_fixupimm_packed_all<avx512vl_f32_info>,
                         EVEX_CD8<32, CD8VF>;
defm VFIXUPIMMPD : avx512_fixupimm_packed_all<avx512vl_f64_info>,
                         EVEX_CD8<64, CD8VF>, VEX_W;



// Patterns used to select SSE scalar fp arithmetic instructions from
// either:
//
// (1) a scalar fp operation followed by a blend
//
// The effect is that the backend no longer emits unnecessary vector
// insert instructions immediately after SSE scalar fp instructions
// like addss or mulss.
//
// For example, given the following code:
//   __m128 foo(__m128 A, __m128 B) {
//     A[0] += B[0];
//     return A;
//   }
//
// Previously we generated:
//   addss %xmm0, %xmm1
//   movss %xmm1, %xmm0
//
// We now generate:
//   addss %xmm1, %xmm0
//
// (2) a vector packed single/double fp operation followed by a vector insert
//
// The effect is that the backend converts the packed fp instruction
// followed by a vector insert into a single SSE scalar fp instruction.
//
// For example, given the following code:
//   __m128 foo(__m128 A, __m128 B) {
//     __m128 C = A + B;
//     return (__m128) {c[0], a[1], a[2], a[3]};
//   }
//
// Previously we generated:
//   addps %xmm0, %xmm1
//   movss %xmm1, %xmm0
//
// We now generate:
//   addss %xmm1, %xmm0

// TODO: Some canonicalization in lowering would simplify the number of
// patterns we have to try to match.
multiclass AVX512_scalar_math_f32_patterns<SDNode Op, string OpcPrefix> {
  let Predicates = [HasAVX512] in {
    // extracted scalar math op with insert via movss
    def : Pat<(v4f32 (X86Movss (v4f32 VR128X:$dst), (v4f32 (scalar_to_vector
          (Op (f32 (extractelt (v4f32 VR128X:$dst), (iPTR 0))),
          FR32X:$src))))),
      (!cast<I>("V"#OpcPrefix#SSZrr_Int) v4f32:$dst,
          (COPY_TO_REGCLASS FR32X:$src, VR128X))>;

    // extracted scalar math op with insert via blend
    def : Pat<(v4f32 (X86Blendi (v4f32 VR128X:$dst), (v4f32 (scalar_to_vector
          (Op (f32 (extractelt (v4f32 VR128X:$dst), (iPTR 0))),
          FR32X:$src))), (i8 1))),
      (!cast<I>("V"#OpcPrefix#SSZrr_Int) v4f32:$dst,
          (COPY_TO_REGCLASS FR32X:$src, VR128X))>;

    // vector math op with insert via movss
    def : Pat<(v4f32 (X86Movss (v4f32 VR128X:$dst),
          (Op (v4f32 VR128X:$dst), (v4f32 VR128X:$src)))),
      (!cast<I>("V"#OpcPrefix#SSZrr_Int) v4f32:$dst, v4f32:$src)>;

    // vector math op with insert via blend
    def : Pat<(v4f32 (X86Blendi (v4f32 VR128X:$dst),
          (Op (v4f32 VR128X:$dst), (v4f32 VR128X:$src)), (i8 1))),
      (!cast<I>("V"#OpcPrefix#SSZrr_Int) v4f32:$dst, v4f32:$src)>;

    // extracted masked scalar math op with insert via movss
    def : Pat<(X86Movss (v4f32 VR128X:$src1),
               (scalar_to_vector
                (X86selects VK1WM:$mask,
                            (Op (f32 (extractelt (v4f32 VR128X:$src1), (iPTR 0))),
                                FR32X:$src2),
                            FR32X:$src0))),
      (!cast<I>("V"#OpcPrefix#SSZrr_Intk) (COPY_TO_REGCLASS FR32X:$src0, VR128X),
          VK1WM:$mask, v4f32:$src1,
          (COPY_TO_REGCLASS FR32X:$src2, VR128X))>;
  }
}

defm : AVX512_scalar_math_f32_patterns<fadd, "ADD">;
defm : AVX512_scalar_math_f32_patterns<fsub, "SUB">;
defm : AVX512_scalar_math_f32_patterns<fmul, "MUL">;
defm : AVX512_scalar_math_f32_patterns<fdiv, "DIV">;

multiclass AVX512_scalar_math_f64_patterns<SDNode Op, string OpcPrefix> {
  let Predicates = [HasAVX512] in {
    // extracted scalar math op with insert via movsd
    def : Pat<(v2f64 (X86Movsd (v2f64 VR128X:$dst), (v2f64 (scalar_to_vector
          (Op (f64 (extractelt (v2f64 VR128X:$dst), (iPTR 0))),
          FR64X:$src))))),
      (!cast<I>("V"#OpcPrefix#SDZrr_Int) v2f64:$dst,
          (COPY_TO_REGCLASS FR64X:$src, VR128X))>;

    // extracted scalar math op with insert via blend
    def : Pat<(v2f64 (X86Blendi (v2f64 VR128X:$dst), (v2f64 (scalar_to_vector
          (Op (f64 (extractelt (v2f64 VR128X:$dst), (iPTR 0))),
          FR64X:$src))), (i8 1))),
      (!cast<I>("V"#OpcPrefix#SDZrr_Int) v2f64:$dst,
          (COPY_TO_REGCLASS FR64X:$src, VR128X))>;

    // vector math op with insert via movsd
    def : Pat<(v2f64 (X86Movsd (v2f64 VR128X:$dst),
          (Op (v2f64 VR128X:$dst), (v2f64 VR128X:$src)))),
      (!cast<I>("V"#OpcPrefix#SDZrr_Int) v2f64:$dst, v2f64:$src)>;

    // vector math op with insert via blend
    def : Pat<(v2f64 (X86Blendi (v2f64 VR128X:$dst),
          (Op (v2f64 VR128X:$dst), (v2f64 VR128X:$src)), (i8 1))),
      (!cast<I>("V"#OpcPrefix#SDZrr_Int) v2f64:$dst, v2f64:$src)>;

    // extracted masked scalar math op with insert via movss
    def : Pat<(X86Movsd (v2f64 VR128X:$src1),
               (scalar_to_vector
                (X86selects VK1WM:$mask,
                            (Op (f64 (extractelt (v2f64 VR128X:$src1), (iPTR 0))),
                                FR64X:$src2),
                            FR64X:$src0))),
      (!cast<I>("V"#OpcPrefix#SDZrr_Intk) (COPY_TO_REGCLASS FR64X:$src0, VR128X),
          VK1WM:$mask, v2f64:$src1,
          (COPY_TO_REGCLASS FR64X:$src2, VR128X))>;
  }
}

defm : AVX512_scalar_math_f64_patterns<fadd, "ADD">;
defm : AVX512_scalar_math_f64_patterns<fsub, "SUB">;
defm : AVX512_scalar_math_f64_patterns<fmul, "MUL">;
defm : AVX512_scalar_math_f64_patterns<fdiv, "DIV">;
