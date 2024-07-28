//===-- VOPMInstructions.td - Vector Instruction Definitions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO-GFX13: Add support for global-isel.

//===----------------------------------------------------------------------===//
// VOPM Encodings
//===----------------------------------------------------------------------===//

class VOPMProfile<list<ValueType> ArgTy, bit _HasAux,
                                         bit _HasClamp,
                                         bit _HasSSrc,
                                         bit _HasSrc4> : VOPProfile<ArgTy> {
  bit HasAux = _HasAux;
  let HasClamp = _HasClamp;
  bit HasSSrc = _HasSSrc;

  bit HasSrc4 = _HasSrc4;

  let HasModifiers = 0;
  let HasOMod = 0;

  // Handle 10-bit source in analogous way to vdst.
  let Src2RC64 = getVALUDstForVT<ArgVT[3]>.ret;
  field ValueType Src3VT = ArgVT[4];
  field RegisterOperand Src3RC64 = getVALUDstForVT<Src3VT>.ret;
}

class VOP2Me <bits<8> op, VOPMProfile pfl> : Enc64 {
  bits<10> vdst;
  bits<11> src0;
  bits<11> src1;
  bits<6> aux_data;

  // TODO-GFX13: Inst{9-0} = vdst;
  let Inst{7-0} = vdst{7-0};
  let Inst{11-10} = ?;    // vdst_idx
  let Inst{13} = ?;       // supr

  let Inst{22-15} = op;
  let Inst{31-24} = 0xdc; // encoding

  // TODO-GFX13: Inst{42-32} = src0;
  let Inst{39-32} = src0{7-0};
  let Inst{42} = src0{8};
  let Inst{44-43} = ?;    // src0_idx

  // TODO-GFX13: Inst{55-45} = !if(pfl.HasSrc1, src1, ?);
  let Inst{52-45} = !if(pfl.HasSrc1, src1{7-0}, ?);
  let Inst{55} = !if(pfl.HasSrc1, src1{8}, ?);
  let Inst{57-56} = ?;    // src1_idx

  let Inst{63-58} = !if(pfl.HasAux, aux_data, ?);
}

class VOP3Me <bits<8> op, VOPMProfile pfl> : Enc96 {

  bits<10> vdst;
  bits<11> src0;
  bits<11> src1;
  bits<10> src2;
  bits<8> ssrc;
  bits<18> aux_data;

  bit clamp;

  // TODO-GFX13: Inst{9-0} = vdst;
  let Inst{7-0} = vdst{7-0};
  let Inst{9-8} = ?;      // vdst cont
  let Inst{11-10} = ?;    // vdst_idx

  let Inst{12} = !if(pfl.HasClamp, clamp, ?);
  let Inst{13} = ?;       // supr

  let Inst{22-15} = op;
  let Inst{31-24} = 0xdd; // encoding

  // TODO-GFX13: Inst{42-32} = src0;
  let Inst{39-32} = src0{7-0};
  let Inst{42} = src0{8};
  let Inst{44-43} = ?;    // src0_idx

  // TODO-GFX13: Inst{55-45} = !if(pfl.HasSrc1, src1, ?);
  let Inst{52-45} = !if(pfl.HasSrc1, src1{7-0}, ?);
  let Inst{55} = !if(pfl.HasSrc1, src1{8}, ?);
  let Inst{57-56} = ?;    // src1_idx

  let Inst{63-58} = !if(pfl.HasAux, aux_data{5-0}, ?); // mod0

  // TODO-GFX13: Inst{73-64} = !if(pfl.HasSrc2, src2, ?);
  let Inst{71-64} = !if(pfl.HasSrc2, src2{7-0}, ?);
  let Inst{75-74} = ?;    // src2_idx

  let Inst{87-76} = !if(pfl.HasAux, aux_data{17-6}, ?); // mod1

  let Inst{95-88} = !if(pfl.HasSSrc, ssrc, ?);
}

class VOP5Me <bits<8> op, VOPMProfile pfl> : Enc128 {
  bits<10> vdst;
  bits<11> src0;
  bits<11> src1;
  bits<10> src2;
  bits<10> src3;
  bits<10> src4;
  bits<8> ssrc;
  bits<26> aux_data;

  bit clamp;

  let Inst{7-0} = vdst{7-0};
  let Inst{9-8} = ?;      // vdst cont
  let Inst{11-10} = ?;    // vdst_idx
  let Inst{12} = !if(pfl.HasClamp, clamp, ?);
  let Inst{13} = ?;       // supr

  let Inst{22-15} = op;
  let Inst{31-24} = 0xde; // encoding

  // TODO-GFX13: Inst{42-32} = src0;
  let Inst{39-32} = src0{7-0};
  let Inst{42} = src0{8};
  let Inst{44-43} = ?;    // src0_idx

  // TODO-GFX13: Inst{55-45} = !if(pfl.HasSrc1, src1, ?);
  let Inst{52-45} = !if(pfl.HasSrc1, src1{7-0}, ?);
  let Inst{55} = !if(pfl.HasSrc1, src1{8}, ?);
  let Inst{57-56} = ?;    // src1_idx

  let Inst{63-58} = !if(pfl.HasAux, aux_data{5-0}, ?); // mod0

  // TODO-GFX13: Inst{73-64} = src2;
  let Inst{71-64} = src2{7-0};
  let Inst{75-74} = ?;    // src2_idx

  let Inst{87-76} = !if(pfl.HasAux, aux_data{17-6}, ?); // mod1

  let Inst{95-88} = !if(pfl.HasSSrc, ssrc, ?);

  // TODO-GFX13: Inst{105-96} = src3;
  let Inst{103-96} = src3{7-0};
  let Inst{105} = ?;     // src3 cont
  let Inst{107-106} = ?; // src3_idx

  // TODO-GFX13: Inst{117-108} = src4;
  let Inst{115-108} = !if(pfl.HasSrc4, src4{7-0}, ?);
  let Inst{117} = ?;     // src4 cont
  let Inst{119-118} = ?; // src4_idx

  let Inst{127-120} = !if(pfl.HasAux, aux_data{25-18}, ?); // mod2
}

class VOP6Me <bits<8> op, VOPMProfile pfl> : Enc128 {
  bits<10> vdst;
  bits<11> src0;
  bits<11> src1;
  bits<10> src2;
  bits<10> src3;
  bits<10> src4;
  bits<10> src5;
  bits<26> aux_data;

  bit clamp;

  let Inst{7-0} = vdst{7-0};
  let Inst{9-8} = ?;      // vdst cont
  let Inst{11-10} = ?;    // vdst_idx
  let Inst{12} = !if(pfl.HasClamp, clamp, ?);
  let Inst{13} = ?;       // supr

  let Inst{22-15} = op;
  let Inst{31-24} = 0xdf; // encoding

  // TODO-GFX13: Inst{42-32} = src0;
  let Inst{39-32} = src0{7-0};
  let Inst{42} = src0{8};
  let Inst{44-43} = ?;    // src0_idx

  // TODO-GFX13: Inst{55-45} = !if(pfl.HasSrc1, src1, ?);
  let Inst{52-45} = src1{7-0};
  let Inst{55} = src1{8};
  let Inst{57-56} = ?;    // src1_idx

  let Inst{63-58} = !if(pfl.HasAux, aux_data{5-0}, ?); // mod0

  // TODO-GFX13: Inst{73-64} = src2;
  let Inst{71-64} = src2{7-0};
  let Inst{75-74} = ?;    // src2_idx

  let Inst{83-76} = !if(pfl.HasAux, aux_data{13-6}, ?); // mod1

  // TODO-GFX13: Inst{93-84} = src5;
  let Inst{91-84} = src5{7-0};
  let Inst{93} = ?;     // src5 cont
  let Inst{95-94} = ?;   // src5_idx

  // TODO-GFX13: Inst{105-96} = src3;
  let Inst{103-96} = src3{7-0};
  let Inst{105} = ?;     // src3 cont
  let Inst{107-106} = ?; // src3_idx

  // TODO-GFX13: Inst{117-108} = src4;
  let Inst{115-108} = !if(pfl.HasSrc4, src4{7-0}, ?);
  let Inst{117} = ?;     // src4 cont
  let Inst{119-118} = ?; // src4_idx

  let Inst{127-120} = !if(pfl.HasAux, aux_data{25-18}, ?); // mod2
}

//===----------------------------------------------------------------------===//
// VOPM Pseudos
//===----------------------------------------------------------------------===//

def PixelShape {
  int 8X4X8 = 0;
  int 4X4X8 = 1;
  int 4X4X16 = 2;
  int 4X2X16 = 3;
}

def FilterType {
  int 1X1 = 0;
  int 3X3 = 1;
}

def IterationCnt {
  int IT_1 = 0;
  int IT_2 = 1;
  int IT_3 = 2;
  int IT_4 = 3;
}

class VOP2M_Profile<list<ValueType> ArgTy, bit _HasAux = 0> : VOPMProfile<ArgTy, _HasAux, 0, 0, 0> {
  bit HasAux = _HasAux;

  let Src1RC64 = !if(!eq(Src1VT, i64), SCSrc_b64, VSrc_f32);
  let Ins64 = !con((ins Src0RC64:$src0), !if(HasSrc1, (ins Src1RC64:$src1), (ins)), !if(HasAux, (ins AuxData:$aux_data), (ins)));

  string Src1Asm = !if(HasSrc1, ", $src1", "");
  string AuxDataAsm = !if(HasAux, "$aux_data", "");

  let Asm64 = "$vdst, $src0"#Src1Asm#AuxDataAsm;
}

def VOPM_I32_I32_I32 : VOP2M_Profile<[i32, i32, i32, untyped]>;
def VOPM_I32_I32_I32_AUX : VOP2M_Profile<[i32, i32, i32, untyped], 1>;
def VOPM_I32_I32_AUX : VOP2M_Profile<[i32, i32, untyped, untyped], 1>;
def VOPM_I32_I32_I64 : VOP2M_Profile<[i32, i32, i64, untyped]>;

class VOP3M_Permute_Profile<list<ValueType> ArgTy> : VOPMProfile<ArgTy, 1, 0, 0, 0> {
  let Outs64 = (outs DstRC64:$vdst, Src2RC64:$src2);
  let Ins64 = (ins Src0RC64:$src0, Src1RC64:$src1, AuxData:$aux_data);

  let Asm64 = "$vdst, $src2, $src0, $src1$aux_data";
}

def VOPM_PERMUTE_I32_I32_I32_I32 : VOP3M_Permute_Profile<[i32, i32, i32, i32]>;

class Conv_Profile<list<ValueType> ArgTy, int Shape, int Filter, int Iterations> : VOPMProfile<ArgTy, 1, 1, 0, 1> {
  let Outs64 = (outs DstRC64:$vdst);
  let Ins64 = !if(!eq(Filter,FilterType.3X3), (ins Src0RC64:$src0, Src1RC64:$src1, Src2RC64:$src2, Src2RC64:$src3, Src2RC64:$src4, AuxData:$aux_data, Clamp0:$clamp),
                                              (ins Src0RC64:$src0, Src1RC64:$src1, Src2RC64:$src2, Src2RC64:$src3, Src2RC64:$src4, Src2RC64:$src5, AuxData:$aux_data, Clamp0:$clamp));

  dag ConvPat = !if(!eq(Filter,FilterType.3X3), (ins Src0VT:$src0, Src1VT:$src1, Src2VT:$src2, Src2VT:$src3, Src2VT:$src4, timm:$aux_data, i1:$clamp),
                                                (ins Src0VT:$src0, Src1VT:$src1, Src2VT:$src2, Src2VT:$src3, Src2VT:$src4, Src2VT:$src5, timm:$aux_data, i1:$clamp));

  dag ConvFrag = !if(!eq(Filter,FilterType.3X3), (ops node:$src0, node:$src1, node:$src2, node:$src3, node:$src4, node:$aux_data, node:$clamp),
                                                  (ops node:$src0, node:$src1, node:$src2, node:$src3, node:$src4, node:$src5, node:$aux_data, node:$clamp));

  let Asm64 = !if(!eq(Filter,FilterType.3X3), "$vdst, $src0, $src1, $src2, $src3, $src4$aux_data$clamp",
                                              !cond(!eq(Iterations, IterationCnt.IT_1) : "$vdst, $src0, $src1, $src2$aux_data$clamp",
                                                    !eq(Iterations, IterationCnt.IT_2) : "$vdst, $src0, $src1, $src2, $src3$aux_data$clamp",
                                                    !eq(Iterations, IterationCnt.IT_3) : "$vdst, $src0, $src1, $src2, $src3, $src4$aux_data$clamp",
                                                    !eq(Iterations, IterationCnt.IT_4) : "$vdst, $src0, $src1, $src2, $src3, $src4, $src5$aux_data$clamp"));

  int PrimTy = Shape;
  int FilterTy = Filter;
  int IterCnt = Iterations;
}

multiclass Conv_Profile_Reg<int AccReg, int WeightsReg, int TensorReg, string OutputBaseTy, string AccInBaseTy, string WeightsTensorBaseTy, int Shape, int Filter, int Iterations = 0> {

  defvar AMult = !if(!eq(!find(AccInBaseTy, "16"), -1), 1, 2);
  defvar WTMult = !if(!eq(!find(WeightsTensorBaseTy, "16"), -1), 1, 2);

  defvar AccElCount = !mul(AMult, AccReg);

  defvar TmpWeightsElCount = !mul(WTMult, WeightsReg);
  // TODO-GFX13: v12f16/v12bf16 is not supported in the backend yet, use v16 for now.
  defvar WeightsElCount = !if(!eq(TmpWeightsElCount, 12), 16, TmpWeightsElCount);

  defvar TensorElCount = !mul(WTMult, TensorReg);

  defvar AccElPrefix = !if(!eq(AccElCount, 1), "", "v" # AccElCount);
  defvar WeightsElPrefix = !if(!eq(WeightsElCount, 1), "" , "v" # WeightsElCount);
  defvar TensorElPrefix = !if(!eq(TensorElCount, 1), "", "v" # TensorElCount);

  defvar OutputFullTy = !cast<ValueType>(AccElPrefix # OutputBaseTy);
  defvar AccInFullTy = !cast<ValueType>(AccElPrefix # AccInBaseTy);
  defvar WeightsFullTy = !cast<ValueType>(WeightsElPrefix # WeightsTensorBaseTy);
  defvar TensorFullTy = !cast<ValueType>(TensorElPrefix # WeightsTensorBaseTy);

  def NAME : Conv_Profile<[OutputFullTy, AccInFullTy, WeightsFullTy, TensorFullTy], Shape, Filter, Iterations>;
}

class getWeightsRegPerIter<int Per4Iter, int Iter> {
  int ret = !div(!add(!mul(Per4Iter, Iter), 3), 4);
}

multiclass Conv_Profile_1x1_Multi_Iter<int AccReg, int WeightsRegPer4Iter, int TensorReg, string OutputBaseTy, string AccInBaseTy, string WeightsTensorBaseTy, int Shape, int Filter> {
  defm _ITER_1 : Conv_Profile_Reg<AccReg, getWeightsRegPerIter<WeightsRegPer4Iter, 1>.ret, TensorReg, OutputBaseTy, AccInBaseTy, WeightsTensorBaseTy, Shape, Filter, IterationCnt.IT_1>;
  defm _ITER_2 : Conv_Profile_Reg<AccReg, getWeightsRegPerIter<WeightsRegPer4Iter, 2>.ret, TensorReg, OutputBaseTy, AccInBaseTy, WeightsTensorBaseTy, Shape, Filter, IterationCnt.IT_2>;
  defm _ITER_3 : Conv_Profile_Reg<AccReg, getWeightsRegPerIter<WeightsRegPer4Iter, 3>.ret, TensorReg, OutputBaseTy, AccInBaseTy, WeightsTensorBaseTy, Shape, Filter, IterationCnt.IT_3>;
  defm _ITER_4 : Conv_Profile_Reg<AccReg, getWeightsRegPerIter<WeightsRegPer4Iter, 4>.ret, TensorReg, OutputBaseTy, AccInBaseTy, WeightsTensorBaseTy, Shape, Filter, IterationCnt.IT_4>;
}

multiclass Conv_Profile_Multi<string OutputBaseTy, string WeightsTensorBaseTy, string AccInBaseTy = OutputBaseTy>{

  // 4x2 with 16-bit accumulators require 2 registers for accumulator/output, not 4.
  defvar Acc_Reg_4x2 = !if(!eq(!find(AccInBaseTy, "16"), -1), 4, 2);

  defm _3x3_8x4 : Conv_Profile_Reg<4, 5, 4, OutputBaseTy, AccInBaseTy, WeightsTensorBaseTy, PixelShape.8X4X8, FilterType.3X3>;
  defm _3x3_4x4 : Conv_Profile_Reg<4, 9, 3, OutputBaseTy, AccInBaseTy, WeightsTensorBaseTy, PixelShape.4X4X16, FilterType.3X3>;
  defm _3x3_4x2 : Conv_Profile_Reg<Acc_Reg_4x2, 18, 3, OutputBaseTy, AccInBaseTy, WeightsTensorBaseTy, PixelShape.4X2X16, FilterType.3X3>;

  defm _1x1_8x4 : Conv_Profile_1x1_Multi_Iter<4, 2, 2, OutputBaseTy, AccInBaseTy, WeightsTensorBaseTy, PixelShape.8X4X8, FilterType.1X1>;
  defm _1x1_4x4 : Conv_Profile_1x1_Multi_Iter<4, 4, 1, OutputBaseTy, AccInBaseTy, WeightsTensorBaseTy, PixelShape.4X4X16, FilterType.1X1>;
  defm _1x1_4x2 : Conv_Profile_1x1_Multi_Iter<Acc_Reg_4x2, 8, 1, OutputBaseTy, AccInBaseTy, WeightsTensorBaseTy, PixelShape.4X2X16, FilterType.1X1>;
}

//                                             dest    weights/tensor
defm VOPM_CONV_F16_F16   : Conv_Profile_Multi<"f16", "f16">;
defm VOPM_CONV_BF16_BF16 : Conv_Profile_Multi<"bf16", "bf16">;

defm VOPM_CONV_F16_FP8   : Conv_Profile_Multi<"f16", "i32">;
defm VOPM_CONV_BF16_BF8  : Conv_Profile_Multi<"bf16", "i32">;
defm VOPM_CONV_F16_IU8   : Conv_Profile_Multi<"f16", "i32">;
defm VOPM_CONV_F16_IU4   : Conv_Profile_Multi<"f16", "i32">;

defm VOPM_CONV_F32_F16   : Conv_Profile_Multi<"f32", "f16">;
defm VOPM_CONV_F32_BF16  : Conv_Profile_Multi<"f32", "bf16">;
defm VOPM_CONV_F32_FP8   : Conv_Profile_Multi<"f32", "i32">;
defm VOPM_CONV_F32_BF8   : Conv_Profile_Multi<"f32", "i32">;
defm VOPM_CONV_F32_IU8   : Conv_Profile_Multi<"f32", "i32">;
defm VOPM_CONV_F32_IU4   : Conv_Profile_Multi<"f32", "i32">;
defm VOPM_CONV_I32_IU8   : Conv_Profile_Multi<"i32", "i32">;
defm VOPM_CONV_I32_IU4   : Conv_Profile_Multi<"i32", "i32">;

//                                              dest   weights/tensor acc_in
defm VOPM_CONV_F32I32_IU8 : Conv_Profile_Multi<"f32", "i32", "i32">;
defm VOPM_CONV_F32I32_IU4 : Conv_Profile_Multi<"f32", "i32", "i32">;

class SBA_Profile<list<ValueType> ArgTy, int Scatter> : VOPMProfile<ArgTy, 1, 1, 1, 1> {

  let Outs64 = !cond(!eq(Scatter, 1) : (outs DstRC64:$vdst),
                     !eq(Scatter, 2) : (outs DstRC64:$vdst, Src2RC64:$src2),
                     !eq(Scatter, 4) : (outs DstRC64:$vdst, Src2RC64:$src2, Src2RC64:$src3, Src2RC64:$src4));

  // Assuming ssrc supports float constants (SSrc_f32).
  let Ins64 = !if(HasSrc1, (ins Src0RC64:$src0, SSrc_f32:$ssrc, Src1RC64:$src1, AuxData:$aux_data, Clamp0:$clamp),
                           (ins Src0RC64:$src0, SSrc_f32:$ssrc, AuxData:$aux_data, Clamp0:$clamp));

  string DestAsm = !cond(!eq(Scatter, 1) : "$vdst",
                         !eq(Scatter, 2) : "$vdst, $src2",
                         !eq(Scatter, 4) : "$vdst, $src2, $src3, $src4");
  string SrcAsm = !if(HasSrc1, "$src0, $ssrc, $src1$aux_data$clamp",
                               "$src0, $ssrc$aux_data$clamp");
  let Asm64 = DestAsm#", "#SrcAsm;

  bit PixelShape4x2 = !or(!eq(Scatter, 2), !eq(ArgTy[0], v4f32));
}

def KMult {
  int X1 = 0;
  int X2 = 1;
  int X3 = 2;
  int X4 = 3;
}

class WMMA_SWMMAC_Profile<list<ValueType> ArgTy, int K>
    : VOPMProfile<ArgTy, 0, 1, 0, 1> {
  int KMultiplier = K;
}

class VOPM_WMMA_Profile<list<ValueType> ArgTy, int K> : WMMA_SWMMAC_Profile<ArgTy, K> {
  dag WmmaInPat;
  dag WmmaOutPat;
  dag WmmaInlineInPat;
  dag WmmaInlineOutPat;
  bit isInt;

  let Src0RC64 = getVOP3SrcForVT<ArgVT[1]>.ret; // srcA
  let Src1RC64 = getVALUDstForVT<ArgVT[2]>.ret; // srcB
  let Src2RC64 = getVOP3SrcForVT<ArgVT[3]>.ret; // srcC
}

class VOPM_WMMA_Profile_Float<list<ValueType> ArgTy, int K> : VOPM_WMMA_Profile<ArgTy, K> {
  let IsWMMA = 1;
  let isInt = 0;

  let Ins64 = (ins Src0RC64:$srcA, Src1RC64:$srcB, Src2RC64:$srcC, Clamp0:$clamp);

  let WmmaInPat = (ins Src0VT:$srcA, Src1VT:$srcB, Src2VT:$srcC, i1:$clamp);
  let WmmaOutPat = (ins Src0VT:$srcA, Src1VT:$srcB, Src2VT:$srcC, i1:$clamp);

  let WmmaInlineInPat = (ins Src0VT:$srcA, Src1VT:$srcB, (Src2VT (WMMAVISrc Src2VT:$srcC)), i1:$clamp);
  let WmmaInlineOutPat = (ins Src0VT:$srcA, Src1VT:$srcB, Src2VT:$srcC, i1:$clamp);

  let Asm64 = "$vdst, $srcA, $srcB, $srcC$clamp";
}

class VOPM_WMMA_Profile_Int<list<ValueType> ArgTy, int K> : VOPM_WMMA_Profile<ArgTy, K> {
  let IsWMMA = 1;
  let isInt = 1;

  let Ins64 = (ins SignedA:$signed_a, Src0RC64:$srcA, SignedB:$signed_b, Src1RC64:$srcB, Src2RC64:$srcC, Clamp0:$clamp);

  let WmmaInPat = (ins i1:$signed_a, Src0VT:$srcA, i1:$signed_b, Src1VT:$srcB, Src2VT:$srcC, i1:$clamp);
  let WmmaOutPat = (ins i1:$signed_a, Src0VT:$srcA, i1:$signed_b, Src1VT:$srcB, Src2VT:$srcC, i1:$clamp);

  let WmmaInlineInPat = (ins i1:$signed_a, Src0VT:$srcA, i1:$signed_b, Src1VT:$srcB, (Src2VT (WMMAVISrc Src2VT:$srcC)), i1:$clamp);
  let WmmaInlineOutPat = (ins i1:$signed_a, Src0VT:$srcA, i1:$signed_b, Src1VT:$srcB, Src2VT:$srcC, i1:$clamp);

  let Asm64 = "$vdst, $srcA, $srcB, $srcC$signed_a$signed_b$clamp";
}

def VOPM_WMMA_F32_F16_X16     : VOPM_WMMA_Profile_Float<[v8f32, v8f16, v8f16, v8f32], KMult.X1>;
def VOPM_WMMA_F32_BF16_X16    : VOPM_WMMA_Profile_Float<[v8f32, v8bf16, v8bf16, v8f32], KMult.X1>;
def VOPM_WMMA_F16_F16_X16     : VOPM_WMMA_Profile_Float<[v8f16, v8f16, v8f16, v8f16], KMult.X1>;
def VOPM_WMMA_BF16_BF16_X16   : VOPM_WMMA_Profile_Float<[v8bf16, v8bf16, v8bf16, v8bf16], KMult.X1>;
def VOPM_WMMA_F32_F8_X16      : VOPM_WMMA_Profile_Float<[v8f32, v2i32, v2i32, v8f32], KMult.X1>;
def VOPM_WMMA_F16_F8_X16      : VOPM_WMMA_Profile_Float<[v8f16, v2i32, v2i32, v8f16], KMult.X1>;
def VOPM_WMMA_F32_F8_X32      : VOPM_WMMA_Profile_Float<[v8f32, v4i32, v4i32, v8f32], KMult.X2>;
def VOPM_WMMA_F16_F8_X32      : VOPM_WMMA_Profile_Float<[v8f16, v4i32, v4i32, v8f16], KMult.X2>;

def VOPM_WMMA_I32_IU8_X16    : VOPM_WMMA_Profile_Int<[v8i32, v2i32, v2i32, v8i32], KMult.X1>;
def VOPM_WMMA_F32_IU8_X16    : VOPM_WMMA_Profile_Int<[v8f32, v2i32, v2i32, v8f32], KMult.X1>;
def VOPM_WMMA_F32I32_IU8_X16 : VOPM_WMMA_Profile_Int<[v8f32, v2i32, v2i32, v8i32], KMult.X1>;
def VOPM_WMMA_I32_IU8_X32    : VOPM_WMMA_Profile_Int<[v8i32, v4i32, v4i32, v8i32], KMult.X2>;
def VOPM_WMMA_F32_IU8_X32    : VOPM_WMMA_Profile_Int<[v8f32, v4i32, v4i32, v8f32], KMult.X2>;
def VOPM_WMMA_F32I32_IU8_X32 : VOPM_WMMA_Profile_Int<[v8f32, v4i32, v4i32, v8i32], KMult.X2>;
def VOPM_WMMA_I32_IU4_X16    : VOPM_WMMA_Profile_Int<[v8i32, i32, i32, v8i32], KMult.X1>;
def VOPM_WMMA_F32_IU4_X16    : VOPM_WMMA_Profile_Int<[v8f32, i32, i32, v8f32], KMult.X1>;
def VOPM_WMMA_F32I32_IU4_X16 : VOPM_WMMA_Profile_Int<[v8f32, i32, i32, v8i32], KMult.X1>;
def VOPM_WMMA_I32_IU4_X32    : VOPM_WMMA_Profile_Int<[v8i32, v2i32, v2i32, v8i32], KMult.X2>;
def VOPM_WMMA_F32_IU4_X32    : VOPM_WMMA_Profile_Int<[v8f32, v2i32, v2i32, v8f32], KMult.X2>;
def VOPM_WMMA_F32I32_IU4_X32 : VOPM_WMMA_Profile_Int<[v8f32, v2i32, v2i32, v8i32], KMult.X2>;
def VOPM_WMMA_I32_IU4_X64    : VOPM_WMMA_Profile_Int<[v8i32, v4i32, v4i32, v8i32], KMult.X4>;
def VOPM_WMMA_F32_IU4_X64    : VOPM_WMMA_Profile_Int<[v8f32, v4i32, v4i32, v8f32], KMult.X4>;
def VOPM_WMMA_F32I32_IU4_X64 : VOPM_WMMA_Profile_Int<[v8f32, v4i32, v4i32, v8i32], KMult.X4>;

class VOPM_SWMMAC_Profile<list<ValueType> ArgTy, int K> : WMMA_SWMMAC_Profile<ArgTy, K> {
  let IsSWMMAC = 1;
  dag IndexKey = (ins IndexKey16bit:$index_key_16bit);

  dag IndexInPat = (ins (i16 (SWMMACIndex16 i32:$src2, i32:$index_key_16bit)));
  dag IndexOutPat = (ins i32:$src2, i32:$index_key_16bit);

  dag SwmmacInPat;
  dag SwmmacOutPat;

  dag WmmaInlineOutPat;
  bit isInt;
}

class VOPM_SWMMAC_Profile_Float<list<ValueType> ArgTy, int K> : VOPM_SWMMAC_Profile<ArgTy, K> {
  let isInt = 0;
  let Ins64 = !con((ins Src0RC64:$src0, Src1RC64:$src1, DstRC:$srcTiedDef, Src2RC64:$src2), IndexKey, (ins Clamp0:$clamp));

  let SwmmacInPat  = !con((ins Src0VT:$src0, Src1VT:$src1, Src3VT:$srcTiedDef), IndexInPat, (ins i1:$clamp));
  let SwmmacOutPat = !con((ins Src0VT:$src0, Src1VT:$src1, Src3VT:$srcTiedDef), IndexOutPat, (ins i1:$clamp));

  let Asm64 = "$vdst, $src0, $src1, $src2$index_key_16bit$clamp";
}

class VOPM_SWMMAC_Profile_Int<list<ValueType> ArgTy, int K> : VOPM_SWMMAC_Profile<ArgTy, K> {
  let isInt = 1;
  let Ins64 = !con((ins SignedA:$signed_a, Src0RC64:$src0, SignedB:$signed_b, Src1RC64:$src1, DstRC:$srcTiedDef, Src2RC64:$src2), IndexKey, (ins Clamp0:$clamp));

  let SwmmacInPat  = !con((ins i1:$signed_a, Src0VT:$src0, i1:$signed_b, Src1VT:$src1, Src3VT:$srcTiedDef), IndexInPat, (ins i1:$clamp));
  let SwmmacOutPat = !con((ins i1:$signed_a, Src0VT:$src0, i1:$signed_b, Src1VT:$src1, Src3VT:$srcTiedDef), IndexOutPat, (ins i1:$clamp));

  let Asm64 = "$vdst, $src0, $src1, $src2$signed_a$signed_b$index_key_16bit$clamp";
}

def VOPM_SWMMAC_F16_F16_X32   : VOPM_SWMMAC_Profile_Float<[v8f16, v8f16, v16f16, i32, v8f16], 1>;
def VOPM_SWMMAC_F32_F16_X32   : VOPM_SWMMAC_Profile_Float<[v8f32, v8f16, v16f16, i32, v8f32], 1>;
def VOPM_SWMMAC_F32_BF16_X32  : VOPM_SWMMAC_Profile_Float<[v8f32, v8bf16, v16bf16, i32, v8f32], 1>;
def VOPM_SWMMAC_BF16_BF16_X32 : VOPM_SWMMAC_Profile_Float<[v8bf16, v8bf16, v16bf16, i32, v8bf16], 1>;
def VOPM_SWMMAC_F32_F8_X32    : VOPM_SWMMAC_Profile_Float<[v8f32, v2i32,  v4i32, i32, v8f32], 1>;
def VOPM_SWMMAC_F16_F8_X32    : VOPM_SWMMAC_Profile_Float<[v8f16, v2i32,  v4i32, i32, v8f16], 1>;
def VOPM_SWMMAC_F32_F8_X64    : VOPM_SWMMAC_Profile_Float<[v8f32, v4i32,  v8i32, i32, v8f32], 2>;
def VOPM_SWMMAC_F16_F8_X64    : VOPM_SWMMAC_Profile_Float<[v8f16, v4i32,  v8i32, i32, v8f16], 2>;

def VOPM_SWMMAC_I32_IU8_X32     : VOPM_SWMMAC_Profile_Int<[v8i32, v2i32, v4i32, i32, v8i32], 1>;
def VOPM_SWMMAC_F32_IU8_X32     : VOPM_SWMMAC_Profile_Int<[v8f32, v2i32, v4i32, i32, v8f32], 1>;
def VOPM_SWMMAC_F32I32_IU8_X32  : VOPM_SWMMAC_Profile_Int<[v8f32, v2i32, v4i32, i32, v8i32], 1>;
def VOPM_SWMMAC_I32_IU8_X64     : VOPM_SWMMAC_Profile_Int<[v8i32, v4i32, v8i32, i32, v8i32], 2>;
def VOPM_SWMMAC_F32_IU8_X64     : VOPM_SWMMAC_Profile_Int<[v8f32, v4i32, v8i32, i32, v8f32], 2>;
def VOPM_SWMMAC_F32I32_IU8_X64  : VOPM_SWMMAC_Profile_Int<[v8f32, v4i32, v8i32, i32, v8i32], 2>;
def VOPM_SWMMAC_I32_IU4_X32     : VOPM_SWMMAC_Profile_Int<[v8i32, i32, v2i32, i32, v8i32], 1>;
def VOPM_SWMMAC_F32_IU4_X32     : VOPM_SWMMAC_Profile_Int<[v8f32, i32, v2i32, i32, v8f32], 1>;
def VOPM_SWMMAC_F32I32_IU4_X32  : VOPM_SWMMAC_Profile_Int<[v8f32, i32, v2i32, i32, v8i32], 1>;
def VOPM_SWMMAC_I32_IU4_X64     : VOPM_SWMMAC_Profile_Int<[v8i32, v2i32, v4i32, i32, v8i32], 2>;
def VOPM_SWMMAC_F32_IU4_X64     : VOPM_SWMMAC_Profile_Int<[v8f32, v2i32, v4i32, i32, v8f32], 2>;
def VOPM_SWMMAC_F32I32_IU4_X64  : VOPM_SWMMAC_Profile_Int<[v8f32, v2i32, v4i32, i32, v8i32], 2>;
def VOPM_SWMMAC_I32_IU4_X128    : VOPM_SWMMAC_Profile_Int<[v8i32, v4i32, v8i32, i32, v8i32], 4>;
def VOPM_SWMMAC_F32_IU4_X128    : VOPM_SWMMAC_Profile_Int<[v8f32, v4i32, v8i32, i32, v8f32], 4>;
def VOPM_SWMMAC_F32I32_IU4_X128 : VOPM_SWMMAC_Profile_Int<[v8f32, v4i32, v8i32, i32, v8i32], 4>;

class Cvt_To_Tensor_Profile<list<ValueType> ArgTy, int PixelShape, int Scatter> : VOPMProfile<ArgTy, 1, 1, 0, 1> {
  let Outs64 = !cond(!eq(Scatter, 1) : (outs DstRC64:$vdst),
                     !eq(Scatter, 2) : (outs DstRC64:$vdst, Src2RC64:$src2),
                     !eq(Scatter, 4) : (outs DstRC64:$vdst, Src2RC64:$src2, Src2RC64:$src3, Src2RC64:$src4));

  let Ins64 = !if(HasSrc1, (ins Src0RC64:$src0, Src1RC64:$src1, AuxData:$aux_data, Clamp0:$clamp),
                           (ins Src0RC64:$src0, AuxData:$aux_data, Clamp0:$clamp));

  dag InPat = !if(HasSrc1, (ins Src0VT:$src0, Src1VT:$src1, timm:$aux_data, i1:$clamp),
                           (ins Src0VT:$src0, timm:$aux_data, i1:$clamp));

  list<SchedReadWrite> SchedRW = !cond(!eq(Scatter, 1) : [Write32Bit],
                                       !eq(Scatter, 2) : [Write32Bit, Write32Bit],
                                       !eq(Scatter, 4) : [Write32Bit, Write32Bit, Write32Bit, Write32Bit]);

  string DestAsm = !cond(!eq(Scatter, 1) : "$vdst",
                         !eq(Scatter, 2) : "$vdst, $src2",
                         !eq(Scatter, 4) : "$vdst, $src2, $src3, $src4");
  string SrcAsm = !if(HasSrc1, "$src0, $src1$aux_data$clamp",
                               "$src0$aux_data$clamp");
  let Asm64 = DestAsm#", "#SrcAsm;

  int PrimTy = PixelShape;
}

// CVT_TO_TENSOR profiles do not follow strict type checking (for example "f32" can be i32, of v2fb16),
// which does not matter in practice because constants/modifiers are not supported.
// However, that helps reducing the number of profiles.
//
//                                                dest  src0   src1    src2
//                                                dest  acc_in sr      dests[2-4]
def VOPM_CVT_TO_TENSOR_16B_8x4 : Cvt_To_Tensor_Profile<[f32, v4f32, untyped, f32], 1, 4>;
def VOPM_CVT_TO_TENSOR_8B_8x4 : Cvt_To_Tensor_Profile<[f32, v4f32, untyped, f32], 1, 2>;
def VOPM_CVT_TO_TENSOR_4B_8x4 : Cvt_To_Tensor_Profile<[f32, v4f32, untyped, f32], 1, 2>;

def VOPM_CVT_TO_TENSOR_16B_4x4 : Cvt_To_Tensor_Profile<[f32, v4f32, untyped, f32], 0, 4>;
def VOPM_CVT_TO_TENSOR_8B_4x4 : Cvt_To_Tensor_Profile<[f32, v4f32, untyped, f32], 0, 2>;
def VOPM_CVT_TO_TENSOR_4B_4x4 : Cvt_To_Tensor_Profile<[f32, v4f32, untyped, untyped], 0, 1>;

def VOPM_CVT_TO_TENSOR_16B_4x2 : Cvt_To_Tensor_Profile<[f32, v4f32, untyped, f32], 2, 2>;
def VOPM_CVT_TO_TENSOR_8B_4x2 : Cvt_To_Tensor_Profile<[f32, v4f32, untyped, untyped], 2, 1>;
def VOPM_CVT_TO_TENSOR_4B_4x2 : Cvt_To_Tensor_Profile<[f32, v4f32, untyped, untyped], 2, 1>;

def VOPM_CVT_TO_TENSOR_16B_ACC16B_4x2 : Cvt_To_Tensor_Profile<[f32, v2f32, untyped, f32], 2, 2>;
def VOPM_CVT_TO_TENSOR_8B_ACC16B_4x2 : Cvt_To_Tensor_Profile<[f32, v2f32, untyped, untyped], 2, 1>;
def VOPM_CVT_TO_TENSOR_4B_ACC16B_4x2 : Cvt_To_Tensor_Profile<[f32, v2f32, untyped, untyped], 2, 1>;

class VOPM_Pseudo <VOPProfile P, list<dag> pattern = [], string opName = !tolower(NAME)> :
  VOP_Pseudo<opName, "", P, P.Outs64, P.Ins64, "", pattern> {
  let mayRaiseFPException = 0;
  let ReadsModeReg = 0;
  let Uses = [EXEC];
}

def V_MOV_2SRC_B64               : VOPM_Pseudo<VOPM_I32_I32_I32>;
def V_BPERMUTE_B32               : VOPM_Pseudo<VOPM_I32_I32_I32,
                                               [(set i32:$vdst, (int_amdgcn_bpermute_b32 i32:$src0, i32:$src1))]>;

def V_PERMUTE_PAIR_GENSGPR_B32   : VOPM_Pseudo<VOPM_I32_I32_I64,
                                               [(set i32:$vdst, (int_amdgcn_permute_pair_gensgpr_b32 i32:$src0, i64:$src1))]>;
def V_PERMUTE_PAIR_BCAST_B32     : VOPM_Pseudo<VOPM_I32_I32_AUX,
                                               [(set i32:$vdst, (int_amdgcn_permute_pair_bcast_b32 i32:$src0, timm:$aux_data))]>;
def V_PERMUTE_PAIR_2SRC_ROTATE_GROUP_B32 : VOPM_Pseudo<VOPM_I32_I32_I32_AUX,
                                                       [(set i32:$vdst, (int_amdgcn_permute_pair_2src_rotate_group_b32 i32:$src0, i32:$src1, timm:$aux_data))]>;

let SchedRW = [Write32Bit, Write32Bit] in {
  def V_PERMUTE_PAIR_2SRC_INTERLEAVE_B64   : VOPM_Pseudo<VOPM_PERMUTE_I32_I32_I32_I32,
                                                         [(set i32:$vdst, i32:$src2, (int_amdgcn_permute_pair_2src_interleave_b64 i32:$src0, i32:$src1, timm:$aux_data))]>;
  def V_PERMUTE_PACK_TENSOR_2SRC_B64       : VOPM_Pseudo<VOPM_PERMUTE_I32_I32_I32_I32,
                                                         [(set i32:$vdst, i32:$src2, (int_amdgcn_permute_pack_tensor_2src_b64 i32:$src0, i32:$src1, timm:$aux_data))]>;
}

multiclass VOPM_Conv_MultiPseudos_1x1_Iter<string ProfilePrefix> {
  def _ITER_1 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_ITER_1")>;
  def _ITER_2 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_ITER_2")>;
  def _ITER_3 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_ITER_3")>;
  def _ITER_4 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_ITER_4")>;
}

multiclass VOPM_Conv_MultiPseudos<string ProfilePrefix> {
  defm _1x1_8x4 : VOPM_Conv_MultiPseudos_1x1_Iter<ProfilePrefix #"_1x1_8x4">;
  defm _1x1_4x4 : VOPM_Conv_MultiPseudos_1x1_Iter<ProfilePrefix #"_1x1_4x4">;
  defm _1x1_4x2 : VOPM_Conv_MultiPseudos_1x1_Iter<ProfilePrefix #"_1x1_4x2">;

  def _3x3_8x4 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_3x3_8x4")>;
  def _3x3_4x4 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_3x3_4x4")>;
  def _3x3_4x2 : VOPM_Pseudo<!cast<Conv_Profile>(ProfilePrefix #"_3x3_4x2")>;
}

defm V_CONVOLVE_F16_F16    : VOPM_Conv_MultiPseudos <"VOPM_CONV_F16_F16">;
defm V_CONVOLVE_BF16_BF16  : VOPM_Conv_MultiPseudos <"VOPM_CONV_BF16_BF16">;

defm V_CONVOLVE_F16_FP8    : VOPM_Conv_MultiPseudos <"VOPM_CONV_F16_FP8">;
defm V_CONVOLVE_BF16_BF8   : VOPM_Conv_MultiPseudos <"VOPM_CONV_BF16_BF8">;
defm V_CONVOLVE_F16_IU8    : VOPM_Conv_MultiPseudos <"VOPM_CONV_F16_IU8">;
defm V_CONVOLVE_F16_IU4    : VOPM_Conv_MultiPseudos <"VOPM_CONV_F16_IU4">;

defm V_CONVOLVE_F32_F16    : VOPM_Conv_MultiPseudos <"VOPM_CONV_F32_F16">;
defm V_CONVOLVE_F32_BF16   : VOPM_Conv_MultiPseudos <"VOPM_CONV_F32_BF16">;
defm V_CONVOLVE_F32_FP8    : VOPM_Conv_MultiPseudos <"VOPM_CONV_F32_FP8">;
defm V_CONVOLVE_F32_BF8    : VOPM_Conv_MultiPseudos <"VOPM_CONV_F32_BF8">;
defm V_CONVOLVE_F32_IU8    : VOPM_Conv_MultiPseudos <"VOPM_CONV_F32_IU8">;
defm V_CONVOLVE_F32_IU4    : VOPM_Conv_MultiPseudos <"VOPM_CONV_F32_IU4">;
defm V_CONVOLVE_I32_IU8    : VOPM_Conv_MultiPseudos <"VOPM_CONV_I32_IU8">;
defm V_CONVOLVE_I32_IU4    : VOPM_Conv_MultiPseudos <"VOPM_CONV_I32_IU4">;
defm V_CONVOLVE_F32I32_IU8 : VOPM_Conv_MultiPseudos <"VOPM_CONV_F32I32_IU8">;
defm V_CONVOLVE_F32I32_IU4 : VOPM_Conv_MultiPseudos <"VOPM_CONV_F32I32_IU4">;

multiclass VOPM_Cvt_To_Tensor_Pseudo<string ProfileName> {
  defvar P = !cast<Cvt_To_Tensor_Profile>(ProfileName);
  let SchedRW = P.SchedRW in
    def NAME : VOPM_Pseudo<P>;
}

multiclass VOPM_Cvt_To_Tensor_Shape_Pseudo<string ProfilePrefix, int acc16bit> {
  defm _8x4 : VOPM_Cvt_To_Tensor_Pseudo<ProfilePrefix #"_8x4">;
  defm _4x4 : VOPM_Cvt_To_Tensor_Pseudo<ProfilePrefix #"_4x4">;
  defm _4x2 : VOPM_Cvt_To_Tensor_Pseudo<ProfilePrefix # !if(!eq(acc16bit,1),"_ACC16B_4x2","_4x2")>;
}

multiclass VOPM_Cvt_To_Tensor_Acc_Pseudo<string ProfilePrefix> {
  defm _F32 : VOPM_Cvt_To_Tensor_Shape_Pseudo<ProfilePrefix, 0>;
  defm _F16 : VOPM_Cvt_To_Tensor_Shape_Pseudo<ProfilePrefix, 1>;
  defm _BF16 : VOPM_Cvt_To_Tensor_Shape_Pseudo<ProfilePrefix, 1>;
}

multiclass VOPM_Cvt_To_Tensor_MultiPseudos<string ProfilePrefix> {
 defm _I4 : VOPM_Cvt_To_Tensor_Acc_Pseudo<ProfilePrefix #"_4B">;
 defm _U4 : VOPM_Cvt_To_Tensor_Acc_Pseudo<ProfilePrefix #"_4B">;

 defm _I8 : VOPM_Cvt_To_Tensor_Acc_Pseudo<ProfilePrefix #"_8B">;
 defm _U8 : VOPM_Cvt_To_Tensor_Acc_Pseudo<ProfilePrefix #"_8B">;
 defm _FP8 : VOPM_Cvt_To_Tensor_Acc_Pseudo<ProfilePrefix #"_8B">;
 defm _BF8 : VOPM_Cvt_To_Tensor_Acc_Pseudo<ProfilePrefix #"_8B">;

 defm _F16 : VOPM_Cvt_To_Tensor_Acc_Pseudo<ProfilePrefix #"_16B">;
 defm _BF16 : VOPM_Cvt_To_Tensor_Acc_Pseudo<ProfilePrefix #"_16B">;
}

defm V_CVT_TO_TENSOR : VOPM_Cvt_To_Tensor_MultiPseudos<"VOPM_CVT_TO_TENSOR">;

def V_SCALE_BIAS_ACTIVATE_F32 : VOPM_Pseudo<SBA_Profile<[v4f32, v4f32,  f32, untyped], 1>,
                                            [(set v4f32:$vdst, (int_amdgcn_scale_bias_activate_f32 v4f32:$src0, f32:$ssrc, f32:$src1, timm:$aux_data, i1:$clamp))]>;
def V_SCALE_BIAS_ACTIVATE_F16 : VOPM_Pseudo<SBA_Profile<[v8f16, v8f16,  v2f16, untyped], 1>,
                                            [(set v8f16:$vdst, (int_amdgcn_scale_bias_activate_f16 v8f16:$src0, f32:$ssrc, v2f16:$src1, timm:$aux_data, i1:$clamp))]>;
def V_SCALE_BIAS_ACTIVATE_BF16 : VOPM_Pseudo<SBA_Profile<[v8bf16, v8bf16, v2bf16, untyped], 1>,
                                            [(set v8bf16:$vdst, (int_amdgcn_scale_bias_activate_bf16 v8bf16:$src0, f32:$ssrc, v2bf16:$src1, timm:$aux_data, i1:$clamp))]>;

let SchedRW = [Write32Bit, Write32Bit] in {
def V_SCALE_BIAS_ACTIVATE_SCATTER2_F16 : VOPM_Pseudo<SBA_Profile<[v2f16, v4f16,  v2f16, v2f16], 2>,
                                                     [(set v2f16:$vdst, v2f16:$src2, (int_amdgcn_scale_bias_activate_scatter2_f16 v4f16:$src0, f32:$ssrc, v2f16:$src1, timm:$aux_data, i1:$clamp))]>;
def V_SCALE_BIAS_ACTIVATE_SCATTER2_BF16 : VOPM_Pseudo<SBA_Profile<[v2bf16, v4bf16,  v2bf16, v2bf16], 2>,
                                                      [(set v2bf16:$vdst, v2bf16:$src2, (int_amdgcn_scale_bias_activate_scatter2_bf16 v4bf16:$src0, f32:$ssrc, v2bf16:$src1, timm:$aux_data, i1:$clamp))]>;
}
let SchedRW = [Write32Bit, Write32Bit, Write32Bit, Write32Bit] in {
def V_SCALE_BIAS_ACTIVATE_SCATTER4_F16 : VOPM_Pseudo<SBA_Profile<[v2f16, v8f16,  v2f16, v2f16], 4>,
                                                     [(set v2f16:$vdst, v2f16:$src2, v2f16:$src3, v2f16:$src4, (int_amdgcn_scale_bias_activate_scatter4_f16 v8f16:$src0, f32:$ssrc, v2f16:$src1, timm:$aux_data, i1:$clamp))]>;
def V_SCALE_BIAS_ACTIVATE_SCATTER4_BF16 : VOPM_Pseudo<SBA_Profile<[v2bf16, v8bf16,  v2bf16, v2bf16], 4>,
                                                      [(set v2bf16:$vdst, v2bf16:$src2, v2bf16:$src3, v2bf16:$src4, (int_amdgcn_scale_bias_activate_scatter4_bf16 v8bf16:$src0, f32:$ssrc, v2bf16:$src1, timm:$aux_data, i1:$clamp))]>;
}

def V_UNIFORM_SCALE_ACTIVATE_F32 : VOPM_Pseudo<SBA_Profile<[v4f32, v4f32,  untyped, untyped], 1>,
                                               [(set v4f32:$vdst, (int_amdgcn_uniform_scale_activate_f32 v4f32:$src0, f32:$ssrc, timm:$aux_data, i1:$clamp))]>;
def V_UNIFORM_SCALE_ACTIVATE_F16 : VOPM_Pseudo<SBA_Profile<[v8f16, v8f16,  untyped, untyped], 1>,
                                               [(set v8f16:$vdst, (int_amdgcn_uniform_scale_activate_f16 v8f16:$src0, f32:$ssrc, timm:$aux_data, i1:$clamp))]>;
def V_UNIFORM_SCALE_ACTIVATE_BF16 : VOPM_Pseudo<SBA_Profile<[v8bf16, v8bf16, untyped, untyped], 1>,
                                                [(set v8bf16:$vdst, (int_amdgcn_uniform_scale_activate_bf16 v8bf16:$src0, f32:$ssrc, timm:$aux_data, i1:$clamp))]>;

let SchedRW = [Write32Bit, Write32Bit] in {
def V_UNIFORM_SCALE_ACTIVATE_SCATTER2_F16 : VOPM_Pseudo<SBA_Profile<[v2f16, v4f16,  untyped, v2f16], 2>,
                                                        [(set v2f16:$vdst, v2f16:$src2, (int_amdgcn_uniform_scale_activate_scatter2_f16 v4f16:$src0, f32:$ssrc, timm:$aux_data, i1:$clamp))]>;
def V_UNIFORM_SCALE_ACTIVATE_SCATTER2_BF16 : VOPM_Pseudo<SBA_Profile<[v2bf16, v4bf16,  untyped, v2bf16], 2>,
                                                         [(set v2bf16:$vdst, v2bf16:$src2, (int_amdgcn_uniform_scale_activate_scatter2_bf16 v4bf16:$src0, f32:$ssrc, timm:$aux_data, i1:$clamp))]>;
}
let SchedRW = [Write32Bit, Write32Bit, Write32Bit, Write32Bit] in {
def V_UNIFORM_SCALE_ACTIVATE_SCATTER4_F16 : VOPM_Pseudo<SBA_Profile<[v2f16, v8f16,  untyped, v2f16], 4>,
                                                        [(set v2f16:$vdst, v2f16:$src2, v2f16:$src3, v2f16:$src4, (int_amdgcn_uniform_scale_activate_scatter4_f16 v8f16:$src0, f32:$ssrc, timm:$aux_data, i1:$clamp))]>;
def V_UNIFORM_SCALE_ACTIVATE_SCATTER4_BF16 : VOPM_Pseudo<SBA_Profile<[v2bf16, v8bf16,  untyped, v2bf16], 4>,
                                                         [(set v2bf16:$vdst, v2bf16:$src2, v2bf16:$src3, v2bf16:$src4, (int_amdgcn_uniform_scale_activate_scatter4_bf16 v8bf16:$src0, f32:$ssrc, timm:$aux_data, i1:$clamp))]>;
}

multiclass VOPM_WMMA_Pseudo<VOPM_WMMA_Profile P> {
  defvar Instr = !tolower(NAME);
  defvar WMMAConstraints2Addr = "@earlyclobber $vdst,$vdst = $srcC";
  defvar WMMAConstraints3Addr = "@earlyclobber $vdst";

  let Mnemonic = Instr in {
    let Constraints = WMMAConstraints2Addr, isConvertibleToThreeAddress = 1 in
      def _twoaddr : VOPM_Pseudo<P, [], Instr>{
        let PseudoInstr = Instr;
      }

    let Constraints = WMMAConstraints3Addr in
      def _threeaddr : VOPM_Pseudo<!cast<VOPProfile>(P), [], Instr>{
        let PseudoInstr = Instr;
      }
  }
  def : WMMAOpcodeMapping<!cast<Instruction>(NAME # _twoaddr),
                          !cast<Instruction>(NAME # _threeaddr)>;
}

defm V_WMMA_BF16_16X16X16_BF16   : VOPM_WMMA_Pseudo <VOPM_WMMA_BF16_BF16_X16>;
defm V_WMMA_F32_16X16X16_BF16    : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_BF16_X16>;
defm V_WMMA_F16_16X16X16_F16     : VOPM_WMMA_Pseudo <VOPM_WMMA_F16_F16_X16>;
defm V_WMMA_F32_16X16X16_BF8_BF8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_F8_X16>;
defm V_WMMA_F32_16X16X16_BF8_FP8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_F8_X16>;
defm V_WMMA_F32_16X16X16_F16     : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_F16_X16>;
defm V_WMMA_F32_16X16X16_FP8_FP8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_F8_X16>;
defm V_WMMA_F32_16X16X16_FP8_BF8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_F8_X16>;

defm V_WMMA_F16_16X16X16_BF8_BF8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F16_F8_X16>;
defm V_WMMA_F16_16X16X16_BF8_FP8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F16_F8_X16>;
defm V_WMMA_F16_16X16X16_FP8_FP8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F16_F8_X16>;
defm V_WMMA_F16_16X16X16_FP8_BF8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F16_F8_X16>;

multiclass VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_Profile P> {
  defvar Instr = !tolower(NAME);
  def _twoaddr : VOPM_Pseudo<P, [], Instr> {
    let PseudoInstr = Instr;
    let Constraints = "@earlyclobber $vdst,$vdst = $srcTiedDef";
  }
}

defm V_WMMA_F16_16X16X32_BF8_BF8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F16_F8_X32>;
defm V_WMMA_F16_16X16X32_BF8_FP8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F16_F8_X32>;
defm V_WMMA_F16_16X16X32_FP8_FP8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F16_F8_X32>;
defm V_WMMA_F16_16X16X32_FP8_BF8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F16_F8_X32>;
defm V_WMMA_F32_16X16X32_BF8_BF8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_F8_X32>;
defm V_WMMA_F32_16X16X32_BF8_FP8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_F8_X32>;
defm V_WMMA_F32_16X16X32_FP8_FP8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_F8_X32>;
defm V_WMMA_F32_16X16X32_FP8_BF8 : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_F8_X32>;

defm V_WMMA_I32_16X16X16_IU8     : VOPM_WMMA_Pseudo <VOPM_WMMA_I32_IU8_X16>;
defm V_WMMA_F32_16X16X16_IU8     : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_IU8_X16>;
defm V_WMMA_F32I32_16X16X16_IU8  : VOPM_WMMA_Pseudo <VOPM_WMMA_F32I32_IU8_X16>;
defm V_WMMA_I32_16X16X32_IU8     : VOPM_WMMA_Pseudo <VOPM_WMMA_I32_IU8_X32>;
defm V_WMMA_F32_16X16X32_IU8     : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_IU8_X32>;
defm V_WMMA_F32I32_16X16X32_IU8  : VOPM_WMMA_Pseudo <VOPM_WMMA_F32I32_IU8_X32>;
defm V_WMMA_I32_16X16X16_IU4     : VOPM_WMMA_Pseudo <VOPM_WMMA_I32_IU4_X16>;
defm V_WMMA_F32_16X16X16_IU4     : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_IU4_X16>;
defm V_WMMA_F32I32_16X16X16_IU4  : VOPM_WMMA_Pseudo <VOPM_WMMA_F32I32_IU4_X16>;
defm V_WMMA_I32_16X16X32_IU4     : VOPM_WMMA_Pseudo <VOPM_WMMA_I32_IU4_X32>;
defm V_WMMA_F32_16X16X32_IU4     : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_IU4_X32>;
defm V_WMMA_F32I32_16X16X32_IU4  : VOPM_WMMA_Pseudo <VOPM_WMMA_F32I32_IU4_X32>;
defm V_WMMA_I32_16X16X64_IU4     : VOPM_WMMA_Pseudo <VOPM_WMMA_I32_IU4_X64>;
defm V_WMMA_F32_16X16X64_IU4     : VOPM_WMMA_Pseudo <VOPM_WMMA_F32_IU4_X64>;
defm V_WMMA_F32I32_16X16X64_IU4  : VOPM_WMMA_Pseudo <VOPM_WMMA_F32I32_IU4_X64>;


defm V_SWMMAC_F16_16X16X32_F16     : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F16_F16_X32>;
defm V_SWMMAC_F32_16X16X32_F16     : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_F16_X32>;
defm V_SWMMAC_F32_16X16X32_BF16    : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_BF16_X32>;
defm V_SWMMAC_BF16_16X16X32_BF16   : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_BF16_BF16_X32>;

defm V_SWMMAC_F32_16X16X32_FP8_FP8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_F8_X32>;
defm V_SWMMAC_F32_16X16X32_FP8_BF8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_F8_X32>;
defm V_SWMMAC_F32_16X16X32_BF8_FP8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_F8_X32>;
defm V_SWMMAC_F32_16X16X32_BF8_BF8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_F8_X32>;
defm V_SWMMAC_F16_16X16X32_FP8_FP8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F16_F8_X32>;
defm V_SWMMAC_F16_16X16X32_FP8_BF8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F16_F8_X32>;
defm V_SWMMAC_F16_16X16X32_BF8_FP8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F16_F8_X32>;
defm V_SWMMAC_F16_16X16X32_BF8_BF8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F16_F8_X32>;

defm V_SWMMAC_F32_16X16X64_FP8_FP8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_F8_X64>;
defm V_SWMMAC_F32_16X16X64_FP8_BF8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_F8_X64>;
defm V_SWMMAC_F32_16X16X64_BF8_FP8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_F8_X64>;
defm V_SWMMAC_F32_16X16X64_BF8_BF8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_F8_X64>;
defm V_SWMMAC_F16_16X16X64_FP8_FP8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F16_F8_X64>;
defm V_SWMMAC_F16_16X16X64_FP8_BF8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F16_F8_X64>;
defm V_SWMMAC_F16_16X16X64_BF8_FP8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F16_F8_X64>;
defm V_SWMMAC_F16_16X16X64_BF8_BF8 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F16_F8_X64>;

defm V_SWMMAC_I32_16X16X32_IU8     : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_I32_IU8_X32>;
defm V_SWMMAC_F32_16X16X32_IU8     : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_IU8_X32>;
defm V_SWMMAC_F32I32_16X16X32_IU8  : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32I32_IU8_X32>;
defm V_SWMMAC_I32_16X16X64_IU8     : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_I32_IU8_X64>;
defm V_SWMMAC_F32_16X16X64_IU8     : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_IU8_X64>;
defm V_SWMMAC_F32I32_16X16X64_IU8  : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32I32_IU8_X64>;
defm V_SWMMAC_I32_16X16X32_IU4     : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_I32_IU4_X32>;
defm V_SWMMAC_F32_16X16X32_IU4     : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_IU4_X32>;
defm V_SWMMAC_F32I32_16X16X32_IU4  : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32I32_IU4_X32>;
defm V_SWMMAC_I32_16X16X64_IU4     : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_I32_IU4_X64>;
defm V_SWMMAC_F32_16X16X64_IU4     : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_IU4_X64>;
defm V_SWMMAC_F32I32_16X16X64_IU4  : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32I32_IU4_X64>;
defm V_SWMMAC_I32_16X16X128_IU4    : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_I32_IU4_X128>;
defm V_SWMMAC_F32_16X16X128_IU4    : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32_IU4_X128>;
defm V_SWMMAC_F32I32_16X16X128_IU4 : VOPM_SWMMAC_Pseudo<VOPM_SWMMAC_F32I32_IU4_X128>;

def FmaFromTensorLayout {
  int CONV_8X4 = 0;
  int CONV_4X4 = 1;
  int CONV_4X2 = 2;
  int DEQUANT_4X4 = 3;
  int DEQUANT_4X2 = 4;
}

class Fma_From_Tensor_Profile<list<ValueType> ArgTy, int OpLayout> : VOPMProfile<ArgTy, 1, 1, 0, 0> {
  let Outs64 = (outs DstRC64:$vdst);
  let Ins64 = (ins Src0RC64:$src0, Src1RC64:$src1, Src2RC64:$src2, Src3RC64:$src3, AuxData:$aux_data, Clamp0:$clamp);

  dag FmaFromTensorInPat = (ins Src0VT:$src0, Src1VT:$src1, Src2VT:$src2, Src3VT:$src3, timm:$aux_data, i1:$clamp);
  dag FmaFromTensorOutPat = (ins Src0VT:$src0, Src1VT:$src1, Src2VT:$src2, Src3VT:$src3, timm:$aux_data, i1:$clamp);

  let Asm64 = "$vdst, $src0, $src1, $src2, $src3$aux_data$clamp";

  int Layout = OpLayout;
}

multiclass Fma_From_Tensor_Profile_Reg<int OutAccReg, int ResidualReg, int ScaleReg, string OutAccInBaseTy, string ResidualBaseTy, int OpLayout> {

  defvar OutputBaseTy = OutAccInBaseTy;
  defvar AccBaseTy = OutAccInBaseTy;
  defvar ScaleBaseTy = OutAccInBaseTy;

  defvar OutAccScaleMult = !if(!eq(!find(OutAccInBaseTy, "16"), -1), 1, 2);
  defvar ResidualMult = !if(!eq(!find(ResidualBaseTy, "16"), -1), 1, 2);

  defvar OutElCount = !mul(OutAccScaleMult, OutAccReg);
  defvar AccElCount = !mul(OutAccScaleMult, OutAccReg);
  defvar ScaleElCount = !mul(OutAccScaleMult, ScaleReg);
  defvar ResidualElCount = !mul(ResidualMult, ResidualReg);

  defvar OutElPrefix = !if(!eq(OutElCount, 1), "", "v" # OutElCount);
  defvar AccElPrefix = !if(!eq(AccElCount, 1), "", "v" # AccElCount);
  defvar ScaleElPrefix = !if(!eq(ScaleElCount, 1), "", "v" # ScaleElCount);
  defvar ResidualElPrefix = !if(!eq(ResidualElCount, 1), "", "v" # ResidualElCount);

  defvar OutputFullTy = !cast<ValueType>(OutElPrefix # OutputBaseTy);
  defvar AccFullTy = !cast<ValueType>(AccElPrefix # AccBaseTy);
  defvar ScaleInFullTy = !cast<ValueType>(ScaleElPrefix # ScaleBaseTy);
  defvar ResidualFullTy = !cast<ValueType>(ResidualElPrefix # ResidualBaseTy);

  def NAME : Fma_From_Tensor_Profile<[OutputFullTy, AccFullTy, ResidualFullTy, ResidualFullTy, ScaleInFullTy], OpLayout>;
}

multiclass Fma_From_Tensor_Profile_Multi<string OutAccBaseTy, string ResidualBaseTy> {

  // 4x2 with 16-bit accumulators require 2 registers for accumulator/output, not 4.
  defvar OutAcc_Reg_4x2 = !if(!eq(!find(OutAccBaseTy, "16"), -1), 4, 2);
  defvar OutAcc_Reg_Nx4 = !if(!eq(OutAccBaseTy, ResidualBaseTy), 2, 4);

  defm _regular_8x4 : Fma_From_Tensor_Profile_Reg<OutAcc_Reg_Nx4, 1, 1, OutAccBaseTy, ResidualBaseTy, FmaFromTensorLayout.CONV_8X4>;
  defm _regular_4x4 : Fma_From_Tensor_Profile_Reg<OutAcc_Reg_Nx4, 1, 1, OutAccBaseTy, ResidualBaseTy, FmaFromTensorLayout.CONV_4X4>;
  defm _regular_4x2 : Fma_From_Tensor_Profile_Reg<OutAcc_Reg_4x2, 1, 1, OutAccBaseTy, ResidualBaseTy, FmaFromTensorLayout.CONV_4X2>;

  defm _dequant_4x4 : Fma_From_Tensor_Profile_Reg<             1, 1, 1, OutAccBaseTy, ResidualBaseTy, FmaFromTensorLayout.DEQUANT_4X4>;
  defm _dequant_4x2 : Fma_From_Tensor_Profile_Reg<             1, 1, 1, OutAccBaseTy, ResidualBaseTy, FmaFromTensorLayout.DEQUANT_4X2>;
}

defm VOPM_FMA_FROM_TENSOR_F32_4B_OR_8B : Fma_From_Tensor_Profile_Multi<"f32", "i32">;
defm VOPM_FMA_FROM_TENSOR_F16_4B_OR_8B : Fma_From_Tensor_Profile_Multi<"f16", "i32">;
defm VOPM_FMA_FROM_TENSOR_BF16_4B_OR_8B : Fma_From_Tensor_Profile_Multi<"bf16", "i32">;

defm VOPM_FMA_FROM_TENSOR_F32_F16 : Fma_From_Tensor_Profile_Multi<"f32", "f16">;
defm VOPM_FMA_FROM_TENSOR_F16_F16 : Fma_From_Tensor_Profile_Multi<"f16", "f16">;
defm VOPM_FMA_FROM_TENSOR_BF16_F16 : Fma_From_Tensor_Profile_Multi<"bf16", "f16">;

defm VOPM_FMA_FROM_TENSOR_F32_BF16 : Fma_From_Tensor_Profile_Multi<"f32", "bf16">;
defm VOPM_FMA_FROM_TENSOR_F16_BF16 : Fma_From_Tensor_Profile_Multi<"f16", "bf16">;
defm VOPM_FMA_FROM_TENSOR_BF16_BF16 : Fma_From_Tensor_Profile_Multi<"bf16", "bf16">;

multiclass VOPM_Fma_From_Tensor_Shape_Pseudos<string ProfilPrefix> {
  if !eq(!find(ProfilPrefix, "dequant"), -1) then
  def _8x4 : VOPM_Pseudo<!cast<Fma_From_Tensor_Profile>(ProfilPrefix # "_8x4")>;
  def _4x4 : VOPM_Pseudo<!cast<Fma_From_Tensor_Profile>(ProfilPrefix # "_4x4")>;
  def _4x2 : VOPM_Pseudo<!cast<Fma_From_Tensor_Profile>(ProfilPrefix # "_4x2")>;
}

multiclass VOPM_Fma_From_Tensor_Deq_Pseudos<string ProfilPrefix> {
  defm _dequant : VOPM_Fma_From_Tensor_Shape_Pseudos<ProfilPrefix # "_dequant">;
  defm _regular : VOPM_Fma_From_Tensor_Shape_Pseudos<ProfilPrefix # "_regular">;
}

multiclass VOPM_Fma_From_Tensor_Resid_Pseudos<string ProfilePrefix> {
  defm _I4 : VOPM_Fma_From_Tensor_Deq_Pseudos<ProfilePrefix # "_4B_OR_8B">;
  defm _U4 : VOPM_Fma_From_Tensor_Deq_Pseudos<ProfilePrefix # "_4B_OR_8B">;
  defm _I8 : VOPM_Fma_From_Tensor_Deq_Pseudos<ProfilePrefix # "_4B_OR_8B">;
  defm _U8 : VOPM_Fma_From_Tensor_Deq_Pseudos<ProfilePrefix # "_4B_OR_8B">;
  defm _FP8 : VOPM_Fma_From_Tensor_Deq_Pseudos<ProfilePrefix # "_4B_OR_8B">;
  defm _BF8 : VOPM_Fma_From_Tensor_Deq_Pseudos<ProfilePrefix # "_4B_OR_8B">;

  defm _F16 : VOPM_Fma_From_Tensor_Deq_Pseudos<ProfilePrefix # "_F16">;
  defm _BF16 : VOPM_Fma_From_Tensor_Deq_Pseudos<ProfilePrefix # "_BF16">;
}

multiclass VOPM_Fma_From_Tensor_MultiPseudos<string ProfilePrefix> {
  defm _F32 : VOPM_Fma_From_Tensor_Resid_Pseudos<ProfilePrefix # "_F32">;
  defm _F16 : VOPM_Fma_From_Tensor_Resid_Pseudos<ProfilePrefix # "_F16">;
  defm _BF16 : VOPM_Fma_From_Tensor_Resid_Pseudos<ProfilePrefix # "_BF16">;
}

defm V_FMA_FROM_TENSOR : VOPM_Fma_From_Tensor_MultiPseudos<"VOPM_FMA_FROM_TENSOR">;

//===----------------------------------------------------------------------===//
// VOPM Reals
//===----------------------------------------------------------------------===//

class VOPM_Real <VOPM_Pseudo ps, int EncodingFamily, string real_name = ps.Mnemonic > :
  VOP_Real <ps>,
  InstSI <ps.OutOperandList, ps.InOperandList, real_name # ps.AsmOperands, []>,
  SIMCInstr <ps.PseudoInstr, EncodingFamily> {

  let VALU = 1;
  let isPseudo = 0;
  let isCodeGenOnly = 0;
  let IsPacked = 1;

  let Constraints     = ps.Constraints;
  let DisableEncoding = ps.DisableEncoding;


  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let OtherPredicates    = ps.OtherPredicates;
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let AsmVariantName     = ps.AsmVariantName;
  let DisableEncoding    = ps.DisableEncoding;
  let TSFlags            = ps.TSFlags;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let Uses                 = ps.Uses;
  let Defs                 = ps.Defs;
  let SchedRW              = ps.SchedRW;
  let mayLoad              = ps.mayLoad;
  let mayStore             = ps.mayStore;
  let TRANS                = ps.TRANS;
  let Constraints     = ps.Constraints;

  VOPProfile Pfl = ps.Pfl;
}

class VOPM_Real_Gen <VOPM_Pseudo ps, GFXGen Gen, string real_name = ps.Mnemonic> :
  VOPM_Real <ps, Gen.Subtarget, real_name> {
  let AssemblerPredicate = Gen.AssemblerPredicate;
  let DecoderNamespace = Gen.DecoderNamespace;
}

multiclass VOP2M_Real_Base<GFXGen Gen, bits<8> op,
                           string backing_ps_name = NAME,
                           string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP2Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;
  }
}

multiclass VOP2M_Real_gfx13 <bits<8> op> : VOP2M_Real_Base <GFX13Gen, op>;

multiclass VOP3M_Real_Base<GFXGen Gen, bits<8> op,
                           string backing_ps_name = NAME,
                           string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP3Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;
  }
}

multiclass VOP3M_Cvt_To_Tensor_Real_Base<GFXGen Gen, bits<8> op,
                                string backing_ps_name = NAME,
                                string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP3Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;

      defvar PrimTy = !cast<Cvt_To_Tensor_Profile>(ps.Pfl).PrimTy;
      let Inst{80-79} = PrimTy;
  }
}

multiclass VOP3M_SBA_Real_Base<GFXGen Gen, bits<8> op,
                               string backing_ps_name = NAME,
                               string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP3Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;

      defvar PixelShape4x2 = !cast<SBA_Profile>(ps.Pfl).PixelShape4x2;
      let Inst{59} = !if(PixelShape4x2, aux_data{1}, 0);
  }
}

multiclass VOP3M_Real_gfx13 <bits<8> op> : VOP3M_Real_Base <GFX13Gen, op>;

multiclass VOP5M_Real_Base<GFXGen Gen, bits<8> op,
                           string backing_ps_name = NAME,
                           string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP5Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;
  }
}


multiclass VOP5M_Cvt_To_Tensor_Real_Base<GFXGen Gen, bits<8> op,
                           string backing_ps_name = NAME,
                           string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP5Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;

      defvar PrimTy = !cast<Cvt_To_Tensor_Profile>(ps.Pfl).PrimTy;
      let Inst{80-79} = PrimTy;
  }
}

multiclass VOP5M_Conv_Real_Base<GFXGen Gen, bits<8> op,
                                string backing_ps_name = NAME,
                                string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP5Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;

      defvar PrimTy = !cast<Conv_Profile>(ps.Pfl).PrimTy;
      let Inst{59-58} = PrimTy;
      defvar FilterTy = !cast<Conv_Profile>(ps.Pfl).FilterTy;
      let Inst{61} = FilterTy;
  }
}

multiclass VOP5M_Conv_Real_gfx13<bits<8> op, string psName> {
  defm _8x4 : VOP5M_Conv_Real_Base<GFX13Gen, op, psName#"_3x3_8x4", !tolower(psName)>;
  defm _4x4 : VOP5M_Conv_Real_Base<GFX13Gen, op, psName#"_3x3_4x4", !tolower(psName)>;
  defm _4x2 : VOP5M_Conv_Real_Base<GFX13Gen, op, psName#"_3x3_4x2", !tolower(psName)>;
}

multiclass VOP5M_Real_gfx13 <bits<8> op> : VOP5M_Real_Base <GFX13Gen, op>;

multiclass VOP6M_Conv_Real_Base<GFXGen Gen, bits<8> op,
                                string backing_ps_name = NAME,
                                string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP6Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;

      defvar PrimTy = !cast<Conv_Profile>(ps.Pfl).PrimTy;
      let Inst{59-58} = PrimTy;
      defvar FilterTy = !cast<Conv_Profile>(ps.Pfl).FilterTy;
      let Inst{61} = FilterTy;
      defvar IterCnt = !cast<Conv_Profile>(ps.Pfl).IterCnt;
      let Inst{83-82} = IterCnt;
  }
}

multiclass VOP6M_Conv_Real_Base_1x1_Iter_gfx13<bits<8> op, string psName, string mnemo> {
  defm _ITER_1: VOP6M_Conv_Real_Base<GFX13Gen, op, psName#"_ITER_1", mnemo>;
  defm _ITER_2: VOP6M_Conv_Real_Base<GFX13Gen, op, psName#"_ITER_2", mnemo>;
  defm _ITER_3: VOP6M_Conv_Real_Base<GFX13Gen, op, psName#"_ITER_3", mnemo>;
  defm _ITER_4: VOP6M_Conv_Real_Base<GFX13Gen, op, psName#"_ITER_4", mnemo>;
}

multiclass VOP6M_Conv_Real_gfx13<bits<8> op, string psName> {
  defm _8x4 : VOP6M_Conv_Real_Base_1x1_Iter_gfx13<op, psName#"_1x1_8x4", !tolower(psName)>;
  defm _4x4 : VOP6M_Conv_Real_Base_1x1_Iter_gfx13<op, psName#"_1x1_4x4", !tolower(psName)>;
  defm _4x2 : VOP6M_Conv_Real_Base_1x1_Iter_gfx13<op, psName#"_1x1_4x2", !tolower(psName)>;
}

multiclass Conv_Real_gfx13<bits<8> op> {
  defm _3x3 : VOP5M_Conv_Real_gfx13<op, NAME>;
  defm _1x1 : VOP6M_Conv_Real_gfx13<op, NAME>;
}

multiclass VOP3M_WMMA_Real_Base<GFXGen Gen, bits<8> op,
                                string backing_ps_name = NAME,
                                string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP3Me<op, !cast<VOPMProfile>(ps.Pfl)> {

      bits<11> srcC;
      bits<11> srcA;
      bits<10> srcB;

      let Inst{39-32} = srcC{7-0};
      let Inst{42} = srcC{8};

      let Inst{52-45} = srcA{7-0};
      let Inst{55} = srcA{8};

      let Inst{71-64} = srcB{7-0};

      bit signed_a;
      bit signed_b;

      let AsmString = asmName # ps.Pfl.Asm64;

      defvar WMMAPfl = !cast<VOPM_WMMA_Profile>(ps.Pfl);
      defvar KMultiplier = WMMAPfl.KMultiplier;
      let Inst{60-58} = KMultiplier;

      let Inst{76} = !if(WMMAPfl.isInt, signed_a, ?);
      let Inst{79} = !if(WMMAPfl.isInt, signed_b, ?);
  }
}

multiclass VOP3M_WMMA_Real_gfx13 <bits<8> op, string asmName = !cast<VOPM_Pseudo>(NAME # "_twoaddr").Mnemonic> {
  // Do not copy constraints from pseudo.
  let Constraints = "" in
    defm _twoaddr : VOP3M_WMMA_Real_Base<GFX13Gen, op, NAME # "_twoaddr", asmName>;
}

multiclass VOP3M_Real_SWMMAC<GFXGen Gen, bits<8> op,
                                string backing_ps_name = NAME,
                                string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP3Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;

      bit signed_a;
      bit signed_b;

      bits<1> index_key_16bit;
      let Inst{77} = index_key_16bit{0};

      defvar WMMAPfl = !cast<VOPM_SWMMAC_Profile>(ps.Pfl);
      let Inst{60-58} = WMMAPfl.KMultiplier;

      let Inst{61} = !if(WMMAPfl.isInt, signed_a, ?);
      let Inst{62} = !if(WMMAPfl.isInt, signed_b, ?);
  }
}

multiclass VOP3M_SWMMAC_Real_gfx13 <bits<8> op, string asmName = !cast<VOPM_Pseudo>(NAME # "_twoaddr").Mnemonic> {
  defm _twoaddr : VOP3M_Real_SWMMAC <GFX13Gen, op, NAME # "_twoaddr", asmName>;
}

multiclass VOP5M_Fma_From_Tensor_Real_Base<GFXGen Gen, bits<8> op,
                                           string backing_ps_name = NAME,
                                           string asmName = !cast<VOPM_Pseudo>(NAME).Mnemonic> {
  defvar ps = !cast<VOPM_Pseudo>(backing_ps_name);

  def Gen.Suffix :
    VOPM_Real_Gen<ps, Gen, asmName>,
    VOP5Me<op, !cast<VOPMProfile>(ps.Pfl)> {
      let AsmString = asmName # ps.Pfl.Asm64;

      defvar Layout = !cast<Fma_From_Tensor_Profile>(ps.Pfl).Layout;
      let Inst{60-58} = Layout;

      let Inst{95-88} = SGPR_NULL_gfx11plus.Index;
  }
}

defm V_PERMUTE_PAIR_2SRC_ROTATE_GROUP_B32 : VOP2M_Real_gfx13 <0x0>;
defm V_PERMUTE_PAIR_GENSGPR_B32           : VOP2M_Real_gfx13 <0x1>;
defm V_PERMUTE_PAIR_BCAST_B32             : VOP2M_Real_gfx13 <0x2>;
defm V_BPERMUTE_B32                       : VOP2M_Real_gfx13 <0x3>;
defm V_MOV_2SRC_B64                       : VOP2M_Real_gfx13 <0x4>;

defm V_PERMUTE_PAIR_2SRC_INTERLEAVE_B64   : VOP3M_Real_gfx13 <0x10>;
defm V_PERMUTE_PACK_TENSOR_2SRC_B64       : VOP3M_Real_gfx13 <0x11>;

defm V_CONVOLVE_F32_IU4    : Conv_Real_gfx13 <0xe0>;
defm V_CONVOLVE_F32_IU8    : Conv_Real_gfx13 <0xe1>;
defm V_CONVOLVE_F32_FP8    : Conv_Real_gfx13 <0xe2>;
defm V_CONVOLVE_F32_BF8    : Conv_Real_gfx13 <0xe3>;
defm V_CONVOLVE_F32_F16    : Conv_Real_gfx13 <0xe4>;
defm V_CONVOLVE_F32_BF16   : Conv_Real_gfx13 <0xe5>;
defm V_CONVOLVE_F16_IU4    : Conv_Real_gfx13 <0xe6>;
defm V_CONVOLVE_F16_IU8    : Conv_Real_gfx13 <0xe7>;
defm V_CONVOLVE_F16_FP8    : Conv_Real_gfx13 <0xe8>;
defm V_CONVOLVE_BF16_BF8   : Conv_Real_gfx13 <0xe9>;
defm V_CONVOLVE_F16_F16    : Conv_Real_gfx13 <0xea>;
defm V_CONVOLVE_BF16_BF16  : Conv_Real_gfx13 <0xeb>;
defm V_CONVOLVE_I32_IU4    : Conv_Real_gfx13 <0xec>;
defm V_CONVOLVE_I32_IU8    : Conv_Real_gfx13 <0xed>;
defm V_CONVOLVE_F32I32_IU4 : Conv_Real_gfx13 <0xee>;
defm V_CONVOLVE_F32I32_IU8 : Conv_Real_gfx13 <0xef>;

multiclass Cvt_To_Tensor_4b_Real_gfx13<bits<8> op> {
  defm _8x4 : VOP3M_Cvt_To_Tensor_Real_Base<GFX13Gen, op, NAME#"_8x4", !tolower(NAME)>;
  defm _4x4 : VOP3M_Cvt_To_Tensor_Real_Base<GFX13Gen, op, NAME#"_4x4", !tolower(NAME)>;
  defm _4x2 : VOP3M_Cvt_To_Tensor_Real_Base<GFX13Gen, op, NAME#"_4x2", !tolower(NAME)>;
}

multiclass Cvt_To_Tensor_8b_Real_gfx13<bits<8> op> {
  defm _8x4 : VOP3M_Cvt_To_Tensor_Real_Base<GFX13Gen, op, NAME#"_8x4", !tolower(NAME)>;
  defm _4x4 : VOP3M_Cvt_To_Tensor_Real_Base<GFX13Gen, op, NAME#"_4x4", !tolower(NAME)>;
  defm _4x2 : VOP3M_Cvt_To_Tensor_Real_Base<GFX13Gen, op, NAME#"_4x2", !tolower(NAME)>;
}

multiclass Cvt_To_Tensor_16b_Real_gfx13<bits<8> op> {
  defm _8x4 : VOP5M_Cvt_To_Tensor_Real_Base<GFX13Gen, op, NAME#"_8x4", !tolower(NAME)>;
  defm _4x4 : VOP5M_Cvt_To_Tensor_Real_Base<GFX13Gen, op, NAME#"_4x4", !tolower(NAME)>;
  defm _4x2 : VOP3M_Cvt_To_Tensor_Real_Base<GFX13Gen, op, NAME#"_4x2", !tolower(NAME)>;
}

defm V_CVT_TO_TENSOR_I4_F32 : Cvt_To_Tensor_4b_Real_gfx13<0x30>;
defm V_CVT_TO_TENSOR_U4_F32 : Cvt_To_Tensor_4b_Real_gfx13<0x31>;
defm V_CVT_TO_TENSOR_I8_F32 : Cvt_To_Tensor_8b_Real_gfx13<0x32>;
defm V_CVT_TO_TENSOR_U8_F32 : Cvt_To_Tensor_8b_Real_gfx13<0x33>;
defm V_CVT_TO_TENSOR_FP8_F32 : Cvt_To_Tensor_8b_Real_gfx13<0x34>;
defm V_CVT_TO_TENSOR_BF8_F32 : Cvt_To_Tensor_8b_Real_gfx13<0x35>;
defm V_CVT_TO_TENSOR_F16_F32 : Cvt_To_Tensor_16b_Real_gfx13<0x36>;
defm V_CVT_TO_TENSOR_BF16_F32 : Cvt_To_Tensor_16b_Real_gfx13<0x37>;
defm V_CVT_TO_TENSOR_I4_F16 : Cvt_To_Tensor_4b_Real_gfx13<0x38>;
defm V_CVT_TO_TENSOR_U4_F16 : Cvt_To_Tensor_4b_Real_gfx13<0x39>;
defm V_CVT_TO_TENSOR_I8_F16 : Cvt_To_Tensor_8b_Real_gfx13<0x3a>;
defm V_CVT_TO_TENSOR_U8_F16 : Cvt_To_Tensor_8b_Real_gfx13<0x3b>;
defm V_CVT_TO_TENSOR_FP8_F16 : Cvt_To_Tensor_8b_Real_gfx13<0x3c>;
defm V_CVT_TO_TENSOR_BF8_F16 : Cvt_To_Tensor_8b_Real_gfx13<0x3d>;
defm V_CVT_TO_TENSOR_F16_F16 : Cvt_To_Tensor_16b_Real_gfx13<0x3e>;
defm V_CVT_TO_TENSOR_BF16_F16 : Cvt_To_Tensor_16b_Real_gfx13<0x3f>;
defm V_CVT_TO_TENSOR_I4_BF16 : Cvt_To_Tensor_4b_Real_gfx13<0x40>;
defm V_CVT_TO_TENSOR_U4_BF16 : Cvt_To_Tensor_4b_Real_gfx13<0x41>;
defm V_CVT_TO_TENSOR_I8_BF16 : Cvt_To_Tensor_8b_Real_gfx13<0x42>;
defm V_CVT_TO_TENSOR_U8_BF16 : Cvt_To_Tensor_8b_Real_gfx13<0x43>;
defm V_CVT_TO_TENSOR_FP8_BF16 : Cvt_To_Tensor_8b_Real_gfx13<0x44>;
defm V_CVT_TO_TENSOR_BF8_BF16 : Cvt_To_Tensor_8b_Real_gfx13<0x45>;
defm V_CVT_TO_TENSOR_F16_BF16 : Cvt_To_Tensor_16b_Real_gfx13<0x46>;
defm V_CVT_TO_TENSOR_BF16_BF16 : Cvt_To_Tensor_16b_Real_gfx13<0x47>;

multiclass SBA<string Suffix, bits<8> op> {
  defm NAME # "_" # Suffix : VOP3M_SBA_Real_Base<GFX13Gen, op>;
  defm NAME # "_SCATTER2" # "_" # Suffix : VOP3M_SBA_Real_Base<GFX13Gen, op>;
  defm NAME # "_SCATTER4" # "_" # Suffix : VOP5M_Real_gfx13<op>;
}

defm V_SCALE_BIAS_ACTIVATE_F32 : VOP3M_SBA_Real_Base <GFX13Gen, 0x20>;
defm V_SCALE_BIAS_ACTIVATE : SBA<"F16", 0x21>;
defm V_SCALE_BIAS_ACTIVATE : SBA<"BF16", 0x22>;

defm V_UNIFORM_SCALE_ACTIVATE_F32 : VOP3M_SBA_Real_Base <GFX13Gen, 0x25>;
defm V_UNIFORM_SCALE_ACTIVATE : SBA<"F16", 0x26>;
defm V_UNIFORM_SCALE_ACTIVATE : SBA<"BF16", 0x27>;

multiclass VOP3M_WMMA_K1_K2_Real<bits<8> op> {
  defm !subst("_16X16_", "_16X16X16_", NAME) : VOP3M_WMMA_Real_gfx13<op, !tolower(NAME)>;
  defm !subst("_16X16_", "_16X16X32_", NAME) : VOP3M_WMMA_Real_gfx13<op, !tolower(NAME)>;
}

multiclass VOP3M_WMMA_K1_K2_K4_Real<bits<8> op> {
  defm !subst("_16X16_", "_16X16X16_", NAME) : VOP3M_WMMA_Real_gfx13<op, !tolower(NAME)>;
  defm !subst("_16X16_", "_16X16X32_", NAME) : VOP3M_WMMA_Real_gfx13<op, !tolower(NAME)>;
  defm !subst("_16X16_", "_16X16X64_", NAME) : VOP3M_WMMA_Real_gfx13<op, !tolower(NAME)>;
}

defm V_WMMA_F32_16X16X16_F16     : VOP3M_WMMA_Real_gfx13 <0x70>;
defm V_WMMA_F16_16X16X16_F16     : VOP3M_WMMA_Real_gfx13 <0x71>;
defm V_WMMA_F32_16X16X16_BF16    : VOP3M_WMMA_Real_gfx13 <0x72>;
defm V_WMMA_BF16_16X16X16_BF16   : VOP3M_WMMA_Real_gfx13 <0x73>;
defm V_WMMA_F32_16X16_FP8_FP8    : VOP3M_WMMA_K1_K2_Real <0x74>;
defm V_WMMA_F32_16X16_FP8_BF8    : VOP3M_WMMA_K1_K2_Real <0x75>;
defm V_WMMA_F32_16X16_BF8_FP8    : VOP3M_WMMA_K1_K2_Real <0x76>;
defm V_WMMA_F32_16X16_BF8_BF8    : VOP3M_WMMA_K1_K2_Real <0x77>;
defm V_WMMA_F16_16X16_FP8_FP8    : VOP3M_WMMA_K1_K2_Real <0x78>;
defm V_WMMA_F16_16X16_FP8_BF8    : VOP3M_WMMA_K1_K2_Real <0x79>;
defm V_WMMA_F16_16X16_BF8_FP8    : VOP3M_WMMA_K1_K2_Real <0x7a>;
defm V_WMMA_F16_16X16_BF8_BF8    : VOP3M_WMMA_K1_K2_Real <0x7b>;
defm V_WMMA_I32_16X16_IU8        : VOP3M_WMMA_K1_K2_Real <0x7c>;
defm V_WMMA_F32_16X16_IU8        : VOP3M_WMMA_K1_K2_Real <0x7d>;
defm V_WMMA_I32_16X16_IU4        : VOP3M_WMMA_K1_K2_K4_Real <0x7e>;
defm V_WMMA_F32_16X16_IU4        : VOP3M_WMMA_K1_K2_K4_Real <0x7f>;
defm V_WMMA_F32I32_16X16_IU8     : VOP3M_WMMA_K1_K2_Real <0x80>;
defm V_WMMA_F32I32_16X16_IU4     : VOP3M_WMMA_K1_K2_K4_Real <0x81>;

multiclass VOP3M_SWMMAC_K1_K2_Real<bits<8> op> {
  defm !subst("_16X16_", "_16X16X32_", NAME) : VOP3M_SWMMAC_Real_gfx13<op, !tolower(NAME)>;
  defm !subst("_16X16_", "_16X16X64_", NAME) : VOP3M_SWMMAC_Real_gfx13<op, !tolower(NAME)>;
}

multiclass VOP3M_SWMMAC_K1_K2_K4_Real<bits<8> op> {
  defm !subst("_16X16_", "_16X16X32_", NAME) : VOP3M_SWMMAC_Real_gfx13<op, !tolower(NAME)>;
  defm !subst("_16X16_", "_16X16X64_", NAME) : VOP3M_SWMMAC_Real_gfx13<op, !tolower(NAME)>;
  defm !subst("_16X16_", "_16X16X128_", NAME) : VOP3M_SWMMAC_Real_gfx13<op, !tolower(NAME)>;
}

defm V_SWMMAC_F32_16X16X32_F16   : VOP3M_SWMMAC_Real_gfx13 <0x83>;
defm V_SWMMAC_F16_16X16X32_F16   : VOP3M_SWMMAC_Real_gfx13 <0x84>;
defm V_SWMMAC_F32_16X16X32_BF16  : VOP3M_SWMMAC_Real_gfx13 <0x85>;
defm V_SWMMAC_BF16_16X16X32_BF16 : VOP3M_SWMMAC_Real_gfx13 <0x86>;
defm V_SWMMAC_F32_16X16_FP8_FP8  : VOP3M_SWMMAC_K1_K2_Real <0x87>;
defm V_SWMMAC_F32_16X16_FP8_BF8  : VOP3M_SWMMAC_K1_K2_Real <0x88>;
defm V_SWMMAC_F32_16X16_BF8_FP8  : VOP3M_SWMMAC_K1_K2_Real <0x89>;
defm V_SWMMAC_F32_16X16_BF8_BF8  : VOP3M_SWMMAC_K1_K2_Real <0x8a>;
defm V_SWMMAC_F16_16X16_FP8_FP8  : VOP3M_SWMMAC_K1_K2_Real <0x8b>;
defm V_SWMMAC_F16_16X16_FP8_BF8  : VOP3M_SWMMAC_K1_K2_Real <0x8c>;
defm V_SWMMAC_F16_16X16_BF8_FP8  : VOP3M_SWMMAC_K1_K2_Real <0x8d>;
defm V_SWMMAC_F16_16X16_BF8_BF8  : VOP3M_SWMMAC_K1_K2_Real <0x8e>;
defm V_SWMMAC_I32_16X16_IU8      : VOP3M_SWMMAC_K1_K2_Real <0x8f>;
defm V_SWMMAC_F32_16X16_IU8      : VOP3M_SWMMAC_K1_K2_Real <0x90>;
defm V_SWMMAC_I32_16X16_IU4      : VOP3M_SWMMAC_K1_K2_K4_Real <0x91>;
defm V_SWMMAC_F32_16X16_IU4      : VOP3M_SWMMAC_K1_K2_K4_Real <0x92>;
defm V_SWMMAC_F32I32_16X16_IU8   : VOP3M_SWMMAC_K1_K2_Real <0x93>;
defm V_SWMMAC_F32I32_16X16_IU4   : VOP3M_SWMMAC_K1_K2_K4_Real <0x94>;

// TODO-GFX13: Add support for VOP3 encoding.
multiclass FMA_Tensor_Real_Shape<bits<8> op, string psName, string mnemo> {
  if !eq(!find(psName, "dequant"), -1) then
  defm _8x4 : VOP5M_Fma_From_Tensor_Real_Base<GFX13Gen, op, psName # "_8x4", mnemo>;
  defm _4x4 : VOP5M_Fma_From_Tensor_Real_Base<GFX13Gen, op, psName # "_4x4", mnemo>;
  defm _4x2 : VOP5M_Fma_From_Tensor_Real_Base<GFX13Gen, op, psName # "_4x2", mnemo>;
}

multiclass FMA_Tensor_Real<bits<8> op> {
  defm _dequant : FMA_Tensor_Real_Shape<op, NAME # "_dequant", !tolower(NAME)>;
  defm _regular : FMA_Tensor_Real_Shape<op, NAME # "_regular", !tolower(NAME)>;
}

defm V_FMA_FROM_TENSOR_F32_I4 : FMA_Tensor_Real <0x50>;
defm V_FMA_FROM_TENSOR_F32_U4 : FMA_Tensor_Real <0x51>;
defm V_FMA_FROM_TENSOR_F32_I8 : FMA_Tensor_Real <0x52>;
defm V_FMA_FROM_TENSOR_F32_U8 : FMA_Tensor_Real <0x53>;
defm V_FMA_FROM_TENSOR_F32_FP8 : FMA_Tensor_Real <0x54>;
defm V_FMA_FROM_TENSOR_F32_BF8 : FMA_Tensor_Real <0x55>;
defm V_FMA_FROM_TENSOR_F32_F16 : FMA_Tensor_Real <0x56>;
defm V_FMA_FROM_TENSOR_F32_BF16 : FMA_Tensor_Real <0x57>;
defm V_FMA_FROM_TENSOR_F16_I4 : FMA_Tensor_Real <0x58>;
defm V_FMA_FROM_TENSOR_F16_U4 : FMA_Tensor_Real <0x59>;
defm V_FMA_FROM_TENSOR_F16_I8 : FMA_Tensor_Real <0x5a>;
defm V_FMA_FROM_TENSOR_F16_U8 : FMA_Tensor_Real <0x5b>;
defm V_FMA_FROM_TENSOR_F16_FP8 : FMA_Tensor_Real <0x5c>;
defm V_FMA_FROM_TENSOR_F16_BF8 : FMA_Tensor_Real <0x5d>;
defm V_FMA_FROM_TENSOR_F16_F16 : FMA_Tensor_Real <0x5e>;
defm V_FMA_FROM_TENSOR_BF16_I4 : FMA_Tensor_Real <0x60>;
defm V_FMA_FROM_TENSOR_BF16_U4 : FMA_Tensor_Real <0x61>;
defm V_FMA_FROM_TENSOR_BF16_I8 : FMA_Tensor_Real <0x62>;
defm V_FMA_FROM_TENSOR_BF16_U8 : FMA_Tensor_Real <0x63>;
defm V_FMA_FROM_TENSOR_BF16_FP8 : FMA_Tensor_Real <0x64>;
defm V_FMA_FROM_TENSOR_BF16_BF8 : FMA_Tensor_Real <0x65>;
defm V_FMA_FROM_TENSOR_BF16_BF16 : FMA_Tensor_Real <0x67>;

//===----------------------------------------------------------------------===//
// VOPM Patterns
//===----------------------------------------------------------------------===//


class 1X1_IT_1<SDPatternOperator node, Conv_Profile P> : PatFrag<P.ConvFrag, !setdagop(P.ConvFrag, node), [{ return (((cast<ConstantSDNode>(N->getOperand(N->getNumOperands()-2)))->getZExtValue() >> 12)&3) == 0; }] >;
class 1X1_IT_2<SDPatternOperator node, Conv_Profile P> : PatFrag<P.ConvFrag, !setdagop(P.ConvFrag, node), [{ return (((cast<ConstantSDNode>(N->getOperand(N->getNumOperands()-2)))->getZExtValue() >> 12)&3) == 1; }] >;
class 1X1_IT_3<SDPatternOperator node, Conv_Profile P> : PatFrag<P.ConvFrag, !setdagop(P.ConvFrag, node), [{ return (((cast<ConstantSDNode>(N->getOperand(N->getNumOperands()-2)))->getZExtValue() >> 12)&3) == 2; }] >;
class 1X1_IT_4<SDPatternOperator node, Conv_Profile P> : PatFrag<P.ConvFrag, !setdagop(P.ConvFrag, node), [{ return (((cast<ConstantSDNode>(N->getOperand(N->getNumOperands()-2)))->getZExtValue() >> 12)&3) == 3; }] >;

multiclass ConvPat_VOPM<string Inst, SDPatternOperator node, string ProfileName> {
  defvar P = !cast<Conv_Profile>(ProfileName);

  def : GCNPat <(P.DstVT !setdagop(P.ConvPat, node)),
                (P.DstVT !setdagop(P.ConvPat, !cast<Instruction>(Inst)))>;
}

multiclass ConvPat1x1Iter_VOPM<string Inst, SDPatternOperator node, string ProfilePrefix> {

  defvar P1 = !cast<Conv_Profile>(ProfilePrefix # "_ITER_1");
  def : GCNPat <(P1.DstVT !setdagop(P1.ConvPat, 1X1_IT_1<node, P1>)),
                (P1.DstVT !setdagop(P1.ConvPat, !cast<Instruction>(Inst # "_ITER_1")))>;

  defvar P2 = !cast<Conv_Profile>(ProfilePrefix # "_ITER_2");
  def : GCNPat <(P2.DstVT !setdagop(P2.ConvPat, 1X1_IT_2<node, P2>)),
                (P2.DstVT !setdagop(P2.ConvPat, !cast<Instruction>(Inst # "_ITER_2")))>;

  defvar P3 = !cast<Conv_Profile>(ProfilePrefix # "_ITER_3");
  def : GCNPat <(P3.DstVT !setdagop(P3.ConvPat, 1X1_IT_3<node, P3>)),
                (P3.DstVT !setdagop(P3.ConvPat, !cast<Instruction>(Inst # "_ITER_3")))>;

  defvar P4 = !cast<Conv_Profile>(ProfilePrefix # "_ITER_4");
  def : GCNPat <(P4.DstVT !setdagop(P4.ConvPat, 1X1_IT_4<node, P4>)),
                (P4.DstVT !setdagop(P4.ConvPat, !cast<Instruction>(Inst # "_ITER_4")))>;
}

multiclass ConvPatMulti1x1_VOPM<string Inst, SDPatternOperator node, string ProfilePrefix> {
  defm : ConvPat1x1Iter_VOPM<Inst # "_8x4", node, ProfilePrefix # "_8x4">;
  defm : ConvPat1x1Iter_VOPM<Inst # "_4x4", node, ProfilePrefix # "_4x4">;
  defm : ConvPat1x1Iter_VOPM<Inst # "_4x2", node, ProfilePrefix # "_4x2">;
}

multiclass ConvPatMulti3x3_VOPM<string Inst, SDPatternOperator node, string ProfilePrefix> {
  defm : ConvPat_VOPM<Inst # "_8x4", node, ProfilePrefix # "_8x4">;
  defm : ConvPat_VOPM<Inst # "_4x4", node, ProfilePrefix # "_4x4">;
  defm : ConvPat_VOPM<Inst # "_4x2", node, ProfilePrefix # "_4x2">;
}

multiclass ConvPatMultiFilters_VOPM<string TypeNames, string NodePrefix> {
  defm _1x1 : ConvPatMulti1x1_VOPM<"V_CONVOLVE_" # TypeNames # "_1x1", !cast<SDPatternOperator>(NodePrefix # "_1x1"), "VOPM_CONV_" # TypeNames # "_1x1">;
  defm _3x3 : ConvPatMulti3x3_VOPM<"V_CONVOLVE_" # TypeNames # "_3x3", !cast<SDPatternOperator>(NodePrefix # "_3x3"), "VOPM_CONV_" # TypeNames # "_3x3">;
}

defm : ConvPatMultiFilters_VOPM<"F16_F16", "int_amdgcn_convolve_f16_f16">;
defm : ConvPatMultiFilters_VOPM<"BF16_BF16", "int_amdgcn_convolve_bf16_bf16">;
defm : ConvPatMultiFilters_VOPM<"F16_FP8", "int_amdgcn_convolve_f16_fp8">;
defm : ConvPatMultiFilters_VOPM<"BF16_BF8", "int_amdgcn_convolve_bf16_bf8">;
defm : ConvPatMultiFilters_VOPM<"F16_IU8", "int_amdgcn_convolve_f16_iu8">;
defm : ConvPatMultiFilters_VOPM<"F16_IU4", "int_amdgcn_convolve_f16_iu4">;

defm : ConvPatMultiFilters_VOPM<"F32_F16", "int_amdgcn_convolve_f32_f16">;
defm : ConvPatMultiFilters_VOPM<"F32_BF16", "int_amdgcn_convolve_f32_bf16">;
defm : ConvPatMultiFilters_VOPM<"F32_FP8", "int_amdgcn_convolve_f32_fp8">;
defm : ConvPatMultiFilters_VOPM<"F32_BF8", "int_amdgcn_convolve_f32_bf8">;
defm : ConvPatMultiFilters_VOPM<"F32_IU8", "int_amdgcn_convolve_f32_iu8">;
defm : ConvPatMultiFilters_VOPM<"F32_IU4", "int_amdgcn_convolve_f32_iu4">;
defm : ConvPatMultiFilters_VOPM<"I32_IU8", "int_amdgcn_convolve_i32_iu8">;
defm : ConvPatMultiFilters_VOPM<"I32_IU4", "int_amdgcn_convolve_i32_iu4">;
defm : ConvPatMultiFilters_VOPM<"F32I32_IU8", "int_amdgcn_convolve_f32i32_iu8">;
defm : ConvPatMultiFilters_VOPM<"F32I32_IU4", "int_amdgcn_convolve_f32i32_iu4">;

multiclass WMMAPat_VOPM<string Inst, SDPatternOperator node, VOPM_WMMA_Profile P> {
  def : GCNPat <(P.DstVT !setdagop(P.WmmaInPat, node)),
                (P.DstVT !setdagop(P.WmmaOutPat, !cast<Instruction>(Inst#"_twoaddr")))>;
  let AddedComplexity = 4 in
  def : GCNPat <(P.DstVT !setdagop(P.WmmaInlineInPat, node)),
                (P.DstVT !setdagop(P.WmmaInlineOutPat, !cast<Instruction>(Inst#"_threeaddr")))>;
}

class SWMMACPat_VOPM<Instruction Inst, SDPatternOperator node, VOPM_SWMMAC_Profile P> :
  GCNPat <(P.DstVT !setdagop(P.SwmmacInPat, node)),
          (P.DstVT !setdagop(P.SwmmacOutPat, Inst))>;

let SubtargetPredicate = isGFX13Plus in {
  defm : WMMAPat_VOPM<"V_WMMA_BF16_16X16X16_BF16",   int_amdgcn_wmma_bf16_16x16x16_bf16_clamp,   VOPM_WMMA_BF16_BF16_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F16_16X16X16_F16",     int_amdgcn_wmma_f16_16x16x16_f16_clamp,     VOPM_WMMA_F16_F16_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X16_BF16",    int_amdgcn_wmma_f32_16x16x16_bf16_clamp,    VOPM_WMMA_F32_BF16_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X16_F16",     int_amdgcn_wmma_f32_16x16x16_f16_clamp,     VOPM_WMMA_F32_F16_X16>;

  defm : WMMAPat_VOPM<"V_WMMA_F16_16X16X16_BF8_BF8", int_amdgcn_wmma_f16_16x16x16_bf8_bf8_clamp, VOPM_WMMA_F16_F8_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F16_16X16X16_BF8_FP8", int_amdgcn_wmma_f16_16x16x16_bf8_fp8_clamp, VOPM_WMMA_F16_F8_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F16_16X16X16_FP8_BF8", int_amdgcn_wmma_f16_16x16x16_fp8_bf8_clamp, VOPM_WMMA_F16_F8_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F16_16X16X16_FP8_FP8", int_amdgcn_wmma_f16_16x16x16_fp8_fp8_clamp, VOPM_WMMA_F16_F8_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X16_BF8_BF8", int_amdgcn_wmma_f32_16x16x16_bf8_bf8_clamp, VOPM_WMMA_F32_F8_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X16_BF8_FP8", int_amdgcn_wmma_f32_16x16x16_bf8_fp8_clamp, VOPM_WMMA_F32_F8_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X16_FP8_BF8", int_amdgcn_wmma_f32_16x16x16_fp8_bf8_clamp, VOPM_WMMA_F32_F8_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X16_FP8_FP8", int_amdgcn_wmma_f32_16x16x16_fp8_fp8_clamp, VOPM_WMMA_F32_F8_X16>;

  defm : WMMAPat_VOPM<"V_WMMA_F16_16X16X32_BF8_BF8", int_amdgcn_wmma_f16_16x16x32_bf8_bf8_clamp, VOPM_WMMA_F16_F8_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_F16_16X16X32_BF8_FP8", int_amdgcn_wmma_f16_16x16x32_bf8_fp8_clamp, VOPM_WMMA_F16_F8_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_F16_16X16X32_FP8_BF8", int_amdgcn_wmma_f16_16x16x32_fp8_bf8_clamp, VOPM_WMMA_F16_F8_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_F16_16X16X32_FP8_FP8", int_amdgcn_wmma_f16_16x16x32_fp8_fp8_clamp, VOPM_WMMA_F16_F8_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X32_BF8_BF8", int_amdgcn_wmma_f32_16x16x32_bf8_bf8_clamp, VOPM_WMMA_F32_F8_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X32_BF8_FP8", int_amdgcn_wmma_f32_16x16x32_bf8_fp8_clamp, VOPM_WMMA_F32_F8_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X32_FP8_BF8", int_amdgcn_wmma_f32_16x16x32_fp8_bf8_clamp, VOPM_WMMA_F32_F8_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X32_FP8_FP8", int_amdgcn_wmma_f32_16x16x32_fp8_fp8_clamp, VOPM_WMMA_F32_F8_X32>;

  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X16_IU4", int_amdgcn_wmma_f32_16x16x16_iu4_clamp,         VOPM_WMMA_F32_IU4_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X16_IU8", int_amdgcn_wmma_f32_16x16x16_iu8_clamp,         VOPM_WMMA_F32_IU8_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X32_IU4", int_amdgcn_wmma_f32_16x16x32_iu4_clamp,         VOPM_WMMA_F32_IU4_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X32_IU8", int_amdgcn_wmma_f32_16x16x32_iu8_clamp,         VOPM_WMMA_F32_IU8_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_F32_16X16X64_IU4", int_amdgcn_wmma_f32_16x16x64_iu4_clamp,         VOPM_WMMA_F32_IU4_X64>;
  defm : WMMAPat_VOPM<"V_WMMA_F32I32_16X16X16_IU4", int_amdgcn_wmma_f32i32_16x16x16_iu4_clamp,   VOPM_WMMA_F32I32_IU4_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F32I32_16X16X16_IU8", int_amdgcn_wmma_f32i32_16x16x16_iu8_clamp,   VOPM_WMMA_F32I32_IU8_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_F32I32_16X16X32_IU4", int_amdgcn_wmma_f32i32_16x16x32_iu4_clamp,   VOPM_WMMA_F32I32_IU4_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_F32I32_16X16X32_IU8", int_amdgcn_wmma_f32i32_16x16x32_iu8_clamp,   VOPM_WMMA_F32I32_IU8_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_F32I32_16X16X64_IU4", int_amdgcn_wmma_f32i32_16x16x64_iu4_clamp,   VOPM_WMMA_F32I32_IU4_X64>;
  defm : WMMAPat_VOPM<"V_WMMA_I32_16X16X16_IU4", int_amdgcn_wmma_i32_16x16x16_iu4_clamp,         VOPM_WMMA_I32_IU4_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_I32_16X16X16_IU8", int_amdgcn_wmma_i32_16x16x16_iu8_clamp,         VOPM_WMMA_I32_IU8_X16>;
  defm : WMMAPat_VOPM<"V_WMMA_I32_16X16X32_IU4", int_amdgcn_wmma_i32_16x16x32_iu4_clamp,         VOPM_WMMA_I32_IU4_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_I32_16X16X32_IU8", int_amdgcn_wmma_i32_16x16x32_iu8_clamp,         VOPM_WMMA_I32_IU8_X32>;
  defm : WMMAPat_VOPM<"V_WMMA_I32_16X16X64_IU4", int_amdgcn_wmma_i32_16x16x64_iu4_clamp,         VOPM_WMMA_I32_IU4_X64>;

  def : SWMMACPat_VOPM<V_SWMMAC_BF16_16X16X32_BF16_twoaddr, int_amdgcn_swmmac_bf16_16x16x32_bf16_clamp,     VOPM_SWMMAC_BF16_BF16_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F16_16X16X32_F16_twoaddr,   int_amdgcn_swmmac_f16_16x16x32_f16_clamp,       VOPM_SWMMAC_F16_F16_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X32_BF16_twoaddr,  int_amdgcn_swmmac_f32_16x16x32_bf16_clamp,      VOPM_SWMMAC_F32_BF16_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X32_F16_twoaddr,   int_amdgcn_swmmac_f32_16x16x32_f16_clamp,       VOPM_SWMMAC_F32_F16_X32>;

  def : SWMMACPat_VOPM<V_SWMMAC_F16_16X16X32_BF8_BF8_twoaddr, int_amdgcn_swmmac_f16_16x16x32_bf8_bf8_clamp, VOPM_SWMMAC_F16_F8_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F16_16X16X32_BF8_FP8_twoaddr, int_amdgcn_swmmac_f16_16x16x32_bf8_fp8_clamp, VOPM_SWMMAC_F16_F8_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F16_16X16X32_FP8_BF8_twoaddr, int_amdgcn_swmmac_f16_16x16x32_fp8_bf8_clamp, VOPM_SWMMAC_F16_F8_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F16_16X16X32_FP8_FP8_twoaddr, int_amdgcn_swmmac_f16_16x16x32_fp8_fp8_clamp, VOPM_SWMMAC_F16_F8_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F16_16X16X64_BF8_BF8_twoaddr, int_amdgcn_swmmac_f16_16x16x64_bf8_bf8_clamp, VOPM_SWMMAC_F16_F8_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_F16_16X16X64_BF8_FP8_twoaddr, int_amdgcn_swmmac_f16_16x16x64_bf8_fp8_clamp, VOPM_SWMMAC_F16_F8_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_F16_16X16X64_FP8_BF8_twoaddr, int_amdgcn_swmmac_f16_16x16x64_fp8_bf8_clamp, VOPM_SWMMAC_F16_F8_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_F16_16X16X64_FP8_FP8_twoaddr, int_amdgcn_swmmac_f16_16x16x64_fp8_fp8_clamp, VOPM_SWMMAC_F16_F8_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X32_BF8_BF8_twoaddr, int_amdgcn_swmmac_f32_16x16x32_bf8_bf8_clamp, VOPM_SWMMAC_F32_F8_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X32_BF8_FP8_twoaddr, int_amdgcn_swmmac_f32_16x16x32_bf8_fp8_clamp, VOPM_SWMMAC_F32_F8_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X32_FP8_BF8_twoaddr, int_amdgcn_swmmac_f32_16x16x32_fp8_bf8_clamp, VOPM_SWMMAC_F32_F8_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X32_FP8_FP8_twoaddr, int_amdgcn_swmmac_f32_16x16x32_fp8_fp8_clamp, VOPM_SWMMAC_F32_F8_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X64_BF8_BF8_twoaddr, int_amdgcn_swmmac_f32_16x16x64_bf8_bf8_clamp, VOPM_SWMMAC_F32_F8_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X64_BF8_FP8_twoaddr, int_amdgcn_swmmac_f32_16x16x64_bf8_fp8_clamp, VOPM_SWMMAC_F32_F8_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X64_FP8_BF8_twoaddr, int_amdgcn_swmmac_f32_16x16x64_fp8_bf8_clamp, VOPM_SWMMAC_F32_F8_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X64_FP8_FP8_twoaddr, int_amdgcn_swmmac_f32_16x16x64_fp8_fp8_clamp, VOPM_SWMMAC_F32_F8_X64>;

  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X32_IU4_twoaddr, int_amdgcn_swmmac_f32_16x16x32_iu4_clamp,         VOPM_SWMMAC_F32_IU4_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X32_IU8_twoaddr, int_amdgcn_swmmac_f32_16x16x32_iu8_clamp,         VOPM_SWMMAC_F32_IU8_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X64_IU4_twoaddr, int_amdgcn_swmmac_f32_16x16x64_iu4_clamp,         VOPM_SWMMAC_F32_IU4_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X64_IU8_twoaddr, int_amdgcn_swmmac_f32_16x16x64_iu8_clamp,         VOPM_SWMMAC_F32_IU8_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32_16X16X128_IU4_twoaddr, int_amdgcn_swmmac_f32_16x16x128_iu4_clamp,       VOPM_SWMMAC_F32_IU4_X128>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32I32_16X16X32_IU4_twoaddr, int_amdgcn_swmmac_f32i32_16x16x32_iu4_clamp,   VOPM_SWMMAC_F32I32_IU4_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32I32_16X16X32_IU8_twoaddr, int_amdgcn_swmmac_f32i32_16x16x32_iu8_clamp,   VOPM_SWMMAC_F32I32_IU8_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32I32_16X16X64_IU4_twoaddr, int_amdgcn_swmmac_f32i32_16x16x64_iu4_clamp,   VOPM_SWMMAC_F32I32_IU4_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32I32_16X16X64_IU8_twoaddr, int_amdgcn_swmmac_f32i32_16x16x64_iu8_clamp,   VOPM_SWMMAC_F32I32_IU8_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_F32I32_16X16X128_IU4_twoaddr, int_amdgcn_swmmac_f32i32_16x16x128_iu4_clamp, VOPM_SWMMAC_F32I32_IU4_X128>;
  def : SWMMACPat_VOPM<V_SWMMAC_I32_16X16X32_IU4_twoaddr, int_amdgcn_swmmac_i32_16x16x32_iu4_clamp,         VOPM_SWMMAC_I32_IU4_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_I32_16X16X32_IU8_twoaddr, int_amdgcn_swmmac_i32_16x16x32_iu8_clamp,         VOPM_SWMMAC_I32_IU8_X32>;
  def : SWMMACPat_VOPM<V_SWMMAC_I32_16X16X64_IU4_twoaddr, int_amdgcn_swmmac_i32_16x16x64_iu4_clamp,         VOPM_SWMMAC_I32_IU4_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_I32_16X16X64_IU8_twoaddr, int_amdgcn_swmmac_i32_16x16x64_iu8_clamp,         VOPM_SWMMAC_I32_IU8_X64>;
  def : SWMMACPat_VOPM<V_SWMMAC_I32_16X16X128_IU4_twoaddr, int_amdgcn_swmmac_i32_16x16x128_iu4_clamp,       VOPM_SWMMAC_I32_IU4_X128>;
}

multiclass FmaFromTensorPat_VOPM<string Inst, SDPatternOperator node, string ProfileName> {
  defvar P = !cast<Fma_From_Tensor_Profile>(ProfileName);

  def : GCNPat <(P.DstVT !setdagop(P.FmaFromTensorInPat, node)),
                (P.DstVT !setdagop(P.FmaFromTensorOutPat, !cast<Instruction>(Inst)))>;
}

multiclass FmaFromTensorPatMulti_VOPM<string Inst, SDPatternOperator node, string ProfilePrefix> {
  if !eq(!find(ProfilePrefix, "dequant"), -1) then
  defm : FmaFromTensorPat_VOPM<Inst # "_8x4", node, ProfilePrefix # "_8x4">;
  defm : FmaFromTensorPat_VOPM<Inst # "_4x4", node, ProfilePrefix # "_4x4">;
  defm : FmaFromTensorPat_VOPM<Inst # "_4x2", node, ProfilePrefix # "_4x2">;
}

multiclass FmaFromTensorPatMultiDeq_VOPM<string TypeNames, string NodePrefix, string ProfileSuffix> {
  defm _dequant : FmaFromTensorPatMulti_VOPM<"V_FMA_FROM_TENSOR_" # TypeNames # "_dequant", !cast<SDPatternOperator>(NodePrefix), "VOPM_FMA_FROM_TENSOR_" # ProfileSuffix # "_dequant">;
  defm _regular : FmaFromTensorPatMulti_VOPM<"V_FMA_FROM_TENSOR_" # TypeNames # "_regular", !cast<SDPatternOperator>(NodePrefix), "VOPM_FMA_FROM_TENSOR_" # ProfileSuffix # "_regular">;
}

defm : FmaFromTensorPatMultiDeq_VOPM<"F32_I4", "int_amdgcn_fma_from_tensor_f32_i4", "F32_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F32_U4", "int_amdgcn_fma_from_tensor_f32_u4", "F32_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F32_I8", "int_amdgcn_fma_from_tensor_f32_i8", "F32_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F32_U8", "int_amdgcn_fma_from_tensor_f32_u8", "F32_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F32_FP8", "int_amdgcn_fma_from_tensor_f32_fp8", "F32_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F32_BF8", "int_amdgcn_fma_from_tensor_f32_bf8", "F32_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F32_F16", "int_amdgcn_fma_from_tensor_f32_f16", "F32_F16">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F32_BF16", "int_amdgcn_fma_from_tensor_f32_bf16", "F32_BF16">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F16_I4", "int_amdgcn_fma_from_tensor_f16_i4", "F16_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F16_U4", "int_amdgcn_fma_from_tensor_f16_u4", "F16_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F16_I8", "int_amdgcn_fma_from_tensor_f16_i8", "F16_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F16_U8", "int_amdgcn_fma_from_tensor_f16_u8", "F16_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F16_FP8", "int_amdgcn_fma_from_tensor_f16_fp8", "F16_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F16_BF8", "int_amdgcn_fma_from_tensor_f16_bf8", "F16_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F16_F16", "int_amdgcn_fma_from_tensor_f16_f16", "F16_F16">;
defm : FmaFromTensorPatMultiDeq_VOPM<"F16_BF16", "int_amdgcn_fma_from_tensor_f16_bf16", "F16_BF16">;
defm : FmaFromTensorPatMultiDeq_VOPM<"BF16_I4", "int_amdgcn_fma_from_tensor_bf16_i4", "BF16_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"BF16_U4", "int_amdgcn_fma_from_tensor_bf16_u4", "BF16_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"BF16_I8", "int_amdgcn_fma_from_tensor_bf16_i8", "BF16_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"BF16_U8", "int_amdgcn_fma_from_tensor_bf16_u8", "BF16_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"BF16_FP8", "int_amdgcn_fma_from_tensor_bf16_fp8", "BF16_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"BF16_BF8", "int_amdgcn_fma_from_tensor_bf16_bf8", "BF16_4B_OR_8B">;
defm : FmaFromTensorPatMultiDeq_VOPM<"BF16_F16", "int_amdgcn_fma_from_tensor_bf16_f16", "BF16_F16">;
defm : FmaFromTensorPatMultiDeq_VOPM<"BF16_BF16", "int_amdgcn_fma_from_tensor_bf16_bf16", "BF16_BF16">;
