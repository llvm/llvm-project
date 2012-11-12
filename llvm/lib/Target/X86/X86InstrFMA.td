//===-- X86InstrFMA.td - FMA Instruction Set ---------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes FMA (Fused Multiply-Add) instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FMA3 - Intel 3 operand Fused Multiply-Add instructions
//===----------------------------------------------------------------------===//

let Constraints = "$src1 = $dst" in {
multiclass fma3p_rm<bits<8> opc, string OpcodeStr,
                    PatFrag MemFrag128, PatFrag MemFrag256,
                    ValueType OpVT128, ValueType OpVT256,
                    SDPatternOperator Op = null_frag> {
  let isCommutable = 1 in
  def r     : FMA3<opc, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src1, VR128:$src2, VR128:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, (OpVT128 (Op VR128:$src2,
                                               VR128:$src1, VR128:$src3)))]>;

  let mayLoad = 1 in
  def m     : FMA3<opc, MRMSrcMem, (outs VR128:$dst),
                   (ins VR128:$src1, VR128:$src2, f128mem:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, (OpVT128 (Op VR128:$src2, VR128:$src1,
                                               (MemFrag128 addr:$src3))))]>;

  let isCommutable = 1 in
  def rY    : FMA3<opc, MRMSrcReg, (outs VR256:$dst),
                   (ins VR256:$src1, VR256:$src2, VR256:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR256:$dst, (OpVT256 (Op VR256:$src2, VR256:$src1,
                                               VR256:$src3)))]>, VEX_L;

  let mayLoad = 1 in
  def mY    : FMA3<opc, MRMSrcMem, (outs VR256:$dst),
                   (ins VR256:$src1, VR256:$src2, f256mem:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR256:$dst,
                     (OpVT256 (Op VR256:$src2, VR256:$src1,
                               (MemFrag256 addr:$src3))))]>, VEX_L;
}
} // Constraints = "$src1 = $dst"

multiclass fma3p_forms<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                       string OpcodeStr, string PackTy,
                       PatFrag MemFrag128, PatFrag MemFrag256,
                       SDNode Op, ValueType OpTy128, ValueType OpTy256> {
  defm r213 : fma3p_rm<opc213,
                       !strconcat(OpcodeStr, !strconcat("213", PackTy)),
                       MemFrag128, MemFrag256, OpTy128, OpTy256, Op>;
let neverHasSideEffects = 1 in {
  defm r132 : fma3p_rm<opc132,
                       !strconcat(OpcodeStr, !strconcat("132", PackTy)),
                       MemFrag128, MemFrag256, OpTy128, OpTy256>;
  defm r231 : fma3p_rm<opc231,
                       !strconcat(OpcodeStr, !strconcat("231", PackTy)),
                       MemFrag128, MemFrag256, OpTy128, OpTy256>;
} // neverHasSideEffects = 1
}

// Fused Multiply-Add
let ExeDomain = SSEPackedSingle in {
  defm VFMADDPS    : fma3p_forms<0x98, 0xA8, 0xB8, "vfmadd", "ps", memopv4f32,
                                 memopv8f32, X86Fmadd, v4f32, v8f32>;
  defm VFMSUBPS    : fma3p_forms<0x9A, 0xAA, 0xBA, "vfmsub", "ps", memopv4f32,
                                 memopv8f32, X86Fmsub, v4f32, v8f32>;
  defm VFMADDSUBPS : fma3p_forms<0x96, 0xA6, 0xB6, "vfmaddsub", "ps",
                                 memopv4f32, memopv8f32, X86Fmaddsub,
                                 v4f32, v8f32>;
  defm VFMSUBADDPS : fma3p_forms<0x97, 0xA7, 0xB7, "vfmsubadd", "ps",
                                 memopv4f32, memopv8f32, X86Fmsubadd,
                                 v4f32, v8f32>;
}

let ExeDomain = SSEPackedDouble in {
  defm VFMADDPD    : fma3p_forms<0x98, 0xA8, 0xB8, "vfmadd", "pd", memopv2f64,
                                 memopv4f64, X86Fmadd, v2f64, v4f64>, VEX_W;
  defm VFMSUBPD    : fma3p_forms<0x9A, 0xAA, 0xBA, "vfmsub", "pd", memopv2f64,
                                 memopv4f64, X86Fmsub, v2f64, v4f64>, VEX_W;
  defm VFMADDSUBPD : fma3p_forms<0x96, 0xA6, 0xB6, "vfmaddsub", "pd",
                                 memopv2f64, memopv4f64, X86Fmaddsub,
                                 v2f64, v4f64>, VEX_W;
  defm VFMSUBADDPD : fma3p_forms<0x97, 0xA7, 0xB7, "vfmsubadd", "pd",
                                 memopv2f64, memopv4f64, X86Fmsubadd,
                                 v2f64, v4f64>, VEX_W;
}

// Fused Negative Multiply-Add
let ExeDomain = SSEPackedSingle in {
  defm VFNMADDPS : fma3p_forms<0x9C, 0xAC, 0xBC, "vfnmadd", "ps",  memopv4f32,
                               memopv8f32, X86Fnmadd, v4f32, v8f32>;
  defm VFNMSUBPS : fma3p_forms<0x9E, 0xAE, 0xBE, "vfnmsub", "ps",  memopv4f32,
                               memopv8f32, X86Fnmsub, v4f32, v8f32>;
}
let ExeDomain = SSEPackedDouble in {
  defm VFNMADDPD : fma3p_forms<0x9C, 0xAC, 0xBC, "vfnmadd", "pd", memopv2f64,
                               memopv4f64, X86Fnmadd, v2f64, v4f64>, VEX_W;
  defm VFNMSUBPD : fma3p_forms<0x9E, 0xAE, 0xBE, "vfnmsub", "pd",
                               memopv2f64, memopv4f64, X86Fnmsub, v2f64,
                               v4f64>, VEX_W;
}

let Constraints = "$src1 = $dst" in {
multiclass fma3s_rm<bits<8> opc, string OpcodeStr, X86MemOperand x86memop,
                    RegisterClass RC, ValueType OpVT, PatFrag mem_frag,
                    SDPatternOperator OpNode = null_frag> {
  let isCommutable = 1 in
  def r     : FMA3<opc, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, RC:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (OpNode RC:$src2, RC:$src1, RC:$src3)))]>;
  let mayLoad = 1 in
  def m     : FMA3<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, x86memop:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (OpNode RC:$src2, RC:$src1,
                            (mem_frag addr:$src3))))]>;
}

multiclass fma3s_rm_int<bits<8> opc, string OpcodeStr, Operand memop,
                        ComplexPattern mem_cpat, Intrinsic IntId,
                        RegisterClass RC> {
  let isCommutable = 1 in
  def r_Int : FMA3<opc, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src1, VR128:$src2, VR128:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, (IntId VR128:$src2, VR128:$src1,
                     VR128:$src3))]>;
  def m_Int : FMA3<opc, MRMSrcMem, (outs VR128:$dst),
                   (ins VR128:$src1, VR128:$src2, memop:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst,
                     (IntId VR128:$src2, VR128:$src1, mem_cpat:$src3))]>;
}
} // Constraints = "$src1 = $dst"

multiclass fma3s_forms<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                       string OpStr, string PackTy, Intrinsic Int,
                       SDNode OpNode, RegisterClass RC, ValueType OpVT,
                       X86MemOperand x86memop, Operand memop, PatFrag mem_frag,
                       ComplexPattern mem_cpat> {
let neverHasSideEffects = 1 in {
  defm r132 : fma3s_rm<opc132, !strconcat(OpStr, !strconcat("132", PackTy)),
                       x86memop, RC, OpVT, mem_frag>;
  defm r231 : fma3s_rm<opc231, !strconcat(OpStr, !strconcat("231", PackTy)),
                       x86memop, RC, OpVT, mem_frag>;
}

defm r213 : fma3s_rm<opc213, !strconcat(OpStr, !strconcat("213", PackTy)),
                     x86memop, RC, OpVT, mem_frag, OpNode>,
            fma3s_rm_int<opc213, !strconcat(OpStr, !strconcat("213", PackTy)),
                         memop, mem_cpat, Int, RC>;
}

multiclass fma3s<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                 string OpStr, Intrinsic IntF32, Intrinsic IntF64,
                 SDNode OpNode> {
  defm SS : fma3s_forms<opc132, opc213, opc231, OpStr, "ss", IntF32, OpNode,
                        FR32, f32, f32mem, ssmem, loadf32, sse_load_f32>;
  defm SD : fma3s_forms<opc132, opc213, opc231, OpStr, "sd", IntF64, OpNode,
                        FR64, f64, f64mem, sdmem, loadf64, sse_load_f64>, VEX_W;
}

defm VFMADD : fma3s<0x99, 0xA9, 0xB9, "vfmadd", int_x86_fma_vfmadd_ss,
                    int_x86_fma_vfmadd_sd, X86Fmadd>, VEX_LIG;
defm VFMSUB : fma3s<0x9B, 0xAB, 0xBB, "vfmsub", int_x86_fma_vfmsub_ss,
                    int_x86_fma_vfmsub_sd, X86Fmsub>, VEX_LIG;

defm VFNMADD : fma3s<0x9D, 0xAD, 0xBD, "vfnmadd", int_x86_fma_vfnmadd_ss,
                     int_x86_fma_vfnmadd_sd, X86Fnmadd>, VEX_LIG;
defm VFNMSUB : fma3s<0x9F, 0xAF, 0xBF, "vfnmsub", int_x86_fma_vfnmsub_ss,
                     int_x86_fma_vfnmsub_sd, X86Fnmsub>, VEX_LIG;


//===----------------------------------------------------------------------===//
// FMA4 - AMD 4 operand Fused Multiply-Add instructions
//===----------------------------------------------------------------------===//


multiclass fma4s<bits<8> opc, string OpcodeStr, RegisterClass RC,
                 X86MemOperand x86memop, ValueType OpVT, SDNode OpNode,
                 PatFrag mem_frag> {
  let isCommutable = 1 in
  def rr : FMA4<opc, MRMSrcReg, (outs RC:$dst),
           (ins RC:$src1, RC:$src2, RC:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set RC:$dst,
             (OpVT (OpNode RC:$src1, RC:$src2, RC:$src3)))]>, VEX_W, MemOp4;
  def rm : FMA4<opc, MRMSrcMem, (outs RC:$dst),
           (ins RC:$src1, RC:$src2, x86memop:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set RC:$dst, (OpNode RC:$src1, RC:$src2,
                           (mem_frag addr:$src3)))]>, VEX_W, MemOp4;
  def mr : FMA4<opc, MRMSrcMem, (outs RC:$dst),
           (ins RC:$src1, x86memop:$src2, RC:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set RC:$dst,
             (OpNode RC:$src1, (mem_frag addr:$src2), RC:$src3))]>;
// For disassembler
let isCodeGenOnly = 1 in
  def rr_REV : FMA4<opc, MRMSrcReg, (outs RC:$dst),
               (ins RC:$src1, RC:$src2, RC:$src3),
               !strconcat(OpcodeStr,
               "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"), []>;
}

multiclass fma4s_int<bits<8> opc, string OpcodeStr, Operand memop,
                     ComplexPattern mem_cpat, Intrinsic Int> {
  let isCommutable = 1 in
  def rr_Int : FMA4<opc, MRMSrcReg, (outs VR128:$dst),
               (ins VR128:$src1, VR128:$src2, VR128:$src3),
               !strconcat(OpcodeStr,
               "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
               [(set VR128:$dst,
                 (Int VR128:$src1, VR128:$src2, VR128:$src3))]>, VEX_W, MemOp4;
  def rm_Int : FMA4<opc, MRMSrcMem, (outs VR128:$dst),
               (ins VR128:$src1, VR128:$src2, memop:$src3),
               !strconcat(OpcodeStr,
               "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
               [(set VR128:$dst, (Int VR128:$src1, VR128:$src2,
                                  mem_cpat:$src3))]>, VEX_W, MemOp4;
  def mr_Int : FMA4<opc, MRMSrcMem, (outs VR128:$dst),
               (ins VR128:$src1, memop:$src2, VR128:$src3),
               !strconcat(OpcodeStr,
               "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
               [(set VR128:$dst,
                 (Int VR128:$src1, mem_cpat:$src2, VR128:$src3))]>;
}

multiclass fma4p<bits<8> opc, string OpcodeStr, SDNode OpNode,
                 ValueType OpVT128, ValueType OpVT256,
                 PatFrag ld_frag128, PatFrag ld_frag256> {
  let isCommutable = 1 in
  def rr : FMA4<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst,
             (OpVT128 (OpNode VR128:$src1, VR128:$src2, VR128:$src3)))]>,
           VEX_W, MemOp4;
  def rm : FMA4<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, f128mem:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst, (OpNode VR128:$src1, VR128:$src2,
                              (ld_frag128 addr:$src3)))]>, VEX_W, MemOp4;
  def mr : FMA4<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, f128mem:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst,
             (OpNode VR128:$src1, (ld_frag128 addr:$src2), VR128:$src3))]>;
  let isCommutable = 1 in
  def rrY : FMA4<opc, MRMSrcReg, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, VR256:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR256:$dst,
             (OpVT256 (OpNode VR256:$src1, VR256:$src2, VR256:$src3)))]>,
           VEX_W, MemOp4, VEX_L;
  def rmY : FMA4<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, f256mem:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR256:$dst, (OpNode VR256:$src1, VR256:$src2,
                              (ld_frag256 addr:$src3)))]>, VEX_W, MemOp4, VEX_L;
  def mrY : FMA4<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, f256mem:$src2, VR256:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR256:$dst, (OpNode VR256:$src1,
                              (ld_frag256 addr:$src2), VR256:$src3))]>, VEX_L;
// For disassembler
let isCodeGenOnly = 1 in {
  def rr_REV : FMA4<opc, MRMSrcReg, (outs VR128:$dst),
               (ins VR128:$src1, VR128:$src2, VR128:$src3),
               !strconcat(OpcodeStr,
               "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"), []>;
  def rrY_REV : FMA4<opc, MRMSrcReg, (outs VR256:$dst),
                (ins VR256:$src1, VR256:$src2, VR256:$src3),
                !strconcat(OpcodeStr,
                "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"), []>,
                VEX_L;
} // isCodeGenOnly = 1
}

let Predicates = [HasFMA4] in {

defm VFMADDSS4  : fma4s<0x6A, "vfmaddss", FR32, f32mem, f32, X86Fmadd, loadf32>,
                  fma4s_int<0x6A, "vfmaddss", ssmem, sse_load_f32,
                            int_x86_fma_vfmadd_ss>;
defm VFMADDSD4  : fma4s<0x6B, "vfmaddsd", FR64, f64mem, f64, X86Fmadd, loadf64>,
                  fma4s_int<0x6B, "vfmaddsd", sdmem, sse_load_f64,
                            int_x86_fma_vfmadd_sd>;
defm VFMSUBSS4  : fma4s<0x6E, "vfmsubss", FR32, f32mem, f32, X86Fmsub, loadf32>,
                  fma4s_int<0x6E, "vfmsubss", ssmem, sse_load_f32,
                            int_x86_fma_vfmsub_ss>;
defm VFMSUBSD4  : fma4s<0x6F, "vfmsubsd", FR64, f64mem, f64, X86Fmsub, loadf64>,
                  fma4s_int<0x6F, "vfmsubsd", sdmem, sse_load_f64,
                            int_x86_fma_vfmsub_sd>;
defm VFNMADDSS4 : fma4s<0x7A, "vfnmaddss", FR32, f32mem, f32,
                        X86Fnmadd, loadf32>,
                  fma4s_int<0x7A, "vfnmaddss", ssmem, sse_load_f32,
                            int_x86_fma_vfnmadd_ss>;
defm VFNMADDSD4 : fma4s<0x7B, "vfnmaddsd", FR64, f64mem, f64,
                        X86Fnmadd, loadf64>,
                  fma4s_int<0x7B, "vfnmaddsd", sdmem, sse_load_f64,
                            int_x86_fma_vfnmadd_sd>;
defm VFNMSUBSS4 : fma4s<0x7E, "vfnmsubss", FR32, f32mem, f32,
                        X86Fnmsub, loadf32>,
                  fma4s_int<0x7E, "vfnmsubss", ssmem, sse_load_f32,
                            int_x86_fma_vfnmsub_ss>;
defm VFNMSUBSD4 : fma4s<0x7F, "vfnmsubsd", FR64, f64mem, f64,
                        X86Fnmsub, loadf64>,
                  fma4s_int<0x7F, "vfnmsubsd", sdmem, sse_load_f64,
                            int_x86_fma_vfnmsub_sd>;

defm VFMADDPS4    : fma4p<0x68, "vfmaddps", X86Fmadd, v4f32, v8f32,
                          memopv4f32, memopv8f32>;
defm VFMADDPD4    : fma4p<0x69, "vfmaddpd", X86Fmadd, v2f64, v4f64,
                          memopv2f64, memopv4f64>;
defm VFMSUBPS4    : fma4p<0x6C, "vfmsubps", X86Fmsub, v4f32, v8f32,
                          memopv4f32, memopv8f32>;
defm VFMSUBPD4    : fma4p<0x6D, "vfmsubpd", X86Fmsub, v2f64, v4f64,
                          memopv2f64, memopv4f64>;
defm VFNMADDPS4   : fma4p<0x78, "vfnmaddps", X86Fnmadd, v4f32, v8f32,
                          memopv4f32, memopv8f32>;
defm VFNMADDPD4   : fma4p<0x79, "vfnmaddpd", X86Fnmadd, v2f64, v4f64,
                          memopv2f64, memopv4f64>;
defm VFNMSUBPS4   : fma4p<0x7C, "vfnmsubps", X86Fnmsub, v4f32, v8f32,
                          memopv4f32, memopv8f32>;
defm VFNMSUBPD4   : fma4p<0x7D, "vfnmsubpd", X86Fnmsub, v2f64, v4f64,
                          memopv2f64, memopv4f64>;
defm VFMADDSUBPS4 : fma4p<0x5C, "vfmaddsubps", X86Fmaddsub, v4f32, v8f32,
                          memopv4f32, memopv8f32>;
defm VFMADDSUBPD4 : fma4p<0x5D, "vfmaddsubpd", X86Fmaddsub, v2f64, v4f64,
                          memopv2f64, memopv4f64>;
defm VFMSUBADDPS4 : fma4p<0x5E, "vfmsubaddps", X86Fmsubadd, v4f32, v8f32,
                          memopv4f32, memopv8f32>;
defm VFMSUBADDPD4 : fma4p<0x5F, "vfmsubaddpd", X86Fmsubadd, v2f64, v4f64,
                          memopv2f64, memopv4f64>;
} // HasFMA4

