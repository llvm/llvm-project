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

// For all FMA opcodes declared in fma3p_rm and fma3s_rm milticlasses defined
// below, both the register and memory variants are commutable.
// For the register form the commutable operands are 1, 2 and 3.
// For the memory variant the folded operand must be in 3. Thus,
// in that case, only the operands 1 and 2 can be swapped.
// Commuting some of operands may require the opcode change.
// FMA*213*:
//   operands 1 and 2 (memory & register forms): *213* --> *213*(no changes);
//   operands 1 and 3 (register forms only):     *213* --> *231*;
//   operands 2 and 3 (register forms only):     *213* --> *132*.
// FMA*132*:
//   operands 1 and 2 (memory & register forms): *132* --> *231*;
//   operands 1 and 3 (register forms only):     *132* --> *132*(no changes);
//   operands 2 and 3 (register forms only):     *132* --> *213*.
// FMA*231*:
//   operands 1 and 2 (memory & register forms): *231* --> *132*;
//   operands 1 and 3 (register forms only):     *231* --> *213*;
//   operands 2 and 3 (register forms only):     *231* --> *231*(no changes).

let Constraints = "$src1 = $dst", hasSideEffects = 0, isCommutable = 1 in
multiclass fma3p_rm<bits<8> opc, string OpcodeStr,
                    PatFrag MemFrag128, PatFrag MemFrag256,
                    ValueType OpVT128, ValueType OpVT256,
                    SDPatternOperator Op = null_frag> {
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

  def Yr    : FMA3<opc, MRMSrcReg, (outs VR256:$dst),
                   (ins VR256:$src1, VR256:$src2, VR256:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR256:$dst, (OpVT256 (Op VR256:$src2, VR256:$src1,
                                               VR256:$src3)))]>, VEX_L;

  let mayLoad = 1 in
  def Ym    : FMA3<opc, MRMSrcMem, (outs VR256:$dst),
                   (ins VR256:$src1, VR256:$src2, f256mem:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR256:$dst,
                     (OpVT256 (Op VR256:$src2, VR256:$src1,
                               (MemFrag256 addr:$src3))))]>, VEX_L;
}

multiclass fma3p_forms<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                       string OpcodeStr, string PackTy, string Suff,
                       PatFrag MemFrag128, PatFrag MemFrag256,
                       SDNode Op, ValueType OpTy128, ValueType OpTy256> {
  defm NAME#213#Suff : fma3p_rm<opc213,
                                !strconcat(OpcodeStr, "213", PackTy),
                                MemFrag128, MemFrag256, OpTy128, OpTy256, Op>;
  defm NAME#132#Suff : fma3p_rm<opc132,
                                !strconcat(OpcodeStr, "132", PackTy),
                                MemFrag128, MemFrag256, OpTy128, OpTy256>;
  defm NAME#231#Suff : fma3p_rm<opc231,
                                !strconcat(OpcodeStr, "231", PackTy),
                                MemFrag128, MemFrag256, OpTy128, OpTy256>;
}

// Fused Multiply-Add
let ExeDomain = SSEPackedSingle in {
  defm VFMADD    : fma3p_forms<0x98, 0xA8, 0xB8, "vfmadd", "ps", "PS",
                               loadv4f32, loadv8f32, X86Fmadd, v4f32, v8f32>;
  defm VFMSUB    : fma3p_forms<0x9A, 0xAA, 0xBA, "vfmsub", "ps", "PS",
                               loadv4f32, loadv8f32, X86Fmsub, v4f32, v8f32>;
  defm VFMADDSUB : fma3p_forms<0x96, 0xA6, 0xB6, "vfmaddsub", "ps", "PS",
                               loadv4f32, loadv8f32, X86Fmaddsub,
                               v4f32, v8f32>;
  defm VFMSUBADD : fma3p_forms<0x97, 0xA7, 0xB7, "vfmsubadd", "ps", "PS",
                               loadv4f32, loadv8f32, X86Fmsubadd,
                               v4f32, v8f32>;
}

let ExeDomain = SSEPackedDouble in {
  defm VFMADD    : fma3p_forms<0x98, 0xA8, 0xB8, "vfmadd", "pd", "PD",
                               loadv2f64, loadv4f64, X86Fmadd, v2f64,
                               v4f64>, VEX_W;
  defm VFMSUB    : fma3p_forms<0x9A, 0xAA, 0xBA, "vfmsub", "pd", "PD",
                               loadv2f64, loadv4f64, X86Fmsub, v2f64,
                               v4f64>, VEX_W;
  defm VFMADDSUB : fma3p_forms<0x96, 0xA6, 0xB6, "vfmaddsub", "pd", "PD",
                               loadv2f64, loadv4f64, X86Fmaddsub,
                               v2f64, v4f64>, VEX_W;
  defm VFMSUBADD : fma3p_forms<0x97, 0xA7, 0xB7, "vfmsubadd", "pd", "PD",
                               loadv2f64, loadv4f64, X86Fmsubadd,
                               v2f64, v4f64>, VEX_W;
}

// Fused Negative Multiply-Add
let ExeDomain = SSEPackedSingle in {
  defm VFNMADD : fma3p_forms<0x9C, 0xAC, 0xBC, "vfnmadd", "ps", "PS", loadv4f32,
                             loadv8f32, X86Fnmadd, v4f32, v8f32>;
  defm VFNMSUB : fma3p_forms<0x9E, 0xAE, 0xBE, "vfnmsub", "ps", "PS", loadv4f32,
                             loadv8f32, X86Fnmsub, v4f32, v8f32>;
}
let ExeDomain = SSEPackedDouble in {
  defm VFNMADD : fma3p_forms<0x9C, 0xAC, 0xBC, "vfnmadd", "pd", "PD", loadv2f64,
                             loadv4f64, X86Fnmadd, v2f64, v4f64>, VEX_W;
  defm VFNMSUB : fma3p_forms<0x9E, 0xAE, 0xBE, "vfnmsub", "pd", "PD", loadv2f64,
                             loadv4f64, X86Fnmsub, v2f64, v4f64>, VEX_W;
}

// All source register operands of FMA opcodes defined in fma3s_rm multiclass
// can be commuted. In many cases such commute transformation requres an opcode
// adjustment, for example, commuting the operands 1 and 2 in FMA*132 form
// would require an opcode change to FMA*231:
//     FMA*132* reg1, reg2, reg3; // reg1 * reg3 + reg2;
//     -->
//     FMA*231* reg2, reg1, reg3; // reg1 * reg3 + reg2;
// Please see more detailed comment at the very beginning of the section
// defining FMA3 opcodes above.
let Constraints = "$src1 = $dst", isCommutable = 1, hasSideEffects = 0 in
multiclass fma3s_rm<bits<8> opc, string OpcodeStr,
                    X86MemOperand x86memop, RegisterClass RC,
                    SDPatternOperator OpNode = null_frag> {
  def r     : FMA3<opc, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, RC:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst, (OpNode RC:$src2, RC:$src1, RC:$src3))]>;

  let mayLoad = 1 in
  def m     : FMA3<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, x86memop:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpNode RC:$src2, RC:$src1, (load addr:$src3)))]>;
}

// These FMA*_Int instructions are defined specially for being used when
// the scalar FMA intrinsics are lowered to machine instructions, and in that
// sense, they are similar to existing ADD*_Int, SUB*_Int, MUL*_Int, etc.
// instructions.
//
// All of the FMA*_Int opcodes are defined as commutable here.
// Commuting the 2nd and 3rd source register operands of FMAs is quite trivial
// and the corresponding optimizations have been developed.
// Commuting the 1st operand of FMA*_Int requires some additional analysis,
// the commute optimization is legal only if all users of FMA*_Int use only
// the lowest element of the FMA*_Int instruction. Even though such analysis
// may be not implemented yet we allow the routines doing the actual commute
// transformation to decide if one or another instruction is commutable or not.
let Constraints = "$src1 = $dst", isCommutable = 1, isCodeGenOnly = 1,
    hasSideEffects = 0 in
multiclass fma3s_rm_int<bits<8> opc, string OpcodeStr,
                        Operand memopr, RegisterClass RC> {
  def r_Int : FMA3<opc, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, RC:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   []>;

  let mayLoad = 1 in
  def m_Int : FMA3<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, memopr:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   []>;
}

multiclass fma3s_forms<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                       string OpStr, string PackTy, string Suff,
                       SDNode OpNode, RegisterClass RC,
                       X86MemOperand x86memop> {
  defm NAME#132#Suff : fma3s_rm<opc132, !strconcat(OpStr, "132", PackTy),
                                x86memop, RC>;
  defm NAME#213#Suff : fma3s_rm<opc213, !strconcat(OpStr, "213", PackTy),
                                x86memop, RC, OpNode>;
  defm NAME#231#Suff : fma3s_rm<opc231, !strconcat(OpStr, "231", PackTy),
                                x86memop, RC>;
}

// The FMA 213 form is created for lowering of scalar FMA intrinscis
// to machine instructions.
// The FMA 132 form can trivially be get by commuting the 2nd and 3rd operands
// of FMA 213 form.
// The FMA 231 form can be get only by commuting the 1st operand of 213 or 132
// forms and is possible only after special analysis of all uses of the initial
// instruction. Such analysis do not exist yet and thus introducing the 231
// form of FMA*_Int instructions is done using an optimistic assumption that
// such analysis will be implemented eventually.
multiclass fma3s_int_forms<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                           string OpStr, string PackTy, string Suff,
                           RegisterClass RC, Operand memop> {
  defm NAME#132#Suff : fma3s_rm_int<opc132, !strconcat(OpStr, "132", PackTy),
                                    memop, RC>;
  defm NAME#213#Suff : fma3s_rm_int<opc213, !strconcat(OpStr, "213", PackTy),
                                    memop, RC>;
  defm NAME#231#Suff : fma3s_rm_int<opc231, !strconcat(OpStr, "231", PackTy),
                                    memop, RC>;
}

multiclass fma3s<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                 string OpStr, Intrinsic IntF32, Intrinsic IntF64,
                 SDNode OpNode> {
  let ExeDomain = SSEPackedSingle in
  defm NAME : fma3s_forms<opc132, opc213, opc231, OpStr, "ss", "SS", OpNode,
                          FR32, f32mem>,
              fma3s_int_forms<opc132, opc213, opc231, OpStr, "ss", "SS",
                              VR128, ssmem>;

  let ExeDomain = SSEPackedDouble in
  defm NAME : fma3s_forms<opc132, opc213, opc231, OpStr, "sd", "SD", OpNode,
                        FR64, f64mem>,
              fma3s_int_forms<opc132, opc213, opc231, OpStr, "sd", "SD",
                              VR128, sdmem>, VEX_W;

  // These patterns use the 123 ordering, instead of 213, even though
  // they match the intrinsic to the 213 version of the instruction.
  // This is because src1 is tied to dest, and the scalar intrinsics
  // require the pass-through values to come from the first source
  // operand, not the second.
  def : Pat<(IntF32 VR128:$src1, VR128:$src2, VR128:$src3),
            (COPY_TO_REGCLASS(!cast<Instruction>(NAME#"213SSr_Int")
             $src1, $src2, $src3), VR128)>;

  def : Pat<(IntF64 VR128:$src1, VR128:$src2, VR128:$src3),
            (COPY_TO_REGCLASS(!cast<Instruction>(NAME#"213SDr_Int")
             $src1, $src2, $src3), VR128)>;
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
             (OpVT (OpNode RC:$src1, RC:$src2, RC:$src3)))]>, VEX_W, VEX_LIG, MemOp4;
  def rm : FMA4<opc, MRMSrcMem, (outs RC:$dst),
           (ins RC:$src1, RC:$src2, x86memop:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set RC:$dst, (OpNode RC:$src1, RC:$src2,
                           (mem_frag addr:$src3)))]>, VEX_W, VEX_LIG, MemOp4;
  def mr : FMA4<opc, MRMSrcMem, (outs RC:$dst),
           (ins RC:$src1, x86memop:$src2, RC:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set RC:$dst,
             (OpNode RC:$src1, (mem_frag addr:$src2), RC:$src3))]>, VEX_LIG;
// For disassembler
let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0 in
  def rr_REV : FMA4<opc, MRMSrcReg, (outs RC:$dst),
               (ins RC:$src1, RC:$src2, RC:$src3),
               !strconcat(OpcodeStr,
               "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"), []>,
               VEX_LIG;
}

multiclass fma4s_int<bits<8> opc, string OpcodeStr, Operand memop,
                     ComplexPattern mem_cpat, Intrinsic Int> {
let isCodeGenOnly = 1 in {
  let isCommutable = 1 in
  def rr_Int : FMA4<opc, MRMSrcReg, (outs VR128:$dst),
               (ins VR128:$src1, VR128:$src2, VR128:$src3),
               !strconcat(OpcodeStr,
               "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
               [(set VR128:$dst,
                 (Int VR128:$src1, VR128:$src2, VR128:$src3))]>, VEX_W, VEX_LIG, MemOp4;
  def rm_Int : FMA4<opc, MRMSrcMem, (outs VR128:$dst),
               (ins VR128:$src1, VR128:$src2, memop:$src3),
               !strconcat(OpcodeStr,
               "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
               [(set VR128:$dst, (Int VR128:$src1, VR128:$src2,
                                  mem_cpat:$src3))]>, VEX_W, VEX_LIG, MemOp4;
  def mr_Int : FMA4<opc, MRMSrcMem, (outs VR128:$dst),
               (ins VR128:$src1, memop:$src2, VR128:$src3),
               !strconcat(OpcodeStr,
               "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
               [(set VR128:$dst,
                 (Int VR128:$src1, mem_cpat:$src2, VR128:$src3))]>, VEX_LIG;
} // isCodeGenOnly = 1
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
  def Yrr : FMA4<opc, MRMSrcReg, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, VR256:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR256:$dst,
             (OpVT256 (OpNode VR256:$src1, VR256:$src2, VR256:$src3)))]>,
           VEX_W, MemOp4, VEX_L;
  def Yrm : FMA4<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, f256mem:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR256:$dst, (OpNode VR256:$src1, VR256:$src2,
                              (ld_frag256 addr:$src3)))]>, VEX_W, MemOp4, VEX_L;
  def Ymr : FMA4<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, f256mem:$src2, VR256:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR256:$dst, (OpNode VR256:$src1,
                              (ld_frag256 addr:$src2), VR256:$src3))]>, VEX_L;
// For disassembler
let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0 in {
  def rr_REV : FMA4<opc, MRMSrcReg, (outs VR128:$dst),
               (ins VR128:$src1, VR128:$src2, VR128:$src3),
               !strconcat(OpcodeStr,
               "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"), []>;
  def Yrr_REV : FMA4<opc, MRMSrcReg, (outs VR256:$dst),
                (ins VR256:$src1, VR256:$src2, VR256:$src3),
                !strconcat(OpcodeStr,
                "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"), []>,
                VEX_L;
} // isCodeGenOnly = 1
}

let ExeDomain = SSEPackedSingle in {
  // Scalar Instructions
  defm VFMADDSS4  : fma4s<0x6A, "vfmaddss", FR32, f32mem, f32, X86Fmadd, loadf32>,
                    fma4s_int<0x6A, "vfmaddss", ssmem, sse_load_f32,
                              int_x86_fma_vfmadd_ss>;
  defm VFMSUBSS4  : fma4s<0x6E, "vfmsubss", FR32, f32mem, f32, X86Fmsub, loadf32>,
                    fma4s_int<0x6E, "vfmsubss", ssmem, sse_load_f32,
                              int_x86_fma_vfmsub_ss>;
  defm VFNMADDSS4 : fma4s<0x7A, "vfnmaddss", FR32, f32mem, f32,
                          X86Fnmadd, loadf32>,
                    fma4s_int<0x7A, "vfnmaddss", ssmem, sse_load_f32,
                              int_x86_fma_vfnmadd_ss>;
  defm VFNMSUBSS4 : fma4s<0x7E, "vfnmsubss", FR32, f32mem, f32,
                          X86Fnmsub, loadf32>,
                    fma4s_int<0x7E, "vfnmsubss", ssmem, sse_load_f32,
                              int_x86_fma_vfnmsub_ss>;
  // Packed Instructions
  defm VFMADDPS4    : fma4p<0x68, "vfmaddps", X86Fmadd, v4f32, v8f32,
                            loadv4f32, loadv8f32>;
  defm VFMSUBPS4    : fma4p<0x6C, "vfmsubps", X86Fmsub, v4f32, v8f32,
                            loadv4f32, loadv8f32>;
  defm VFNMADDPS4   : fma4p<0x78, "vfnmaddps", X86Fnmadd, v4f32, v8f32,
                            loadv4f32, loadv8f32>;
  defm VFNMSUBPS4   : fma4p<0x7C, "vfnmsubps", X86Fnmsub, v4f32, v8f32,
                            loadv4f32, loadv8f32>;
  defm VFMADDSUBPS4 : fma4p<0x5C, "vfmaddsubps", X86Fmaddsub, v4f32, v8f32,
                            loadv4f32, loadv8f32>;
  defm VFMSUBADDPS4 : fma4p<0x5E, "vfmsubaddps", X86Fmsubadd, v4f32, v8f32,
                            loadv4f32, loadv8f32>;
}

let ExeDomain = SSEPackedDouble in {
  // Scalar Instructions
  defm VFMADDSD4  : fma4s<0x6B, "vfmaddsd", FR64, f64mem, f64, X86Fmadd, loadf64>,
                    fma4s_int<0x6B, "vfmaddsd", sdmem, sse_load_f64,
                              int_x86_fma_vfmadd_sd>;
  defm VFMSUBSD4  : fma4s<0x6F, "vfmsubsd", FR64, f64mem, f64, X86Fmsub, loadf64>,
                    fma4s_int<0x6F, "vfmsubsd", sdmem, sse_load_f64,
                              int_x86_fma_vfmsub_sd>;
  defm VFNMADDSD4 : fma4s<0x7B, "vfnmaddsd", FR64, f64mem, f64,
                          X86Fnmadd, loadf64>,
                    fma4s_int<0x7B, "vfnmaddsd", sdmem, sse_load_f64,
                              int_x86_fma_vfnmadd_sd>;
  defm VFNMSUBSD4 : fma4s<0x7F, "vfnmsubsd", FR64, f64mem, f64,
                          X86Fnmsub, loadf64>,
                    fma4s_int<0x7F, "vfnmsubsd", sdmem, sse_load_f64,
                              int_x86_fma_vfnmsub_sd>;
  // Packed Instructions
  defm VFMADDPD4    : fma4p<0x69, "vfmaddpd", X86Fmadd, v2f64, v4f64,
                            loadv2f64, loadv4f64>;
  defm VFMSUBPD4    : fma4p<0x6D, "vfmsubpd", X86Fmsub, v2f64, v4f64,
                            loadv2f64, loadv4f64>;
  defm VFNMADDPD4   : fma4p<0x79, "vfnmaddpd", X86Fnmadd, v2f64, v4f64,
                            loadv2f64, loadv4f64>;
  defm VFNMSUBPD4   : fma4p<0x7D, "vfnmsubpd", X86Fnmsub, v2f64, v4f64,
                            loadv2f64, loadv4f64>;
  defm VFMADDSUBPD4 : fma4p<0x5D, "vfmaddsubpd", X86Fmaddsub, v2f64, v4f64,
                            loadv2f64, loadv4f64>;
  defm VFMSUBADDPD4 : fma4p<0x5F, "vfmsubaddpd", X86Fmsubadd, v2f64, v4f64,
                            loadv2f64, loadv4f64>;
}

