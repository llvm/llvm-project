//===-- X86InstrXOP.td - XOP Instruction Set ---------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes XOP (eXtended OPerations)
//
//===----------------------------------------------------------------------===//

multiclass xop2op<bits<8> opc, string OpcodeStr, Intrinsic Int, PatFrag memop> {
  def rr : IXOP<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           [(set VR128:$dst, (Int VR128:$src))]>, XOP, Sched<[WritePHAdd]>;
  def rm : IXOP<opc, MRMSrcMem, (outs VR128:$dst), (ins i128mem:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           [(set VR128:$dst, (Int (bitconvert (memop addr:$src))))]>, XOP,
           Sched<[WritePHAddLd, ReadAfterLd]>;
}

let ExeDomain = SSEPackedInt in {
  defm VPHSUBWD  : xop2op<0xE2, "vphsubwd", int_x86_xop_vphsubwd, loadv2i64>;
  defm VPHSUBDQ  : xop2op<0xE3, "vphsubdq", int_x86_xop_vphsubdq, loadv2i64>;
  defm VPHSUBBW  : xop2op<0xE1, "vphsubbw", int_x86_xop_vphsubbw, loadv2i64>;
  defm VPHADDWQ  : xop2op<0xC7, "vphaddwq", int_x86_xop_vphaddwq, loadv2i64>;
  defm VPHADDWD  : xop2op<0xC6, "vphaddwd", int_x86_xop_vphaddwd, loadv2i64>;
  defm VPHADDUWQ : xop2op<0xD7, "vphadduwq", int_x86_xop_vphadduwq, loadv2i64>;
  defm VPHADDUWD : xop2op<0xD6, "vphadduwd", int_x86_xop_vphadduwd, loadv2i64>;
  defm VPHADDUDQ : xop2op<0xDB, "vphaddudq", int_x86_xop_vphaddudq, loadv2i64>;
  defm VPHADDUBW : xop2op<0xD1, "vphaddubw", int_x86_xop_vphaddubw, loadv2i64>;
  defm VPHADDUBQ : xop2op<0xD3, "vphaddubq", int_x86_xop_vphaddubq, loadv2i64>;
  defm VPHADDUBD : xop2op<0xD2, "vphaddubd", int_x86_xop_vphaddubd, loadv2i64>;
  defm VPHADDDQ  : xop2op<0xCB, "vphadddq", int_x86_xop_vphadddq, loadv2i64>;
  defm VPHADDBW  : xop2op<0xC1, "vphaddbw", int_x86_xop_vphaddbw, loadv2i64>;
  defm VPHADDBQ  : xop2op<0xC3, "vphaddbq", int_x86_xop_vphaddbq, loadv2i64>;
  defm VPHADDBD  : xop2op<0xC2, "vphaddbd", int_x86_xop_vphaddbd, loadv2i64>;
}

// Scalar load 2 addr operand instructions
multiclass xop2opsld<bits<8> opc, string OpcodeStr, Intrinsic Int,
                     Operand memop, ComplexPattern mem_cpat> {
  def rr : IXOP<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           [(set VR128:$dst, (Int VR128:$src))]>, XOP, Sched<[WriteFAdd]>;
  def rm : IXOP<opc, MRMSrcMem, (outs VR128:$dst), (ins memop:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           [(set VR128:$dst, (Int (bitconvert mem_cpat:$src)))]>, XOP,
           Sched<[WriteFAddLd, ReadAfterLd]>;
}

multiclass xop2op128<bits<8> opc, string OpcodeStr, Intrinsic Int,
                     PatFrag memop> {
  def rr : IXOP<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           [(set VR128:$dst, (Int VR128:$src))]>, XOP, Sched<[WriteFAdd]>;
  def rm : IXOP<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           [(set VR128:$dst, (Int (bitconvert (memop addr:$src))))]>, XOP,
           Sched<[WriteFAddLd, ReadAfterLd]>;
}

multiclass xop2op256<bits<8> opc, string OpcodeStr, Intrinsic Int,
                     PatFrag memop> {
  def Yrr : IXOP<opc, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           [(set VR256:$dst, (Int VR256:$src))]>, XOP, VEX_L, Sched<[WriteFAdd]>;
  def Yrm : IXOP<opc, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           [(set VR256:$dst, (Int (bitconvert (memop addr:$src))))]>, XOP, VEX_L,
           Sched<[WriteFAddLd, ReadAfterLd]>;
}

let ExeDomain = SSEPackedSingle in {
  defm VFRCZSS : xop2opsld<0x82, "vfrczss", int_x86_xop_vfrcz_ss,
                           ssmem, sse_load_f32>;
  defm VFRCZPS : xop2op128<0x80, "vfrczps", int_x86_xop_vfrcz_ps, loadv4f32>;
  defm VFRCZPS : xop2op256<0x80, "vfrczps", int_x86_xop_vfrcz_ps_256, loadv8f32>;
}

let ExeDomain = SSEPackedDouble in {
  defm VFRCZSD : xop2opsld<0x83, "vfrczsd", int_x86_xop_vfrcz_sd,
                           sdmem, sse_load_f64>;
  defm VFRCZPD : xop2op128<0x81, "vfrczpd", int_x86_xop_vfrcz_pd, loadv2f64>;
  defm VFRCZPD : xop2op256<0x81, "vfrczpd", int_x86_xop_vfrcz_pd_256, loadv4f64>;
}

multiclass xop3op<bits<8> opc, string OpcodeStr, SDNode OpNode,
                  ValueType vt128> {
  def rr : IXOP<opc, MRMSrcReg4VOp3, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           [(set VR128:$dst,
              (vt128 (OpNode (vt128 VR128:$src1), (vt128 VR128:$src2))))]>,
           XOP, Sched<[WriteVarVecShift]>;
  def rm : IXOP<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, i128mem:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           [(set VR128:$dst,
              (vt128 (OpNode (vt128 VR128:$src1),
                             (vt128 (bitconvert (loadv2i64 addr:$src2))))))]>,
           XOP_4V, VEX_W, Sched<[WriteVarVecShiftLd, ReadAfterLd]>;
  def mr : IXOP<opc, MRMSrcMem4VOp3, (outs VR128:$dst),
           (ins i128mem:$src1, VR128:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           [(set VR128:$dst,
              (vt128 (OpNode (vt128 (bitconvert (loadv2i64 addr:$src1))),
                             (vt128 VR128:$src2))))]>,
             XOP, Sched<[WriteVarVecShiftLd, ReadAfterLd]>;
  // For disassembler
  let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0 in
  def rr_REV : IXOP<opc, MRMSrcReg, (outs VR128:$dst),
               (ins VR128:$src1, VR128:$src2),
               !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
               []>,
               XOP_4V, VEX_W, Sched<[WriteVarVecShift]>, FoldGenData<NAME#rr>;
}

let ExeDomain = SSEPackedInt in {
  defm VPROTB : xop3op<0x90, "vprotb", rotl, v16i8>;
  defm VPROTD : xop3op<0x92, "vprotd", rotl, v4i32>;
  defm VPROTQ : xop3op<0x93, "vprotq", rotl, v2i64>;
  defm VPROTW : xop3op<0x91, "vprotw", rotl, v8i16>;
  defm VPSHAB : xop3op<0x98, "vpshab", X86vpsha, v16i8>;
  defm VPSHAD : xop3op<0x9A, "vpshad", X86vpsha, v4i32>;
  defm VPSHAQ : xop3op<0x9B, "vpshaq", X86vpsha, v2i64>;
  defm VPSHAW : xop3op<0x99, "vpshaw", X86vpsha, v8i16>;
  defm VPSHLB : xop3op<0x94, "vpshlb", X86vpshl, v16i8>;
  defm VPSHLD : xop3op<0x96, "vpshld", X86vpshl, v4i32>;
  defm VPSHLQ : xop3op<0x97, "vpshlq", X86vpshl, v2i64>;
  defm VPSHLW : xop3op<0x95, "vpshlw", X86vpshl, v8i16>;
}

multiclass xop3opimm<bits<8> opc, string OpcodeStr, SDNode OpNode,
                     ValueType vt128> {
  def ri : IXOPi8<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, u8imm:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           [(set VR128:$dst,
              (vt128 (OpNode (vt128 VR128:$src1), imm:$src2)))]>,
           XOP, Sched<[WriteVecShift]>;
  def mi : IXOPi8<opc, MRMSrcMem, (outs VR128:$dst),
           (ins i128mem:$src1, u8imm:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           [(set VR128:$dst,
              (vt128 (OpNode (vt128 (bitconvert (loadv2i64 addr:$src1))), imm:$src2)))]>,
           XOP, Sched<[WriteVecShiftLd, ReadAfterLd]>;
}

let ExeDomain = SSEPackedInt in {
  defm VPROTB : xop3opimm<0xC0, "vprotb", X86vrotli, v16i8>;
  defm VPROTD : xop3opimm<0xC2, "vprotd", X86vrotli, v4i32>;
  defm VPROTQ : xop3opimm<0xC3, "vprotq", X86vrotli, v2i64>;
  defm VPROTW : xop3opimm<0xC1, "vprotw", X86vrotli, v8i16>;
}

// Instruction where second source can be memory, but third must be register
multiclass xop4opm2<bits<8> opc, string OpcodeStr, Intrinsic Int> {
  let isCommutable = 1 in
  def rr : IXOPi8Reg<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst,
              (Int VR128:$src1, VR128:$src2, VR128:$src3))]>, XOP_4V,
           Sched<[WriteVecIMul]>;
  def rm : IXOPi8Reg<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, i128mem:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst,
              (Int VR128:$src1, (bitconvert (loadv2i64 addr:$src2)),
              VR128:$src3))]>, XOP_4V, Sched<[WriteVecIMulLd, ReadAfterLd]>;
}

let ExeDomain = SSEPackedInt in {
  defm VPMADCSWD  : xop4opm2<0xB6, "vpmadcswd", int_x86_xop_vpmadcswd>;
  defm VPMADCSSWD : xop4opm2<0xA6, "vpmadcsswd", int_x86_xop_vpmadcsswd>;
  defm VPMACSWW   : xop4opm2<0x95, "vpmacsww", int_x86_xop_vpmacsww>;
  defm VPMACSWD   : xop4opm2<0x96, "vpmacswd", int_x86_xop_vpmacswd>;
  defm VPMACSSWW  : xop4opm2<0x85, "vpmacssww", int_x86_xop_vpmacssww>;
  defm VPMACSSWD  : xop4opm2<0x86, "vpmacsswd", int_x86_xop_vpmacsswd>;
  defm VPMACSSDQL : xop4opm2<0x87, "vpmacssdql", int_x86_xop_vpmacssdql>;
  defm VPMACSSDQH : xop4opm2<0x8F, "vpmacssdqh", int_x86_xop_vpmacssdqh>;
  defm VPMACSSDD  : xop4opm2<0x8E, "vpmacssdd", int_x86_xop_vpmacssdd>;
  defm VPMACSDQL  : xop4opm2<0x97, "vpmacsdql", int_x86_xop_vpmacsdql>;
  defm VPMACSDQH  : xop4opm2<0x9F, "vpmacsdqh", int_x86_xop_vpmacsdqh>;
  defm VPMACSDD   : xop4opm2<0x9E, "vpmacsdd", int_x86_xop_vpmacsdd>;
}

// IFMA patterns - for cases where we can safely ignore the overflow bits from
// the multiply or easily match with existing intrinsics.
let Predicates = [HasXOP] in {
  def : Pat<(v8i16 (add (mul (v8i16 VR128:$src1), (v8i16 VR128:$src2)),
                        (v8i16 VR128:$src3))),
            (VPMACSWWrr VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(v4i32 (add (mul (v4i32 VR128:$src1), (v4i32 VR128:$src2)),
                        (v4i32 VR128:$src3))),
            (VPMACSDDrr VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(v2i64 (add (X86pmuldq (bc_v2i64 (X86PShufd (v4i32 VR128:$src1), (i8 -11))),
                                   (bc_v2i64 (X86PShufd (v4i32 VR128:$src2), (i8 -11)))),
                        (v2i64 VR128:$src3))),
            (VPMACSDQHrr VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(v2i64 (add (X86pmuldq (v2i64 VR128:$src1), (v2i64 VR128:$src2)),
                        (v2i64 VR128:$src3))),
            (VPMACSDQLrr VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(v4i32 (add (X86vpmaddwd (v8i16 VR128:$src1), (v8i16 VR128:$src2)),
                        (v4i32 VR128:$src3))),
            (VPMADCSWDrr VR128:$src1, VR128:$src2, VR128:$src3)>;
}

// Transforms to swizzle an immediate to help matching memory operand in first
// operand.
def CommuteVPCOMCC : SDNodeXForm<imm, [{
  uint8_t Imm = N->getZExtValue() & 0x7;
  Imm = X86::getSwappedVPCOMImm(Imm);
  return getI8Imm(Imm, SDLoc(N));
}]>;

// Instruction where second source can be memory, third must be imm8
multiclass xopvpcom<bits<8> opc, string Suffix, SDNode OpNode, ValueType vt128> {
  let ExeDomain = SSEPackedInt in { // SSE integer instructions
    let isCommutable = 1 in
    def ri : IXOPi8<opc, MRMSrcReg, (outs VR128:$dst),
             (ins VR128:$src1, VR128:$src2, XOPCC:$cc),
             !strconcat("vpcom${cc}", Suffix,
             "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set VR128:$dst,
                (vt128 (OpNode (vt128 VR128:$src1), (vt128 VR128:$src2),
                               imm:$cc)))]>,
             XOP_4V, Sched<[WriteVecALU]>;
    def mi : IXOPi8<opc, MRMSrcMem, (outs VR128:$dst),
             (ins VR128:$src1, i128mem:$src2, XOPCC:$cc),
             !strconcat("vpcom${cc}", Suffix,
             "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set VR128:$dst,
                (vt128 (OpNode (vt128 VR128:$src1),
                               (vt128 (bitconvert (loadv2i64 addr:$src2))),
                                imm:$cc)))]>,
             XOP_4V, Sched<[WriteVecALULd, ReadAfterLd]>;
    let isAsmParserOnly = 1, hasSideEffects = 0 in {
      def ri_alt : IXOPi8<opc, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src1, VR128:$src2, u8imm:$src3),
                   !strconcat("vpcom", Suffix,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                   []>, XOP_4V, Sched<[WriteVecALU]>;
      let mayLoad = 1 in
      def mi_alt : IXOPi8<opc, MRMSrcMem, (outs VR128:$dst),
                   (ins VR128:$src1, i128mem:$src2, u8imm:$src3),
                   !strconcat("vpcom", Suffix,
                   "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                   []>, XOP_4V, Sched<[WriteVecALULd, ReadAfterLd]>;
    }
  }

  def : Pat<(OpNode (bitconvert (loadv2i64 addr:$src2)),
                    (vt128 VR128:$src1), imm:$cc),
            (!cast<Instruction>(NAME#"mi") VR128:$src1, addr:$src2,
                                           (CommuteVPCOMCC imm:$cc))>;
}

defm VPCOMB  : xopvpcom<0xCC, "b", X86vpcom, v16i8>;
defm VPCOMW  : xopvpcom<0xCD, "w", X86vpcom, v8i16>;
defm VPCOMD  : xopvpcom<0xCE, "d", X86vpcom, v4i32>;
defm VPCOMQ  : xopvpcom<0xCF, "q", X86vpcom, v2i64>;
defm VPCOMUB : xopvpcom<0xEC, "ub", X86vpcomu, v16i8>;
defm VPCOMUW : xopvpcom<0xED, "uw", X86vpcomu, v8i16>;
defm VPCOMUD : xopvpcom<0xEE, "ud", X86vpcomu, v4i32>;
defm VPCOMUQ : xopvpcom<0xEF, "uq", X86vpcomu, v2i64>;

multiclass xop4op<bits<8> opc, string OpcodeStr, SDNode OpNode,
                  ValueType vt128> {
  def rrr : IXOPi8Reg<opc, MRMSrcReg, (outs VR128:$dst),
            (ins VR128:$src1, VR128:$src2, VR128:$src3),
            !strconcat(OpcodeStr,
            "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
            [(set VR128:$dst,
              (vt128 (OpNode (vt128 VR128:$src1), (vt128 VR128:$src2),
                             (vt128 VR128:$src3))))]>,
            XOP_4V, Sched<[WriteShuffle]>;
  def rrm : IXOPi8Reg<opc, MRMSrcMemOp4, (outs VR128:$dst),
            (ins VR128:$src1, VR128:$src2, i128mem:$src3),
            !strconcat(OpcodeStr,
            "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
            [(set VR128:$dst,
              (vt128 (OpNode (vt128 VR128:$src1), (vt128 VR128:$src2),
                             (vt128 (bitconvert (loadv2i64 addr:$src3))))))]>,
            XOP_4V, VEX_W, Sched<[WriteShuffleLd, ReadAfterLd, ReadAfterLd]>;
  def rmr : IXOPi8Reg<opc, MRMSrcMem, (outs VR128:$dst),
            (ins VR128:$src1, i128mem:$src2, VR128:$src3),
            !strconcat(OpcodeStr,
            "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
            [(set VR128:$dst,
              (v16i8 (OpNode (vt128 VR128:$src1), (vt128 (bitconvert (loadv2i64 addr:$src2))),
                             (vt128 VR128:$src3))))]>,
            XOP_4V, Sched<[WriteShuffleLd, ReadAfterLd,
                           // 128mem:$src2
                           ReadDefault, ReadDefault, ReadDefault, ReadDefault,
                           ReadDefault,
                           // VR128:$src3
                           ReadAfterLd]>;
  // For disassembler
  let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0 in
  def rrr_REV : IXOPi8Reg<opc, MRMSrcRegOp4, (outs VR128:$dst),
                (ins VR128:$src1, VR128:$src2, VR128:$src3),
                !strconcat(OpcodeStr,
                "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
                []>, XOP_4V, VEX_W, Sched<[WriteShuffle]>, FoldGenData<NAME#rrr>;
}

let ExeDomain = SSEPackedInt in {
  defm VPPERM : xop4op<0xA3, "vpperm", X86vpperm, v16i8>;
}

// Instruction where either second or third source can be memory
multiclass xop4op_int<bits<8> opc, string OpcodeStr, RegisterClass RC,
                       X86MemOperand x86memop, ValueType VT> {
  def rrr : IXOPi8Reg<opc, MRMSrcReg, (outs RC:$dst),
            (ins RC:$src1, RC:$src2, RC:$src3),
            !strconcat(OpcodeStr,
            "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
            [(set RC:$dst, (VT (or (and RC:$src3, RC:$src1),
                                   (X86andnp RC:$src3, RC:$src2))))]>, XOP_4V,
            Sched<[WriteShuffle]>;
  def rrm : IXOPi8Reg<opc, MRMSrcMemOp4, (outs RC:$dst),
            (ins RC:$src1, RC:$src2, x86memop:$src3),
            !strconcat(OpcodeStr,
            "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
            [(set RC:$dst, (VT (or (and (load addr:$src3), RC:$src1),
                                   (X86andnp (load addr:$src3), RC:$src2))))]>,
            XOP_4V, VEX_W, Sched<[WriteShuffleLd, ReadAfterLd, ReadAfterLd]>;
  def rmr : IXOPi8Reg<opc, MRMSrcMem, (outs RC:$dst),
            (ins RC:$src1, x86memop:$src2, RC:$src3),
            !strconcat(OpcodeStr,
            "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
            [(set RC:$dst, (VT (or (and RC:$src3, RC:$src1),
                                   (X86andnp RC:$src3, (load addr:$src2)))))]>,
            XOP_4V, Sched<[WriteShuffleLd, ReadAfterLd,
                           // x86memop:$src2
                           ReadDefault, ReadDefault, ReadDefault, ReadDefault,
                           ReadDefault,
                           // RC::$src3
                           ReadAfterLd]>;
  // For disassembler
  let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0 in
  def rrr_REV : IXOPi8Reg<opc, MRMSrcRegOp4, (outs RC:$dst),
            (ins RC:$src1, RC:$src2, RC:$src3),
            !strconcat(OpcodeStr,
            "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
            []>, XOP_4V, VEX_W, Sched<[WriteShuffle]>, FoldGenData<NAME#rrr>;
}

let ExeDomain = SSEPackedInt in {
  defm VPCMOV : xop4op_int<0xA2, "vpcmov", VR128, i128mem, v2i64>;
  defm VPCMOVY : xop4op_int<0xA2, "vpcmov", VR256, i256mem, v4i64>, VEX_L;
}

multiclass xop_vpermil2<bits<8> Opc, string OpcodeStr, RegisterClass RC,
                        X86MemOperand intmemop, X86MemOperand fpmemop,
                        ValueType VT, PatFrag FPLdFrag,
                        PatFrag IntLdFrag> {
  def rr : IXOP5<Opc, MRMSrcReg, (outs RC:$dst),
        (ins RC:$src1, RC:$src2, RC:$src3, u8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        [(set RC:$dst,
           (VT (X86vpermil2 RC:$src1, RC:$src2, RC:$src3, (i8 imm:$src4))))]>,
        Sched<[WriteFShuffle]>;
  def rm : IXOP5<Opc, MRMSrcMemOp4, (outs RC:$dst),
        (ins RC:$src1, RC:$src2, intmemop:$src3, u8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        [(set RC:$dst,
          (VT (X86vpermil2 RC:$src1, RC:$src2,
                           (bitconvert (IntLdFrag addr:$src3)),
                           (i8 imm:$src4))))]>, VEX_W,
        Sched<[WriteFShuffleLd, ReadAfterLd, ReadAfterLd]>;
  def mr : IXOP5<Opc, MRMSrcMem, (outs RC:$dst),
        (ins RC:$src1, fpmemop:$src2, RC:$src3, u8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        [(set RC:$dst,
          (VT (X86vpermil2 RC:$src1, (FPLdFrag addr:$src2),
                           RC:$src3, (i8 imm:$src4))))]>,
        Sched<[WriteFShuffleLd, ReadAfterLd,
               // fpmemop:$src2
               ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault,
               // RC:$src3
               ReadAfterLd]>;
  // For disassembler
  let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0 in
  def rr_REV : IXOP5<Opc, MRMSrcRegOp4, (outs RC:$dst),
        (ins RC:$src1, RC:$src2, RC:$src3, u8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        []>, VEX_W, Sched<[WriteFShuffle]>, FoldGenData<NAME#rr>;
}

let ExeDomain = SSEPackedDouble in {
  defm VPERMIL2PD : xop_vpermil2<0x49, "vpermil2pd", VR128, i128mem, f128mem,
                                 v2f64, loadv2f64, loadv2i64>;
  defm VPERMIL2PDY : xop_vpermil2<0x49, "vpermil2pd", VR256, i256mem, f256mem,
                                  v4f64, loadv4f64, loadv4i64>, VEX_L;
}

let ExeDomain = SSEPackedSingle in {
  defm VPERMIL2PS : xop_vpermil2<0x48, "vpermil2ps", VR128, i128mem, f128mem,
                                 v4f32, loadv4f32, loadv2i64>;
  defm VPERMIL2PSY : xop_vpermil2<0x48, "vpermil2ps", VR256, i256mem, f256mem,
                                  v8f32, loadv8f32, loadv4i64>, VEX_L;
}

