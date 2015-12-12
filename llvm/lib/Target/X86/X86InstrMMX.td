//===-- X86InstrMMX.td - Describe the MMX Instruction Set --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 MMX instruction set, defining the instructions,
// and properties of the instructions which are needed for code generation,
// machine code emission, and analysis.
//
// All instructions that use MMX should be in this file, even if they also use
// SSE.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MMX Multiclasses
//===----------------------------------------------------------------------===//

let Sched = WriteVecALU in {
def MMX_INTALU_ITINS : OpndItins<
  IIC_MMX_ALU_RR, IIC_MMX_ALU_RM
>;

def MMX_INTALUQ_ITINS : OpndItins<
  IIC_MMX_ALUQ_RR, IIC_MMX_ALUQ_RM
>;

def MMX_PHADDSUBW : OpndItins<
  IIC_MMX_PHADDSUBW_RR, IIC_MMX_PHADDSUBW_RM
>;

def MMX_PHADDSUBD : OpndItins<
  IIC_MMX_PHADDSUBD_RR, IIC_MMX_PHADDSUBD_RM
>;
}

let Sched = WriteVecLogic in
def MMX_INTALU_ITINS_VECLOGICSCHED : OpndItins<
  IIC_MMX_ALU_RR, IIC_MMX_ALU_RM
>;

let Sched = WriteVecIMul in
def MMX_PMUL_ITINS : OpndItins<
  IIC_MMX_PMUL, IIC_MMX_PMUL
>;

let Sched = WriteVecIMul in {
def MMX_PSADBW_ITINS : OpndItins<
  IIC_MMX_PSADBW, IIC_MMX_PSADBW
>;

def MMX_MISC_FUNC_ITINS : OpndItins<
  IIC_MMX_MISC_FUNC_MEM, IIC_MMX_MISC_FUNC_REG
>;
}

def MMX_SHIFT_ITINS : ShiftOpndItins<
  IIC_MMX_SHIFT_RR, IIC_MMX_SHIFT_RM, IIC_MMX_SHIFT_RI
>;

let Sched = WriteShuffle in {
def MMX_UNPCK_H_ITINS : OpndItins<
  IIC_MMX_UNPCK_H_RR, IIC_MMX_UNPCK_H_RM
>;

def MMX_UNPCK_L_ITINS : OpndItins<
  IIC_MMX_UNPCK_L, IIC_MMX_UNPCK_L
>;

def MMX_PCK_ITINS : OpndItins<
  IIC_MMX_PCK_RR, IIC_MMX_PCK_RM
>;

def MMX_PSHUF_ITINS : OpndItins<
  IIC_MMX_PSHUF, IIC_MMX_PSHUF
>;
} // Sched

let Sched = WriteCvtF2I in {
def MMX_CVT_PD_ITINS : OpndItins<
  IIC_MMX_CVT_PD_RR, IIC_MMX_CVT_PD_RM
>;

def MMX_CVT_PS_ITINS : OpndItins<
  IIC_MMX_CVT_PS_RR, IIC_MMX_CVT_PS_RM
>;
}

let Constraints = "$src1 = $dst" in {
  // MMXI_binop_rm_int - Simple MMX binary operator based on intrinsic.
  // When this is cleaned up, remove the FIXME from X86RecognizableInstr.cpp.
  multiclass MMXI_binop_rm_int<bits<8> opc, string OpcodeStr, Intrinsic IntId,
                               OpndItins itins, bit Commutable = 0> {
    def irr : MMXI<opc, MRMSrcReg, (outs VR64:$dst),
                 (ins VR64:$src1, VR64:$src2),
                 !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                 [(set VR64:$dst, (IntId VR64:$src1, VR64:$src2))], itins.rr>,
              Sched<[itins.Sched]> {
      let isCommutable = Commutable;
    }
    def irm : MMXI<opc, MRMSrcMem, (outs VR64:$dst),
                 (ins VR64:$src1, i64mem:$src2),
                 !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                 [(set VR64:$dst, (IntId VR64:$src1,
                                   (bitconvert (load_mmx addr:$src2))))],
                 itins.rm>, Sched<[itins.Sched.Folded, ReadAfterLd]>;
  }

  multiclass MMXI_binop_rmi_int<bits<8> opc, bits<8> opc2, Format ImmForm,
                                string OpcodeStr, Intrinsic IntId,
                                Intrinsic IntId2, ShiftOpndItins itins> {
    def rr : MMXI<opc, MRMSrcReg, (outs VR64:$dst),
                                  (ins VR64:$src1, VR64:$src2),
                  !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                  [(set VR64:$dst, (IntId VR64:$src1, VR64:$src2))], itins.rr>,
             Sched<[WriteVecShift]>;
    def rm : MMXI<opc, MRMSrcMem, (outs VR64:$dst),
                                  (ins VR64:$src1, i64mem:$src2),
                  !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
                  [(set VR64:$dst, (IntId VR64:$src1,
                                    (bitconvert (load_mmx addr:$src2))))],
                  itins.rm>, Sched<[WriteVecShiftLd, ReadAfterLd]>;
    def ri : MMXIi8<opc2, ImmForm, (outs VR64:$dst),
                                   (ins VR64:$src1, i32u8imm:$src2),
                    !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           [(set VR64:$dst, (IntId2 VR64:$src1, imm:$src2))], itins.ri>,
           Sched<[WriteVecShift]>;
  }
}

/// Unary MMX instructions requiring SSSE3.
multiclass SS3I_unop_rm_int_mm<bits<8> opc, string OpcodeStr,
                               Intrinsic IntId64, OpndItins itins> {
  def rr64 : MMXSS38I<opc, MRMSrcReg, (outs VR64:$dst), (ins VR64:$src),
                   !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                   [(set VR64:$dst, (IntId64 VR64:$src))], itins.rr>,
             Sched<[itins.Sched]>;

  def rm64 : MMXSS38I<opc, MRMSrcMem, (outs VR64:$dst), (ins i64mem:$src),
                   !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
                   [(set VR64:$dst,
                     (IntId64 (bitconvert (memopmmx addr:$src))))],
                   itins.rm>, Sched<[itins.Sched.Folded]>;
}

/// Binary MMX instructions requiring SSSE3.
let ImmT = NoImm, Constraints = "$src1 = $dst" in {
multiclass SS3I_binop_rm_int_mm<bits<8> opc, string OpcodeStr,
                             Intrinsic IntId64, OpndItins itins> {
  let isCommutable = 0 in
  def rr64 : MMXSS38I<opc, MRMSrcReg, (outs VR64:$dst),
       (ins VR64:$src1, VR64:$src2),
        !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
       [(set VR64:$dst, (IntId64 VR64:$src1, VR64:$src2))], itins.rr>,
      Sched<[itins.Sched]>;
  def rm64 : MMXSS38I<opc, MRMSrcMem, (outs VR64:$dst),
       (ins VR64:$src1, i64mem:$src2),
        !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
       [(set VR64:$dst,
         (IntId64 VR64:$src1,
          (bitconvert (memopmmx addr:$src2))))], itins.rm>,
      Sched<[itins.Sched.Folded, ReadAfterLd]>;
}
}

/// PALIGN MMX instructions (require SSSE3).
multiclass ssse3_palign_mm<string asm, Intrinsic IntId> {
  def R64irr  : MMXSS3AI<0x0F, MRMSrcReg, (outs VR64:$dst),
      (ins VR64:$src1, VR64:$src2, u8imm:$src3),
      !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
      [(set VR64:$dst, (IntId VR64:$src1, VR64:$src2, (i8 imm:$src3)))]>,
      Sched<[WriteShuffle]>;
  def R64irm  : MMXSS3AI<0x0F, MRMSrcMem, (outs VR64:$dst),
      (ins VR64:$src1, i64mem:$src2, u8imm:$src3),
      !strconcat(asm, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
      [(set VR64:$dst, (IntId VR64:$src1,
                       (bitconvert (load_mmx addr:$src2)), (i8 imm:$src3)))]>,
      Sched<[WriteShuffleLd, ReadAfterLd]>;
}

multiclass sse12_cvt_pint<bits<8> opc, RegisterClass SrcRC, RegisterClass DstRC,
                         Intrinsic Int, X86MemOperand x86memop, PatFrag ld_frag,
                         string asm, OpndItins itins, Domain d> {
  def irr : MMXPI<opc, MRMSrcReg, (outs DstRC:$dst), (ins SrcRC:$src), asm,
                  [(set DstRC:$dst, (Int SrcRC:$src))], itins.rr, d>,
            Sched<[itins.Sched]>;
  def irm : MMXPI<opc, MRMSrcMem, (outs DstRC:$dst), (ins x86memop:$src), asm,
                  [(set DstRC:$dst, (Int (ld_frag addr:$src)))], itins.rm, d>,
            Sched<[itins.Sched.Folded]>;
}

multiclass sse12_cvt_pint_3addr<bits<8> opc, RegisterClass SrcRC,
                    RegisterClass DstRC, Intrinsic Int, X86MemOperand x86memop,
                    PatFrag ld_frag, string asm, Domain d> {
  def irr : MMXPI<opc, MRMSrcReg, (outs DstRC:$dst),
                  (ins DstRC:$src1, SrcRC:$src2), asm,
                  [(set DstRC:$dst, (Int DstRC:$src1, SrcRC:$src2))],
                  NoItinerary, d>, Sched<[WriteCvtI2F]>;
  def irm : MMXPI<opc, MRMSrcMem, (outs DstRC:$dst),
                  (ins DstRC:$src1, x86memop:$src2), asm,
                  [(set DstRC:$dst, (Int DstRC:$src1, (ld_frag addr:$src2)))],
                  NoItinerary, d>, Sched<[WriteCvtI2FLd]>;
}

//===----------------------------------------------------------------------===//
// MMX EMMS Instruction
//===----------------------------------------------------------------------===//

def MMX_EMMS  : MMXI<0x77, RawFrm, (outs), (ins), "emms",
                     [(int_x86_mmx_emms)], IIC_MMX_EMMS>;

//===----------------------------------------------------------------------===//
// MMX Scalar Instructions
//===----------------------------------------------------------------------===//

// Data Transfer Instructions
def MMX_MOVD64rr : MMXI<0x6E, MRMSrcReg, (outs VR64:$dst), (ins GR32:$src),
                        "movd\t{$src, $dst|$dst, $src}",
                        [(set VR64:$dst,
                         (x86mmx (scalar_to_vector GR32:$src)))],
                        IIC_MMX_MOV_MM_RM>, Sched<[WriteMove]>;
def MMX_MOVD64rm : MMXI<0x6E, MRMSrcMem, (outs VR64:$dst), (ins i32mem:$src),
                        "movd\t{$src, $dst|$dst, $src}",
                        [(set VR64:$dst,
                        (x86mmx (scalar_to_vector (loadi32 addr:$src))))],
                        IIC_MMX_MOV_MM_RM>, Sched<[WriteLoad]>;

let Predicates = [HasMMX] in {
  let AddedComplexity = 15 in
    def : Pat<(x86mmx (MMX_X86movw2d GR32:$src)),
              (MMX_MOVD64rr GR32:$src)>;
  let AddedComplexity = 20 in
    def : Pat<(x86mmx (MMX_X86movw2d (loadi32 addr:$src))),
              (MMX_MOVD64rm addr:$src)>;
}

let mayStore = 1 in
def MMX_MOVD64mr : MMXI<0x7E, MRMDestMem, (outs), (ins i32mem:$dst, VR64:$src),
                        "movd\t{$src, $dst|$dst, $src}", [], IIC_MMX_MOV_MM_RM>,
                   Sched<[WriteStore]>;

def MMX_MOVD64grr : MMXI<0x7E, MRMDestReg, (outs GR32:$dst), (ins VR64:$src),
                         "movd\t{$src, $dst|$dst, $src}",
                         [(set GR32:$dst,
                          (MMX_X86movd2w (x86mmx VR64:$src)))],
                          IIC_MMX_MOV_REG_MM>, Sched<[WriteMove]>;

let isBitcast = 1 in
def MMX_MOVD64to64rr : MMXRI<0x6E, MRMSrcReg, (outs VR64:$dst), (ins GR64:$src),
                             "movd\t{$src, $dst|$dst, $src}",
                             [(set VR64:$dst, (bitconvert GR64:$src))],
                             IIC_MMX_MOV_MM_RM>, Sched<[WriteMove]>;

let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0, mayLoad = 1 in
def MMX_MOVD64to64rm : MMXRI<0x6E, MRMSrcMem, (outs VR64:$dst),
                             (ins i64mem:$src), "movd\t{$src, $dst|$dst, $src}",
                             [], IIC_MMX_MOVQ_RM>, Sched<[WriteLoad]>;

// These are 64 bit moves, but since the OS X assembler doesn't
// recognize a register-register movq, we write them as
// movd.
let SchedRW = [WriteMove], isBitcast = 1 in {
def MMX_MOVD64from64rr : MMXRI<0x7E, MRMDestReg,
                               (outs GR64:$dst), (ins VR64:$src),
                               "movd\t{$src, $dst|$dst, $src}",
                             [(set GR64:$dst,
                              (bitconvert VR64:$src))], IIC_MMX_MOV_REG_MM>;
let hasSideEffects = 0 in
def MMX_MOVQ64rr : MMXI<0x6F, MRMSrcReg, (outs VR64:$dst), (ins VR64:$src),
                        "movq\t{$src, $dst|$dst, $src}", [],
                        IIC_MMX_MOVQ_RR>;
let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0 in {
def MMX_MOVQ64rr_REV : MMXI<0x7F, MRMDestReg, (outs VR64:$dst), (ins VR64:$src),
                        "movq\t{$src, $dst|$dst, $src}", [],
                        IIC_MMX_MOVQ_RR>;
}
} // SchedRW

let isCodeGenOnly = 1, ForceDisassemble = 1, hasSideEffects = 0, mayStore = 1 in
def MMX_MOVD64from64rm : MMXRI<0x7E, MRMDestMem,
                               (outs i64mem:$dst), (ins VR64:$src),
                               "movd\t{$src, $dst|$dst, $src}",
                               [], IIC_MMX_MOV_REG_MM>, Sched<[WriteStore]>;

let SchedRW = [WriteLoad] in {
let canFoldAsLoad = 1 in
def MMX_MOVQ64rm : MMXI<0x6F, MRMSrcMem, (outs VR64:$dst), (ins i64mem:$src),
                        "movq\t{$src, $dst|$dst, $src}",
                        [(set VR64:$dst, (load_mmx addr:$src))],
                        IIC_MMX_MOVQ_RM>;
} // SchedRW
let SchedRW = [WriteStore] in
def MMX_MOVQ64mr : MMXI<0x7F, MRMDestMem, (outs), (ins i64mem:$dst, VR64:$src),
                        "movq\t{$src, $dst|$dst, $src}",
                        [(store (x86mmx VR64:$src), addr:$dst)],
                        IIC_MMX_MOVQ_RM>;

let SchedRW = [WriteMove] in {
def MMX_MOVDQ2Qrr : MMXSDIi8<0xD6, MRMSrcReg, (outs VR64:$dst),
                             (ins VR128:$src), "movdq2q\t{$src, $dst|$dst, $src}",
                             [(set VR64:$dst,
                               (x86mmx (bitconvert
                               (i64 (extractelt (v2i64 VR128:$src),
                                     (iPTR 0))))))],
                             IIC_MMX_MOVQ_RR>;

def MMX_MOVQ2DQrr : MMXS2SIi8<0xD6, MRMSrcReg, (outs VR128:$dst),
                              (ins VR64:$src), "movq2dq\t{$src, $dst|$dst, $src}",
                              [(set VR128:$dst,
                                (v2i64
                                  (scalar_to_vector
                                    (i64 (bitconvert (x86mmx VR64:$src))))))],
                              IIC_MMX_MOVQ_RR>;

let isCodeGenOnly = 1, hasSideEffects = 1 in {
def MMX_MOVQ2FR64rr: MMXS2SIi8<0xD6, MRMSrcReg, (outs FR64:$dst),
                               (ins VR64:$src), "movq2dq\t{$src, $dst|$dst, $src}",
                               [], IIC_MMX_MOVQ_RR>;

def MMX_MOVFR642Qrr: MMXSDIi8<0xD6, MRMSrcReg, (outs VR64:$dst),
                              (ins FR64:$src), "movdq2q\t{$src, $dst|$dst, $src}",
                              [], IIC_MMX_MOVQ_RR>;
}
} // SchedRW

let Predicates = [HasSSE1] in
def MMX_MOVNTQmr  : MMXI<0xE7, MRMDestMem, (outs), (ins i64mem:$dst, VR64:$src),
                         "movntq\t{$src, $dst|$dst, $src}",
                         [(int_x86_mmx_movnt_dq addr:$dst, VR64:$src)],
                         IIC_MMX_MOVQ_RM>, Sched<[WriteStore]>;

let Predicates = [HasMMX] in {
  let AddedComplexity = 15 in
  // movd to MMX register zero-extends
  def : Pat<(x86mmx (X86vzmovl (x86mmx (scalar_to_vector GR32:$src)))),
            (MMX_MOVD64rr GR32:$src)>;
  let AddedComplexity = 20 in
  def : Pat<(x86mmx (X86vzmovl (x86mmx (scalar_to_vector (loadi32 addr:$src))))),
            (MMX_MOVD64rm addr:$src)>;
}

// Arithmetic Instructions
defm MMX_PABSB : SS3I_unop_rm_int_mm<0x1C, "pabsb", int_x86_ssse3_pabs_b,
                                     MMX_INTALU_ITINS>;
defm MMX_PABSW : SS3I_unop_rm_int_mm<0x1D, "pabsw", int_x86_ssse3_pabs_w,
                                     MMX_INTALU_ITINS>;
defm MMX_PABSD : SS3I_unop_rm_int_mm<0x1E, "pabsd", int_x86_ssse3_pabs_d,
                                     MMX_INTALU_ITINS>;
// -- Addition
defm MMX_PADDB : MMXI_binop_rm_int<0xFC, "paddb", int_x86_mmx_padd_b,
                                   MMX_INTALU_ITINS, 1>;
defm MMX_PADDW : MMXI_binop_rm_int<0xFD, "paddw", int_x86_mmx_padd_w,
                                   MMX_INTALU_ITINS, 1>;
defm MMX_PADDD : MMXI_binop_rm_int<0xFE, "paddd", int_x86_mmx_padd_d,
                                   MMX_INTALU_ITINS, 1>;
let Predicates = [HasSSE2] in
defm MMX_PADDQ : MMXI_binop_rm_int<0xD4, "paddq", int_x86_mmx_padd_q,
                                   MMX_INTALUQ_ITINS, 1>;
defm MMX_PADDSB  : MMXI_binop_rm_int<0xEC, "paddsb" , int_x86_mmx_padds_b,
                                   MMX_INTALU_ITINS, 1>;
defm MMX_PADDSW  : MMXI_binop_rm_int<0xED, "paddsw" , int_x86_mmx_padds_w,
                                   MMX_INTALU_ITINS, 1>;

defm MMX_PADDUSB : MMXI_binop_rm_int<0xDC, "paddusb", int_x86_mmx_paddus_b,
                                   MMX_INTALU_ITINS, 1>;
defm MMX_PADDUSW : MMXI_binop_rm_int<0xDD, "paddusw", int_x86_mmx_paddus_w,
                                   MMX_INTALU_ITINS, 1>;

defm MMX_PHADDW  : SS3I_binop_rm_int_mm<0x01, "phaddw", int_x86_ssse3_phadd_w,
                                   MMX_PHADDSUBW>;
defm MMX_PHADD   : SS3I_binop_rm_int_mm<0x02, "phaddd", int_x86_ssse3_phadd_d,
                                   MMX_PHADDSUBD>;
defm MMX_PHADDSW : SS3I_binop_rm_int_mm<0x03, "phaddsw",int_x86_ssse3_phadd_sw,
                                   MMX_PHADDSUBW>;


// -- Subtraction
defm MMX_PSUBB : MMXI_binop_rm_int<0xF8, "psubb", int_x86_mmx_psub_b,
                                   MMX_INTALU_ITINS>;
defm MMX_PSUBW : MMXI_binop_rm_int<0xF9, "psubw", int_x86_mmx_psub_w,
                                   MMX_INTALU_ITINS>;
defm MMX_PSUBD : MMXI_binop_rm_int<0xFA, "psubd", int_x86_mmx_psub_d,
                                   MMX_INTALU_ITINS>;
let Predicates = [HasSSE2] in
defm MMX_PSUBQ : MMXI_binop_rm_int<0xFB, "psubq", int_x86_mmx_psub_q,
                                   MMX_INTALUQ_ITINS>;

defm MMX_PSUBSB  : MMXI_binop_rm_int<0xE8, "psubsb" , int_x86_mmx_psubs_b,
                                   MMX_INTALU_ITINS>;
defm MMX_PSUBSW  : MMXI_binop_rm_int<0xE9, "psubsw" , int_x86_mmx_psubs_w,
                                   MMX_INTALU_ITINS>;

defm MMX_PSUBUSB : MMXI_binop_rm_int<0xD8, "psubusb", int_x86_mmx_psubus_b,
                                   MMX_INTALU_ITINS>;
defm MMX_PSUBUSW : MMXI_binop_rm_int<0xD9, "psubusw", int_x86_mmx_psubus_w,
                                   MMX_INTALU_ITINS>;

defm MMX_PHSUBW  : SS3I_binop_rm_int_mm<0x05, "phsubw", int_x86_ssse3_phsub_w,
                                   MMX_PHADDSUBW>;
defm MMX_PHSUBD  : SS3I_binop_rm_int_mm<0x06, "phsubd", int_x86_ssse3_phsub_d,
                                   MMX_PHADDSUBD>;
defm MMX_PHSUBSW : SS3I_binop_rm_int_mm<0x07, "phsubsw",int_x86_ssse3_phsub_sw,
                                   MMX_PHADDSUBW>;

// -- Multiplication
defm MMX_PMULLW  : MMXI_binop_rm_int<0xD5, "pmullw", int_x86_mmx_pmull_w,
                                     MMX_PMUL_ITINS, 1>;

defm MMX_PMULHW  : MMXI_binop_rm_int<0xE5, "pmulhw",  int_x86_mmx_pmulh_w,
                                     MMX_PMUL_ITINS, 1>;
let Predicates = [HasSSE1] in
defm MMX_PMULHUW : MMXI_binop_rm_int<0xE4, "pmulhuw", int_x86_mmx_pmulhu_w,
                                     MMX_PMUL_ITINS, 1>;
let Predicates = [HasSSE2] in
defm MMX_PMULUDQ : MMXI_binop_rm_int<0xF4, "pmuludq", int_x86_mmx_pmulu_dq,
                                     MMX_PMUL_ITINS, 1>;
let isCommutable = 1 in
defm MMX_PMULHRSW : SS3I_binop_rm_int_mm<0x0B, "pmulhrsw",
                                     int_x86_ssse3_pmul_hr_sw, MMX_PMUL_ITINS>;

// -- Miscellanea
defm MMX_PMADDWD : MMXI_binop_rm_int<0xF5, "pmaddwd", int_x86_mmx_pmadd_wd,
                                     MMX_PMUL_ITINS, 1>;

defm MMX_PMADDUBSW : SS3I_binop_rm_int_mm<0x04, "pmaddubsw",
                                     int_x86_ssse3_pmadd_ub_sw, MMX_PMUL_ITINS>;
let Predicates = [HasSSE1] in {
defm MMX_PAVGB   : MMXI_binop_rm_int<0xE0, "pavgb", int_x86_mmx_pavg_b,
                                     MMX_MISC_FUNC_ITINS, 1>;
defm MMX_PAVGW   : MMXI_binop_rm_int<0xE3, "pavgw", int_x86_mmx_pavg_w,
                                     MMX_MISC_FUNC_ITINS, 1>;

defm MMX_PMINUB  : MMXI_binop_rm_int<0xDA, "pminub", int_x86_mmx_pminu_b,
                                     MMX_MISC_FUNC_ITINS, 1>;
defm MMX_PMINSW  : MMXI_binop_rm_int<0xEA, "pminsw", int_x86_mmx_pmins_w,
                                     MMX_MISC_FUNC_ITINS, 1>;

defm MMX_PMAXUB  : MMXI_binop_rm_int<0xDE, "pmaxub", int_x86_mmx_pmaxu_b,
                                     MMX_MISC_FUNC_ITINS, 1>;
defm MMX_PMAXSW  : MMXI_binop_rm_int<0xEE, "pmaxsw", int_x86_mmx_pmaxs_w,
                                     MMX_MISC_FUNC_ITINS, 1>;

defm MMX_PSADBW  : MMXI_binop_rm_int<0xF6, "psadbw", int_x86_mmx_psad_bw,
                                     MMX_PSADBW_ITINS, 1>;
}

defm MMX_PSIGNB :  SS3I_binop_rm_int_mm<0x08, "psignb", int_x86_ssse3_psign_b,
                                        MMX_MISC_FUNC_ITINS>;
defm MMX_PSIGNW :  SS3I_binop_rm_int_mm<0x09, "psignw", int_x86_ssse3_psign_w,
                                        MMX_MISC_FUNC_ITINS>;
defm MMX_PSIGND :  SS3I_binop_rm_int_mm<0x0A, "psignd", int_x86_ssse3_psign_d,
                                        MMX_MISC_FUNC_ITINS>;
let Constraints = "$src1 = $dst" in
  defm MMX_PALIGN : ssse3_palign_mm<"palignr", int_x86_mmx_palignr_b>;

// Logical Instructions
defm MMX_PAND : MMXI_binop_rm_int<0xDB, "pand", int_x86_mmx_pand,
                                  MMX_INTALU_ITINS_VECLOGICSCHED, 1>;
defm MMX_POR  : MMXI_binop_rm_int<0xEB, "por" , int_x86_mmx_por,
                                  MMX_INTALU_ITINS_VECLOGICSCHED, 1>;
defm MMX_PXOR : MMXI_binop_rm_int<0xEF, "pxor", int_x86_mmx_pxor,
                                  MMX_INTALU_ITINS_VECLOGICSCHED, 1>;
defm MMX_PANDN : MMXI_binop_rm_int<0xDF, "pandn", int_x86_mmx_pandn,
                                  MMX_INTALU_ITINS_VECLOGICSCHED>;

// Shift Instructions
defm MMX_PSRLW : MMXI_binop_rmi_int<0xD1, 0x71, MRM2r, "psrlw",
                                    int_x86_mmx_psrl_w, int_x86_mmx_psrli_w,
                                    MMX_SHIFT_ITINS>;
defm MMX_PSRLD : MMXI_binop_rmi_int<0xD2, 0x72, MRM2r, "psrld",
                                    int_x86_mmx_psrl_d, int_x86_mmx_psrli_d,
                                    MMX_SHIFT_ITINS>;
defm MMX_PSRLQ : MMXI_binop_rmi_int<0xD3, 0x73, MRM2r, "psrlq",
                                    int_x86_mmx_psrl_q, int_x86_mmx_psrli_q,
                                    MMX_SHIFT_ITINS>;

def : Pat<(int_x86_mmx_psrl_w VR64:$src1, (load_mvmmx addr:$src2)),
          (MMX_PSRLWrm VR64:$src1, addr:$src2)>;
def : Pat<(int_x86_mmx_psrl_d VR64:$src1, (load_mvmmx addr:$src2)),
          (MMX_PSRLDrm VR64:$src1, addr:$src2)>;
def : Pat<(int_x86_mmx_psrl_q VR64:$src1, (load_mvmmx addr:$src2)),
          (MMX_PSRLQrm VR64:$src1, addr:$src2)>;

defm MMX_PSLLW : MMXI_binop_rmi_int<0xF1, 0x71, MRM6r, "psllw",
                                    int_x86_mmx_psll_w, int_x86_mmx_pslli_w,
                                    MMX_SHIFT_ITINS>;
defm MMX_PSLLD : MMXI_binop_rmi_int<0xF2, 0x72, MRM6r, "pslld",
                                    int_x86_mmx_psll_d, int_x86_mmx_pslli_d,
                                    MMX_SHIFT_ITINS>;
defm MMX_PSLLQ : MMXI_binop_rmi_int<0xF3, 0x73, MRM6r, "psllq",
                                    int_x86_mmx_psll_q, int_x86_mmx_pslli_q,
                                    MMX_SHIFT_ITINS>;

def : Pat<(int_x86_mmx_psll_w VR64:$src1, (load_mvmmx addr:$src2)),
          (MMX_PSLLWrm VR64:$src1, addr:$src2)>;
def : Pat<(int_x86_mmx_psll_d VR64:$src1, (load_mvmmx addr:$src2)),
          (MMX_PSLLDrm VR64:$src1, addr:$src2)>;
def : Pat<(int_x86_mmx_psll_q VR64:$src1, (load_mvmmx addr:$src2)),
          (MMX_PSLLQrm VR64:$src1, addr:$src2)>;

defm MMX_PSRAW : MMXI_binop_rmi_int<0xE1, 0x71, MRM4r, "psraw",
                                    int_x86_mmx_psra_w, int_x86_mmx_psrai_w,
                                    MMX_SHIFT_ITINS>;
defm MMX_PSRAD : MMXI_binop_rmi_int<0xE2, 0x72, MRM4r, "psrad",
                                    int_x86_mmx_psra_d, int_x86_mmx_psrai_d,
                                    MMX_SHIFT_ITINS>;

def : Pat<(int_x86_mmx_psra_w VR64:$src1, (load_mvmmx addr:$src2)),
          (MMX_PSRAWrm VR64:$src1, addr:$src2)>;
def : Pat<(int_x86_mmx_psra_d VR64:$src1, (load_mvmmx addr:$src2)),
          (MMX_PSRADrm VR64:$src1, addr:$src2)>;

// Comparison Instructions
defm MMX_PCMPEQB : MMXI_binop_rm_int<0x74, "pcmpeqb", int_x86_mmx_pcmpeq_b,
                                     MMX_INTALU_ITINS>;
defm MMX_PCMPEQW : MMXI_binop_rm_int<0x75, "pcmpeqw", int_x86_mmx_pcmpeq_w,
                                     MMX_INTALU_ITINS>;
defm MMX_PCMPEQD : MMXI_binop_rm_int<0x76, "pcmpeqd", int_x86_mmx_pcmpeq_d,
                                     MMX_INTALU_ITINS>;

defm MMX_PCMPGTB : MMXI_binop_rm_int<0x64, "pcmpgtb", int_x86_mmx_pcmpgt_b,
                                     MMX_INTALU_ITINS>;
defm MMX_PCMPGTW : MMXI_binop_rm_int<0x65, "pcmpgtw", int_x86_mmx_pcmpgt_w,
                                     MMX_INTALU_ITINS>;
defm MMX_PCMPGTD : MMXI_binop_rm_int<0x66, "pcmpgtd", int_x86_mmx_pcmpgt_d,
                                     MMX_INTALU_ITINS>;

// -- Unpack Instructions
defm MMX_PUNPCKHBW : MMXI_binop_rm_int<0x68, "punpckhbw",
                                       int_x86_mmx_punpckhbw,
                                       MMX_UNPCK_H_ITINS>;
defm MMX_PUNPCKHWD : MMXI_binop_rm_int<0x69, "punpckhwd",
                                       int_x86_mmx_punpckhwd,
                                       MMX_UNPCK_H_ITINS>;
defm MMX_PUNPCKHDQ : MMXI_binop_rm_int<0x6A, "punpckhdq",
                                       int_x86_mmx_punpckhdq,
                                       MMX_UNPCK_H_ITINS>;
defm MMX_PUNPCKLBW : MMXI_binop_rm_int<0x60, "punpcklbw",
                                       int_x86_mmx_punpcklbw,
                                       MMX_UNPCK_L_ITINS>;
defm MMX_PUNPCKLWD : MMXI_binop_rm_int<0x61, "punpcklwd",
                                       int_x86_mmx_punpcklwd,
                                       MMX_UNPCK_L_ITINS>;
defm MMX_PUNPCKLDQ : MMXI_binop_rm_int<0x62, "punpckldq",
                                       int_x86_mmx_punpckldq,
                                       MMX_UNPCK_L_ITINS>;

// -- Pack Instructions
defm MMX_PACKSSWB : MMXI_binop_rm_int<0x63, "packsswb", int_x86_mmx_packsswb,
                                      MMX_PCK_ITINS>;
defm MMX_PACKSSDW : MMXI_binop_rm_int<0x6B, "packssdw", int_x86_mmx_packssdw,
                                      MMX_PCK_ITINS>;
defm MMX_PACKUSWB : MMXI_binop_rm_int<0x67, "packuswb", int_x86_mmx_packuswb,
                                      MMX_PCK_ITINS>;

// -- Shuffle Instructions
defm MMX_PSHUFB : SS3I_binop_rm_int_mm<0x00, "pshufb", int_x86_ssse3_pshuf_b,
                                       MMX_PSHUF_ITINS>;

def MMX_PSHUFWri : MMXIi8<0x70, MRMSrcReg,
                          (outs VR64:$dst), (ins VR64:$src1, u8imm:$src2),
                          "pshufw\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                          [(set VR64:$dst,
                             (int_x86_sse_pshuf_w VR64:$src1, imm:$src2))],
                          IIC_MMX_PSHUF>, Sched<[WriteShuffle]>;
def MMX_PSHUFWmi : MMXIi8<0x70, MRMSrcMem,
                          (outs VR64:$dst), (ins i64mem:$src1, u8imm:$src2),
                          "pshufw\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                          [(set VR64:$dst,
                             (int_x86_sse_pshuf_w (load_mmx addr:$src1),
                                                   imm:$src2))],
                          IIC_MMX_PSHUF>, Sched<[WriteShuffleLd]>;




// -- Conversion Instructions
defm MMX_CVTPS2PI : sse12_cvt_pint<0x2D, VR128, VR64, int_x86_sse_cvtps2pi,
                      f64mem, load, "cvtps2pi\t{$src, $dst|$dst, $src}",
                      MMX_CVT_PS_ITINS, SSEPackedSingle>, PS;
defm MMX_CVTPD2PI : sse12_cvt_pint<0x2D, VR128, VR64, int_x86_sse_cvtpd2pi,
                      f128mem, memop, "cvtpd2pi\t{$src, $dst|$dst, $src}",
                      MMX_CVT_PD_ITINS, SSEPackedDouble>, PD;
defm MMX_CVTTPS2PI : sse12_cvt_pint<0x2C, VR128, VR64, int_x86_sse_cvttps2pi,
                       f64mem, load, "cvttps2pi\t{$src, $dst|$dst, $src}",
                       MMX_CVT_PS_ITINS, SSEPackedSingle>, PS;
defm MMX_CVTTPD2PI : sse12_cvt_pint<0x2C, VR128, VR64, int_x86_sse_cvttpd2pi,
                       f128mem, memop, "cvttpd2pi\t{$src, $dst|$dst, $src}",
                       MMX_CVT_PD_ITINS, SSEPackedDouble>, PD;
defm MMX_CVTPI2PD : sse12_cvt_pint<0x2A, VR64, VR128, int_x86_sse_cvtpi2pd,
                         i64mem, load, "cvtpi2pd\t{$src, $dst|$dst, $src}",
                         MMX_CVT_PD_ITINS, SSEPackedDouble>, PD;
let Constraints = "$src1 = $dst" in {
  defm MMX_CVTPI2PS : sse12_cvt_pint_3addr<0x2A, VR64, VR128,
                         int_x86_sse_cvtpi2ps,
                         i64mem, load, "cvtpi2ps\t{$src2, $dst|$dst, $src2}",
                          SSEPackedSingle>, PS;
}

// Extract / Insert
let Predicates = [HasSSE1] in
def MMX_PEXTRWirri: MMXIi8<0xC5, MRMSrcReg,
                       (outs GR32orGR64:$dst), (ins VR64:$src1, i32u8imm:$src2),
                       "pextrw\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                       [(set GR32orGR64:$dst, (int_x86_mmx_pextr_w VR64:$src1,
                                               imm:$src2))],
                       IIC_MMX_PEXTR>, Sched<[WriteShuffle]>;
let Constraints = "$src1 = $dst" in {
let Predicates = [HasSSE1] in {
  def MMX_PINSRWirri : MMXIi8<0xC4, MRMSrcReg,
                      (outs VR64:$dst),
                      (ins VR64:$src1, GR32orGR64:$src2, i32u8imm:$src3),
                      "pinsrw\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(set VR64:$dst, (int_x86_mmx_pinsr_w VR64:$src1,
                                        GR32orGR64:$src2, imm:$src3))],
                      IIC_MMX_PINSRW>, Sched<[WriteShuffle]>;

  def MMX_PINSRWirmi : MMXIi8<0xC4, MRMSrcMem,
                     (outs VR64:$dst),
                     (ins VR64:$src1, i16mem:$src2, i32u8imm:$src3),
                     "pinsrw\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set VR64:$dst, (int_x86_mmx_pinsr_w VR64:$src1,
                                         (i32 (anyext (loadi16 addr:$src2))),
                                       imm:$src3))],
                     IIC_MMX_PINSRW>, Sched<[WriteShuffleLd, ReadAfterLd]>;
}
}

// Mask creation
let Predicates = [HasSSE1] in
def MMX_PMOVMSKBrr : MMXI<0xD7, MRMSrcReg, (outs GR32orGR64:$dst),
                          (ins VR64:$src),
                          "pmovmskb\t{$src, $dst|$dst, $src}",
                          [(set GR32orGR64:$dst,
                                (int_x86_mmx_pmovmskb VR64:$src))]>;


// Low word of XMM to MMX.
def MMX_X86movdq2q : SDNode<"X86ISD::MOVDQ2Q", SDTypeProfile<1, 1,
                            [SDTCisVT<0, x86mmx>, SDTCisVT<1, v2i64>]>>;

def : Pat<(x86mmx (MMX_X86movdq2q VR128:$src)),
          (x86mmx (MMX_MOVDQ2Qrr VR128:$src))>;

def : Pat<(x86mmx (MMX_X86movdq2q (loadv2i64 addr:$src))),
          (x86mmx (MMX_MOVQ64rm addr:$src))>;

// Misc.
let SchedRW = [WriteShuffle] in {
let Uses = [EDI], Predicates = [HasSSE1,In32BitMode] in
def MMX_MASKMOVQ : MMXI32<0xF7, MRMSrcReg, (outs), (ins VR64:$src, VR64:$mask),
                          "maskmovq\t{$mask, $src|$src, $mask}",
                          [(int_x86_mmx_maskmovq VR64:$src, VR64:$mask, EDI)],
                          IIC_MMX_MASKMOV>;
let Uses = [RDI], Predicates = [HasSSE1,In64BitMode] in
def MMX_MASKMOVQ64: MMXI64<0xF7, MRMSrcReg, (outs), (ins VR64:$src, VR64:$mask),
                           "maskmovq\t{$mask, $src|$src, $mask}",
                           [(int_x86_mmx_maskmovq VR64:$src, VR64:$mask, RDI)],
                           IIC_MMX_MASKMOV>;
}

// 64-bit bit convert.
let Predicates = [HasSSE2] in {
def : Pat<(f64 (bitconvert (x86mmx VR64:$src))),
          (MMX_MOVQ2FR64rr VR64:$src)>;
def : Pat<(x86mmx (bitconvert (f64 FR64:$src))),
          (MMX_MOVFR642Qrr FR64:$src)>;
}


