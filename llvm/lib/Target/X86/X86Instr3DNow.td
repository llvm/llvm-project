//===-- X86Instr3DNow.td - The 3DNow! Instruction Set ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the 3DNow! instruction set, which extends MMX to support
// floating point and also adds a few more random instructions for good measure.
//
//===----------------------------------------------------------------------===//

let Sched = WriteFAdd in {
def I3DNOW_FALU_ITINS : OpndItins<
  IIC_3DNOW_FALU_RR, IIC_3DNOW_FALU_RM
>;
}

let Sched = WriteCvtF2I in {
def I3DNOW_FCVT_F2I_ITINS : OpndItins<
  IIC_3DNOW_FCVT_F2I_RR, IIC_3DNOW_FCVT_F2I_RM
>;
}

let Sched = WriteCvtI2F in {
def I3DNOW_FCVT_I2F_ITINS : OpndItins<
  IIC_3DNOW_FCVT_I2F_RR, IIC_3DNOW_FCVT_I2F_RM
>;
}

let Sched = WriteVecIMul in {
def I3DNOW_MISC_FUNC_ITINS : OpndItins<
  IIC_3DNOW_MISC_FUNC_REG, IIC_3DNOW_MISC_FUNC_MEM
>;
}

let Sched = WriteShuffle in {
def I3DNOW_PSHUF_ITINS : OpndItins<
  IIC_MMX_PSHUF, IIC_MMX_PSHUF
>;
}

class I3DNow<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pat,
             InstrItinClass itin = NoItinerary>
      : I<o, F, outs, ins, asm, pat, itin>, TB, Requires<[Has3DNow]> {
}

class I3DNow_binop<bits<8> o, Format F, dag ins, string Mnemonic, list<dag> pat,
                   InstrItinClass itin>
      : I3DNow<o, F, (outs VR64:$dst), ins,
          !strconcat(Mnemonic, "\t{$src2, $dst|$dst, $src2}"), pat, itin>,
        Has3DNow0F0FOpcode {
  // FIXME: The disassembler doesn't support Has3DNow0F0FOpcode yet.
  let isAsmParserOnly = 1;
  let Constraints = "$src1 = $dst";
}

class I3DNow_conv<bits<8> o, Format F, dag ins, string Mnemonic, list<dag> pat,
                  InstrItinClass itin>
      : I3DNow<o, F, (outs VR64:$dst), ins,
          !strconcat(Mnemonic, "\t{$src, $dst|$dst, $src}"), pat, itin>,
        Has3DNow0F0FOpcode {
  // FIXME: The disassembler doesn't support Has3DNow0F0FOpcode yet.
  let isAsmParserOnly = 1;
}

multiclass I3DNow_binop_rm_int<bits<8> opc, string Mn, OpndItins itins,
                               bit Commutable = 0, string Ver = ""> {
  let isCommutable = Commutable in
  def rr : I3DNow_binop<opc, MRMSrcReg, (ins VR64:$src1, VR64:$src2), Mn,
    [(set VR64:$dst, (!cast<Intrinsic>(
      !strconcat("int_x86_3dnow", Ver, "_", Mn)) VR64:$src1, VR64:$src2))],
      itins.rr>, Sched<[itins.Sched]>;
  def rm : I3DNow_binop<opc, MRMSrcMem, (ins VR64:$src1, i64mem:$src2), Mn,
    [(set VR64:$dst, (!cast<Intrinsic>(
      !strconcat("int_x86_3dnow", Ver, "_", Mn)) VR64:$src1,
        (bitconvert (load_mmx addr:$src2))))], itins.rm>,
        Sched<[itins.Sched.Folded, ReadAfterLd]>;
}

multiclass I3DNow_conv_rm_int<bits<8> opc, string Mn, OpndItins itins,
                              string Ver = ""> {
  def rr : I3DNow_conv<opc, MRMSrcReg, (ins VR64:$src), Mn,
    [(set VR64:$dst, (!cast<Intrinsic>(
      !strconcat("int_x86_3dnow", Ver, "_", Mn)) VR64:$src))], itins.rr>,
      Sched<[itins.Sched]>;
  def rm : I3DNow_conv<opc, MRMSrcMem, (ins i64mem:$src), Mn,
    [(set VR64:$dst, (!cast<Intrinsic>(
      !strconcat("int_x86_3dnow", Ver, "_", Mn))
        (bitconvert (load_mmx addr:$src))))], itins.rm>,
        Sched<[itins.Sched.Folded, ReadAfterLd]>;
}

defm PAVGUSB  : I3DNow_binop_rm_int<0xBF, "pavgusb", I3DNOW_MISC_FUNC_ITINS, 1>;
defm PF2ID    : I3DNow_conv_rm_int<0x1D, "pf2id", I3DNOW_FCVT_F2I_ITINS>;
defm PFACC    : I3DNow_binop_rm_int<0xAE, "pfacc", I3DNOW_FALU_ITINS>;
defm PFADD    : I3DNow_binop_rm_int<0x9E, "pfadd", I3DNOW_FALU_ITINS, 1>;
defm PFCMPEQ  : I3DNow_binop_rm_int<0xB0, "pfcmpeq", I3DNOW_FALU_ITINS, 1>;
defm PFCMPGE  : I3DNow_binop_rm_int<0x90, "pfcmpge", I3DNOW_FALU_ITINS>;
defm PFCMPGT  : I3DNow_binop_rm_int<0xA0, "pfcmpgt", I3DNOW_FALU_ITINS>;
defm PFMAX    : I3DNow_binop_rm_int<0xA4, "pfmax", I3DNOW_FALU_ITINS>;
defm PFMIN    : I3DNow_binop_rm_int<0x94, "pfmin", I3DNOW_FALU_ITINS>;
defm PFMUL    : I3DNow_binop_rm_int<0xB4, "pfmul", I3DNOW_FALU_ITINS, 1>;
defm PFRCP    : I3DNow_conv_rm_int<0x96, "pfrcp", I3DNOW_FALU_ITINS>;
defm PFRCPIT1 : I3DNow_binop_rm_int<0xA6, "pfrcpit1", I3DNOW_FALU_ITINS>;
defm PFRCPIT2 : I3DNow_binop_rm_int<0xB6, "pfrcpit2", I3DNOW_FALU_ITINS>;
defm PFRSQIT1 : I3DNow_binop_rm_int<0xA7, "pfrsqit1", I3DNOW_FALU_ITINS>;
defm PFRSQRT  : I3DNow_conv_rm_int<0x97, "pfrsqrt", I3DNOW_FALU_ITINS>;
defm PFSUB    : I3DNow_binop_rm_int<0x9A, "pfsub", I3DNOW_FALU_ITINS, 1>;
defm PFSUBR   : I3DNow_binop_rm_int<0xAA, "pfsubr", I3DNOW_FALU_ITINS, 1>;
defm PI2FD    : I3DNow_conv_rm_int<0x0D, "pi2fd", I3DNOW_FCVT_I2F_ITINS>;
defm PMULHRW  : I3DNow_binop_rm_int<0xB7, "pmulhrw", I3DNOW_MISC_FUNC_ITINS, 1>;

def FEMMS : I3DNow<0x0E, RawFrm, (outs), (ins), "femms",
                   [(int_x86_mmx_femms)]>;

def PREFETCH : I3DNow<0x0D, MRM0m, (outs), (ins i8mem:$addr),
                      "prefetch\t$addr",
                      [(prefetch addr:$addr, (i32 0), imm, (i32 1))]>;

def PREFETCHW : I<0x0D, MRM1m, (outs), (ins i8mem:$addr), "prefetchw\t$addr",
                  [(prefetch addr:$addr, (i32 1), (i32 3), (i32 1))]>, TB,
                Requires<[HasPrefetchW]>;

// "3DNowA" instructions
defm PF2IW    : I3DNow_conv_rm_int<0x1C, "pf2iw", I3DNOW_FCVT_F2I_ITINS, "a">;
defm PI2FW    : I3DNow_conv_rm_int<0x0C, "pi2fw", I3DNOW_FCVT_I2F_ITINS, "a">;
defm PFNACC   : I3DNow_binop_rm_int<0x8A, "pfnacc", I3DNOW_FALU_ITINS, 0, "a">;
defm PFPNACC  : I3DNow_binop_rm_int<0x8E, "pfpnacc", I3DNOW_FALU_ITINS, 0, "a">;
defm PSWAPD   : I3DNow_conv_rm_int<0xBB, "pswapd", I3DNOW_PSHUF_ITINS, "a">;
