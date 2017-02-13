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

class I3DNow<bits<8> o, Format F, dag outs, dag ins, string asm, list<dag> pat>
      : I<o, F, outs, ins, asm, pat>, TB, Requires<[Has3DNow]> {
}

class I3DNow_binop<bits<8> o, Format F, dag ins, string Mnemonic, list<dag> pat>
      : I3DNow<o, F, (outs VR64:$dst), ins,
          !strconcat(Mnemonic, "\t{$src2, $dst|$dst, $src2}"), pat>,
        Has3DNow0F0FOpcode {
  // FIXME: The disassembler doesn't support Has3DNow0F0FOpcode yet.
  let isAsmParserOnly = 1;
  let Constraints = "$src1 = $dst";
}

class I3DNow_conv<bits<8> o, Format F, dag ins, string Mnemonic, list<dag> pat>
      : I3DNow<o, F, (outs VR64:$dst), ins,
          !strconcat(Mnemonic, "\t{$src, $dst|$dst, $src}"), pat>,
        Has3DNow0F0FOpcode {
  // FIXME: The disassembler doesn't support Has3DNow0F0FOpcode yet.
  let isAsmParserOnly = 1;
}

multiclass I3DNow_binop_rm<bits<8> opc, string Mn> {
  def rr : I3DNow_binop<opc, MRMSrcReg, (ins VR64:$src1, VR64:$src2), Mn, []>;
  def rm : I3DNow_binop<opc, MRMSrcMem, (ins VR64:$src1, i64mem:$src2), Mn, []>;
}

multiclass I3DNow_binop_rm_int<bits<8> opc, string Mn, bit Commutable = 0,
                               string Ver = ""> {
  let isCommutable = Commutable in
  def rr : I3DNow_binop<opc, MRMSrcReg, (ins VR64:$src1, VR64:$src2), Mn,
    [(set VR64:$dst, (!cast<Intrinsic>(
      !strconcat("int_x86_3dnow", Ver, "_", Mn)) VR64:$src1, VR64:$src2))]>;
  def rm : I3DNow_binop<opc, MRMSrcMem, (ins VR64:$src1, i64mem:$src2), Mn,
    [(set VR64:$dst, (!cast<Intrinsic>(
      !strconcat("int_x86_3dnow", Ver, "_", Mn)) VR64:$src1,
        (bitconvert (load_mmx addr:$src2))))]>;
}

multiclass I3DNow_conv_rm<bits<8> opc, string Mn> {
  def rr : I3DNow_conv<opc, MRMSrcReg, (ins VR64:$src1), Mn, []>;
  def rm : I3DNow_conv<opc, MRMSrcMem, (ins i64mem:$src1), Mn, []>;
}

multiclass I3DNow_conv_rm_int<bits<8> opc, string Mn, string Ver = ""> {
  def rr : I3DNow_conv<opc, MRMSrcReg, (ins VR64:$src), Mn,
    [(set VR64:$dst, (!cast<Intrinsic>(
      !strconcat("int_x86_3dnow", Ver, "_", Mn)) VR64:$src))]>;
  def rm : I3DNow_conv<opc, MRMSrcMem, (ins i64mem:$src), Mn,
    [(set VR64:$dst, (!cast<Intrinsic>(
      !strconcat("int_x86_3dnow", Ver, "_", Mn))
        (bitconvert (load_mmx addr:$src))))]>;
}

defm PAVGUSB  : I3DNow_binop_rm_int<0xBF, "pavgusb", 1>;
defm PF2ID    : I3DNow_conv_rm_int<0x1D, "pf2id">;
defm PFACC    : I3DNow_binop_rm_int<0xAE, "pfacc">;
defm PFADD    : I3DNow_binop_rm_int<0x9E, "pfadd", 1>;
defm PFCMPEQ  : I3DNow_binop_rm_int<0xB0, "pfcmpeq", 1>;
defm PFCMPGE  : I3DNow_binop_rm_int<0x90, "pfcmpge">;
defm PFCMPGT  : I3DNow_binop_rm_int<0xA0, "pfcmpgt">;
defm PFMAX    : I3DNow_binop_rm_int<0xA4, "pfmax">;
defm PFMIN    : I3DNow_binop_rm_int<0x94, "pfmin">;
defm PFMUL    : I3DNow_binop_rm_int<0xB4, "pfmul", 1>;
defm PFRCP    : I3DNow_conv_rm_int<0x96, "pfrcp">;
defm PFRCPIT1 : I3DNow_binop_rm_int<0xA6, "pfrcpit1">;
defm PFRCPIT2 : I3DNow_binop_rm_int<0xB6, "pfrcpit2">;
defm PFRSQIT1 : I3DNow_binop_rm_int<0xA7, "pfrsqit1">;
defm PFRSQRT  : I3DNow_conv_rm_int<0x97, "pfrsqrt">;
defm PFSUB    : I3DNow_binop_rm_int<0x9A, "pfsub", 1>;
defm PFSUBR   : I3DNow_binop_rm_int<0xAA, "pfsubr", 1>;
defm PI2FD    : I3DNow_conv_rm_int<0x0D, "pi2fd">;
defm PMULHRW  : I3DNow_binop_rm_int<0xB7, "pmulhrw", 1>;


def FEMMS : I3DNow<0x0E, RawFrm, (outs), (ins), "femms",
                   [(int_x86_mmx_femms)]>;

def PREFETCH : I3DNow<0x0D, MRM0m, (outs), (ins i8mem:$addr),
                      "prefetch\t$addr",
                      [(prefetch addr:$addr, (i32 0), imm, (i32 1))]>;

def PREFETCHW : I<0x0D, MRM1m, (outs), (ins i8mem:$addr), "prefetchw\t$addr",
                  [(prefetch addr:$addr, (i32 1), (i32 3), (i32 1))]>, TB,
                Requires<[HasPrefetchW]>;

// "3DNowA" instructions
defm PF2IW    : I3DNow_conv_rm_int<0x1C, "pf2iw", "a">;
defm PI2FW    : I3DNow_conv_rm_int<0x0C, "pi2fw", "a">;
defm PFNACC   : I3DNow_binop_rm_int<0x8A, "pfnacc", 0, "a">;
defm PFPNACC  : I3DNow_binop_rm_int<0x8E, "pfpnacc", 0, "a">;
defm PSWAPD   : I3DNow_conv_rm_int<0xBB, "pswapd", "a">;
