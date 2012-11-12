//===- X86InstrFPStack.td - FPU Instruction Set ------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 x87 FPU instruction set, defining the
// instructions, and properties of the instructions which are needed for code
// generation, machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FPStack specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDTX86FpGet2    : SDTypeProfile<2, 0, [SDTCisVT<0, f80>, 
                                           SDTCisVT<1, f80>]>;
def SDTX86Fld       : SDTypeProfile<1, 2, [SDTCisFP<0>,
                                           SDTCisPtrTy<1>, 
                                           SDTCisVT<2, OtherVT>]>;
def SDTX86Fst       : SDTypeProfile<0, 3, [SDTCisFP<0>,
                                           SDTCisPtrTy<1>, 
                                           SDTCisVT<2, OtherVT>]>;
def SDTX86Fild      : SDTypeProfile<1, 2, [SDTCisFP<0>, SDTCisPtrTy<1>,
                                           SDTCisVT<2, OtherVT>]>;
def SDTX86Fnstsw    : SDTypeProfile<1, 1, [SDTCisVT<0, i16>, SDTCisVT<1, i16>]>;
def SDTX86FpToIMem  : SDTypeProfile<0, 2, [SDTCisFP<0>, SDTCisPtrTy<1>]>;

def SDTX86CwdStore  : SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>;

def X86fld          : SDNode<"X86ISD::FLD", SDTX86Fld,
                             [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def X86fst          : SDNode<"X86ISD::FST", SDTX86Fst,
                             [SDNPHasChain, SDNPInGlue, SDNPMayStore,
                              SDNPMemOperand]>;
def X86fild         : SDNode<"X86ISD::FILD", SDTX86Fild,
                             [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def X86fildflag     : SDNode<"X86ISD::FILD_FLAG", SDTX86Fild,
                             [SDNPHasChain, SDNPOutGlue, SDNPMayLoad,
                              SDNPMemOperand]>;
def X86fp_stsw      : SDNode<"X86ISD::FNSTSW16r", SDTX86Fnstsw>;
def X86fp_to_i16mem : SDNode<"X86ISD::FP_TO_INT16_IN_MEM", SDTX86FpToIMem,
                             [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def X86fp_to_i32mem : SDNode<"X86ISD::FP_TO_INT32_IN_MEM", SDTX86FpToIMem,
                             [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def X86fp_to_i64mem : SDNode<"X86ISD::FP_TO_INT64_IN_MEM", SDTX86FpToIMem,
                             [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def X86fp_cwd_get16 : SDNode<"X86ISD::FNSTCW16m",          SDTX86CwdStore,
                             [SDNPHasChain, SDNPMayStore, SDNPSideEffect,
                              SDNPMemOperand]>;

//===----------------------------------------------------------------------===//
// FPStack pattern fragments
//===----------------------------------------------------------------------===//

def fpimm0 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def fpimmneg0 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(-0.0);
}]>;

def fpimm1 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(+1.0);
}]>;

def fpimmneg1 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(-1.0);
}]>;

// Some 'special' instructions
let usesCustomInserter = 1 in {  // Expanded after instruction selection.
  def FP32_TO_INT16_IN_MEM : PseudoI<(outs), (ins i16mem:$dst, RFP32:$src),
                              [(X86fp_to_i16mem RFP32:$src, addr:$dst)]>;
  def FP32_TO_INT32_IN_MEM : PseudoI<(outs), (ins i32mem:$dst, RFP32:$src),
                              [(X86fp_to_i32mem RFP32:$src, addr:$dst)]>;
  def FP32_TO_INT64_IN_MEM : PseudoI<(outs), (ins i64mem:$dst, RFP32:$src),
                              [(X86fp_to_i64mem RFP32:$src, addr:$dst)]>;
  def FP64_TO_INT16_IN_MEM : PseudoI<(outs), (ins i16mem:$dst, RFP64:$src),
                              [(X86fp_to_i16mem RFP64:$src, addr:$dst)]>;
  def FP64_TO_INT32_IN_MEM : PseudoI<(outs), (ins i32mem:$dst, RFP64:$src),
                              [(X86fp_to_i32mem RFP64:$src, addr:$dst)]>;
  def FP64_TO_INT64_IN_MEM : PseudoI<(outs), (ins i64mem:$dst, RFP64:$src),
                              [(X86fp_to_i64mem RFP64:$src, addr:$dst)]>;
  def FP80_TO_INT16_IN_MEM : PseudoI<(outs), (ins i16mem:$dst, RFP80:$src),
                              [(X86fp_to_i16mem RFP80:$src, addr:$dst)]>;
  def FP80_TO_INT32_IN_MEM : PseudoI<(outs), (ins i32mem:$dst, RFP80:$src),
                              [(X86fp_to_i32mem RFP80:$src, addr:$dst)]>;
  def FP80_TO_INT64_IN_MEM : PseudoI<(outs), (ins i64mem:$dst, RFP80:$src),
                              [(X86fp_to_i64mem RFP80:$src, addr:$dst)]>;
}

// All FP Stack operations are represented with four instructions here.  The
// first three instructions, generated by the instruction selector, use "RFP32"
// "RFP64" or "RFP80" registers: traditional register files to reference 32-bit,
// 64-bit or 80-bit floating point values.  These sizes apply to the values, 
// not the registers, which are always 80 bits; RFP32, RFP64 and RFP80 can be
// copied to each other without losing information.  These instructions are all
// pseudo instructions and use the "_Fp" suffix.
// In some cases there are additional variants with a mixture of different
// register sizes.
// The second instruction is defined with FPI, which is the actual instruction
// emitted by the assembler.  These use "RST" registers, although frequently
// the actual register(s) used are implicit.  These are always 80 bits.
// The FP stackifier pass converts one to the other after register allocation 
// occurs.
//
// Note that the FpI instruction should have instruction selection info (e.g.
// a pattern) and the FPI instruction should have emission info (e.g. opcode
// encoding and asm printing info).

// Pseudo Instruction for FP stack return values.
def FpPOP_RETVAL : FpI_<(outs RFP80:$dst), (ins), SpecialFP, []>;

// FpIf32, FpIf64 - Floating Point Pseudo Instruction template.
// f32 instructions can use SSE1 and are predicated on FPStackf32 == !SSE1.
// f64 instructions can use SSE2 and are predicated on FPStackf64 == !SSE2.
// f80 instructions cannot use SSE and use neither of these.
class FpIf32<dag outs, dag ins, FPFormat fp, list<dag> pattern> :
  FpI_<outs, ins, fp, pattern>, Requires<[FPStackf32]>;
class FpIf64<dag outs, dag ins, FPFormat fp, list<dag> pattern> :
  FpI_<outs, ins, fp, pattern>, Requires<[FPStackf64]>;

// Factoring for arithmetic.
multiclass FPBinary_rr<SDNode OpNode> {
// Register op register -> register
// These are separated out because they have no reversed form.
def _Fp32 : FpIf32<(outs RFP32:$dst), (ins RFP32:$src1, RFP32:$src2), TwoArgFP,
                [(set RFP32:$dst, (OpNode RFP32:$src1, RFP32:$src2))]>;
def _Fp64 : FpIf64<(outs RFP64:$dst), (ins RFP64:$src1, RFP64:$src2), TwoArgFP,
                [(set RFP64:$dst, (OpNode RFP64:$src1, RFP64:$src2))]>;
def _Fp80 : FpI_<(outs RFP80:$dst), (ins RFP80:$src1, RFP80:$src2), TwoArgFP,
                [(set RFP80:$dst, (OpNode RFP80:$src1, RFP80:$src2))]>;
}
// The FopST0 series are not included here because of the irregularities
// in where the 'r' goes in assembly output.
// These instructions cannot address 80-bit memory.
multiclass FPBinary<SDNode OpNode, Format fp, string asmstring> {
// ST(0) = ST(0) + [mem]
def _Fp32m  : FpIf32<(outs RFP32:$dst), 
                     (ins RFP32:$src1, f32mem:$src2), OneArgFPRW,
                  [(set RFP32:$dst, 
                    (OpNode RFP32:$src1, (loadf32 addr:$src2)))]>;
def _Fp64m  : FpIf64<(outs RFP64:$dst), 
                     (ins RFP64:$src1, f64mem:$src2), OneArgFPRW,
                  [(set RFP64:$dst, 
                    (OpNode RFP64:$src1, (loadf64 addr:$src2)))]>;
def _Fp64m32: FpIf64<(outs RFP64:$dst), 
                     (ins RFP64:$src1, f32mem:$src2), OneArgFPRW,
                  [(set RFP64:$dst, 
                    (OpNode RFP64:$src1, (f64 (extloadf32 addr:$src2))))]>;
def _Fp80m32: FpI_<(outs RFP80:$dst), 
                   (ins RFP80:$src1, f32mem:$src2), OneArgFPRW,
                  [(set RFP80:$dst, 
                    (OpNode RFP80:$src1, (f80 (extloadf32 addr:$src2))))]>;
def _Fp80m64: FpI_<(outs RFP80:$dst), 
                   (ins RFP80:$src1, f64mem:$src2), OneArgFPRW,
                  [(set RFP80:$dst, 
                    (OpNode RFP80:$src1, (f80 (extloadf64 addr:$src2))))]>;
def _F32m  : FPI<0xD8, fp, (outs), (ins f32mem:$src), 
                 !strconcat("f", asmstring, "{s}\t$src")> { 
  let mayLoad = 1; 
}
def _F64m  : FPI<0xDC, fp, (outs), (ins f64mem:$src), 
                 !strconcat("f", asmstring, "{l}\t$src")> { 
  let mayLoad = 1; 
}
// ST(0) = ST(0) + [memint]
def _FpI16m32 : FpIf32<(outs RFP32:$dst), (ins RFP32:$src1, i16mem:$src2), 
                       OneArgFPRW,
                    [(set RFP32:$dst, (OpNode RFP32:$src1,
                                       (X86fild addr:$src2, i16)))]>;
def _FpI32m32 : FpIf32<(outs RFP32:$dst), (ins RFP32:$src1, i32mem:$src2), 
                       OneArgFPRW,
                    [(set RFP32:$dst, (OpNode RFP32:$src1,
                                       (X86fild addr:$src2, i32)))]>;
def _FpI16m64 : FpIf64<(outs RFP64:$dst), (ins RFP64:$src1, i16mem:$src2), 
                       OneArgFPRW,
                    [(set RFP64:$dst, (OpNode RFP64:$src1,
                                       (X86fild addr:$src2, i16)))]>;
def _FpI32m64 : FpIf64<(outs RFP64:$dst), (ins RFP64:$src1, i32mem:$src2), 
                       OneArgFPRW,
                    [(set RFP64:$dst, (OpNode RFP64:$src1,
                                       (X86fild addr:$src2, i32)))]>;
def _FpI16m80 : FpI_<(outs RFP80:$dst), (ins RFP80:$src1, i16mem:$src2), 
                       OneArgFPRW,
                    [(set RFP80:$dst, (OpNode RFP80:$src1,
                                       (X86fild addr:$src2, i16)))]>;
def _FpI32m80 : FpI_<(outs RFP80:$dst), (ins RFP80:$src1, i32mem:$src2), 
                       OneArgFPRW,
                    [(set RFP80:$dst, (OpNode RFP80:$src1,
                                       (X86fild addr:$src2, i32)))]>;
def _FI16m  : FPI<0xDE, fp, (outs), (ins i16mem:$src), 
                  !strconcat("fi", asmstring, "{s}\t$src")> { 
  let mayLoad = 1; 
}
def _FI32m  : FPI<0xDA, fp, (outs), (ins i32mem:$src), 
                  !strconcat("fi", asmstring, "{l}\t$src")> { 
  let mayLoad = 1; 
}
}

let Defs = [FPSW] in {
defm ADD : FPBinary_rr<fadd>;
defm SUB : FPBinary_rr<fsub>;
defm MUL : FPBinary_rr<fmul>;
defm DIV : FPBinary_rr<fdiv>;
defm ADD : FPBinary<fadd, MRM0m, "add">;
defm SUB : FPBinary<fsub, MRM4m, "sub">;
defm SUBR: FPBinary<fsub ,MRM5m, "subr">;
defm MUL : FPBinary<fmul, MRM1m, "mul">;
defm DIV : FPBinary<fdiv, MRM6m, "div">;
defm DIVR: FPBinary<fdiv, MRM7m, "divr">;
}

class FPST0rInst<bits<8> o, string asm>
  : FPI<o, AddRegFrm, (outs), (ins RST:$op), asm>, D8;
class FPrST0Inst<bits<8> o, string asm>
  : FPI<o, AddRegFrm, (outs), (ins RST:$op), asm>, DC;
class FPrST0PInst<bits<8> o, string asm>
  : FPI<o, AddRegFrm, (outs), (ins RST:$op), asm>, DE;

// NOTE: GAS and apparently all other AT&T style assemblers have a broken notion
// of some of the 'reverse' forms of the fsub and fdiv instructions.  As such,
// we have to put some 'r's in and take them out of weird places.
def ADD_FST0r   : FPST0rInst <0xC0, "fadd\t$op">;
def ADD_FrST0   : FPrST0Inst <0xC0, "fadd\t{%st(0), $op|$op, ST(0)}">;
def ADD_FPrST0  : FPrST0PInst<0xC0, "faddp\t$op">;
def SUBR_FST0r  : FPST0rInst <0xE8, "fsubr\t$op">;
def SUB_FrST0   : FPrST0Inst <0xE8, "fsub{r}\t{%st(0), $op|$op, ST(0)}">;
def SUB_FPrST0  : FPrST0PInst<0xE8, "fsub{r}p\t$op">;
def SUB_FST0r   : FPST0rInst <0xE0, "fsub\t$op">;
def SUBR_FrST0  : FPrST0Inst <0xE0, "fsub{|r}\t{%st(0), $op|$op, ST(0)}">;
def SUBR_FPrST0 : FPrST0PInst<0xE0, "fsub{|r}p\t$op">;
def MUL_FST0r   : FPST0rInst <0xC8, "fmul\t$op">;
def MUL_FrST0   : FPrST0Inst <0xC8, "fmul\t{%st(0), $op|$op, ST(0)}">;
def MUL_FPrST0  : FPrST0PInst<0xC8, "fmulp\t$op">;
def DIVR_FST0r  : FPST0rInst <0xF8, "fdivr\t$op">;
def DIV_FrST0   : FPrST0Inst <0xF8, "fdiv{r}\t{%st(0), $op|$op, ST(0)}">;
def DIV_FPrST0  : FPrST0PInst<0xF8, "fdiv{r}p\t$op">;
def DIV_FST0r   : FPST0rInst <0xF0, "fdiv\t$op">;
def DIVR_FrST0  : FPrST0Inst <0xF0, "fdiv{|r}\t{%st(0), $op|$op, ST(0)}">;
def DIVR_FPrST0 : FPrST0PInst<0xF0, "fdiv{|r}p\t$op">;

def COM_FST0r   : FPST0rInst <0xD0, "fcom\t$op">;
def COMP_FST0r  : FPST0rInst <0xD8, "fcomp\t$op">;

// Unary operations.
multiclass FPUnary<SDNode OpNode, bits<8> opcode, string asmstring> {
def _Fp32  : FpIf32<(outs RFP32:$dst), (ins RFP32:$src), OneArgFPRW,
                 [(set RFP32:$dst, (OpNode RFP32:$src))]>;
def _Fp64  : FpIf64<(outs RFP64:$dst), (ins RFP64:$src), OneArgFPRW,
                 [(set RFP64:$dst, (OpNode RFP64:$src))]>;
def _Fp80  : FpI_<(outs RFP80:$dst), (ins RFP80:$src), OneArgFPRW,
                 [(set RFP80:$dst, (OpNode RFP80:$src))]>;
def _F     : FPI<opcode, RawFrm, (outs), (ins), asmstring>, D9;
}

let Defs = [FPSW] in {
defm CHS : FPUnary<fneg, 0xE0, "fchs">;
defm ABS : FPUnary<fabs, 0xE1, "fabs">;
defm SQRT: FPUnary<fsqrt,0xFA, "fsqrt">;
defm SIN : FPUnary<fsin, 0xFE, "fsin">;
defm COS : FPUnary<fcos, 0xFF, "fcos">;

let neverHasSideEffects = 1 in {
def TST_Fp32  : FpIf32<(outs), (ins RFP32:$src), OneArgFP, []>;
def TST_Fp64  : FpIf64<(outs), (ins RFP64:$src), OneArgFP, []>;
def TST_Fp80  : FpI_<(outs), (ins RFP80:$src), OneArgFP, []>;
}
def TST_F  : FPI<0xE4, RawFrm, (outs), (ins), "ftst">, D9;
} // Defs = [FPSW]

// Versions of FP instructions that take a single memory operand.  Added for the
//   disassembler; remove as they are included with patterns elsewhere.
def FCOM32m  : FPI<0xD8, MRM2m, (outs), (ins f32mem:$src), "fcom{s}\t$src">;
def FCOMP32m : FPI<0xD8, MRM3m, (outs), (ins f32mem:$src), "fcomp{s}\t$src">;

def FLDENVm  : FPI<0xD9, MRM4m, (outs), (ins f32mem:$src), "fldenv\t$src">;
def FSTENVm  : FPI<0xD9, MRM6m, (outs f32mem:$dst), (ins), "fnstenv\t$dst">;

def FICOM32m : FPI<0xDA, MRM2m, (outs), (ins i32mem:$src), "ficom{l}\t$src">;
def FICOMP32m: FPI<0xDA, MRM3m, (outs), (ins i32mem:$src), "ficomp{l}\t$src">;

def FCOM64m  : FPI<0xDC, MRM2m, (outs), (ins f64mem:$src), "fcom{l}\t$src">;
def FCOMP64m : FPI<0xDC, MRM3m, (outs), (ins f64mem:$src), "fcomp{l}\t$src">;

def FRSTORm  : FPI<0xDD, MRM4m, (outs f32mem:$dst), (ins), "frstor\t$dst">;
def FSAVEm   : FPI<0xDD, MRM6m, (outs f32mem:$dst), (ins), "fnsave\t$dst">;
def FNSTSWm  : FPI<0xDD, MRM7m, (outs f32mem:$dst), (ins), "fnstsw\t$dst">;

def FICOM16m : FPI<0xDE, MRM2m, (outs), (ins i16mem:$src), "ficom{s}\t$src">;
def FICOMP16m: FPI<0xDE, MRM3m, (outs), (ins i16mem:$src), "ficomp{s}\t$src">;

def FBLDm    : FPI<0xDF, MRM4m, (outs), (ins f32mem:$src), "fbld\t$src">;
def FBSTPm   : FPI<0xDF, MRM6m, (outs f32mem:$dst), (ins), "fbstp\t$dst">;

// Floating point cmovs.
class FpIf32CMov<dag outs, dag ins, FPFormat fp, list<dag> pattern> :
  FpI_<outs, ins, fp, pattern>, Requires<[FPStackf32, HasCMov]>;
class FpIf64CMov<dag outs, dag ins, FPFormat fp, list<dag> pattern> :
  FpI_<outs, ins, fp, pattern>, Requires<[FPStackf64, HasCMov]>;

multiclass FPCMov<PatLeaf cc> {
  def _Fp32  : FpIf32CMov<(outs RFP32:$dst), (ins RFP32:$src1, RFP32:$src2),
                       CondMovFP,
                     [(set RFP32:$dst, (X86cmov RFP32:$src1, RFP32:$src2,
                                        cc, EFLAGS))]>;
  def _Fp64  : FpIf64CMov<(outs RFP64:$dst), (ins RFP64:$src1, RFP64:$src2),
                       CondMovFP,
                     [(set RFP64:$dst, (X86cmov RFP64:$src1, RFP64:$src2,
                                        cc, EFLAGS))]>;
  def _Fp80  : FpI_<(outs RFP80:$dst), (ins RFP80:$src1, RFP80:$src2),
                     CondMovFP,
                     [(set RFP80:$dst, (X86cmov RFP80:$src1, RFP80:$src2,
                                        cc, EFLAGS))]>,
                                        Requires<[HasCMov]>;
}

let Defs = [FPSW] in {
let Uses = [EFLAGS], Constraints = "$src1 = $dst" in {
defm CMOVB  : FPCMov<X86_COND_B>;
defm CMOVBE : FPCMov<X86_COND_BE>;
defm CMOVE  : FPCMov<X86_COND_E>;
defm CMOVP  : FPCMov<X86_COND_P>;
defm CMOVNB : FPCMov<X86_COND_AE>;
defm CMOVNBE: FPCMov<X86_COND_A>;
defm CMOVNE : FPCMov<X86_COND_NE>;
defm CMOVNP : FPCMov<X86_COND_NP>;
} // Uses = [EFLAGS], Constraints = "$src1 = $dst"

let Predicates = [HasCMov] in {
// These are not factored because there's no clean way to pass DA/DB.
def CMOVB_F  : FPI<0xC0, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovb\t{$op, %st(0)|ST(0), $op}">, DA;
def CMOVBE_F : FPI<0xD0, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovbe\t{$op, %st(0)|ST(0), $op}">, DA;
def CMOVE_F  : FPI<0xC8, AddRegFrm, (outs RST:$op), (ins),
                  "fcmove\t{$op, %st(0)|ST(0), $op}">, DA;
def CMOVP_F  : FPI<0xD8, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovu\t {$op, %st(0)|ST(0), $op}">, DA;
def CMOVNB_F : FPI<0xC0, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovnb\t{$op, %st(0)|ST(0), $op}">, DB;
def CMOVNBE_F: FPI<0xD0, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovnbe\t{$op, %st(0)|ST(0), $op}">, DB;
def CMOVNE_F : FPI<0xC8, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovne\t{$op, %st(0)|ST(0), $op}">, DB;
def CMOVNP_F : FPI<0xD8, AddRegFrm, (outs RST:$op), (ins),
                  "fcmovnu\t{$op, %st(0)|ST(0), $op}">, DB;
} // Predicates = [HasCMov]

// Floating point loads & stores.
let canFoldAsLoad = 1 in {
def LD_Fp32m   : FpIf32<(outs RFP32:$dst), (ins f32mem:$src), ZeroArgFP,
                  [(set RFP32:$dst, (loadf32 addr:$src))]>;
let isReMaterializable = 1 in
  def LD_Fp64m : FpIf64<(outs RFP64:$dst), (ins f64mem:$src), ZeroArgFP,
                  [(set RFP64:$dst, (loadf64 addr:$src))]>;
def LD_Fp80m   : FpI_<(outs RFP80:$dst), (ins f80mem:$src), ZeroArgFP,
                  [(set RFP80:$dst, (loadf80 addr:$src))]>;
}
def LD_Fp32m64 : FpIf64<(outs RFP64:$dst), (ins f32mem:$src), ZeroArgFP,
                  [(set RFP64:$dst, (f64 (extloadf32 addr:$src)))]>;
def LD_Fp64m80 : FpI_<(outs RFP80:$dst), (ins f64mem:$src), ZeroArgFP,
                  [(set RFP80:$dst, (f80 (extloadf64 addr:$src)))]>;
def LD_Fp32m80 : FpI_<(outs RFP80:$dst), (ins f32mem:$src), ZeroArgFP,
                  [(set RFP80:$dst, (f80 (extloadf32 addr:$src)))]>;
def ILD_Fp16m32: FpIf32<(outs RFP32:$dst), (ins i16mem:$src), ZeroArgFP,
                  [(set RFP32:$dst, (X86fild addr:$src, i16))]>;
def ILD_Fp32m32: FpIf32<(outs RFP32:$dst), (ins i32mem:$src), ZeroArgFP,
                  [(set RFP32:$dst, (X86fild addr:$src, i32))]>;
def ILD_Fp64m32: FpIf32<(outs RFP32:$dst), (ins i64mem:$src), ZeroArgFP,
                  [(set RFP32:$dst, (X86fild addr:$src, i64))]>;
def ILD_Fp16m64: FpIf64<(outs RFP64:$dst), (ins i16mem:$src), ZeroArgFP,
                  [(set RFP64:$dst, (X86fild addr:$src, i16))]>;
def ILD_Fp32m64: FpIf64<(outs RFP64:$dst), (ins i32mem:$src), ZeroArgFP,
                  [(set RFP64:$dst, (X86fild addr:$src, i32))]>;
def ILD_Fp64m64: FpIf64<(outs RFP64:$dst), (ins i64mem:$src), ZeroArgFP,
                  [(set RFP64:$dst, (X86fild addr:$src, i64))]>;
def ILD_Fp16m80: FpI_<(outs RFP80:$dst), (ins i16mem:$src), ZeroArgFP,
                  [(set RFP80:$dst, (X86fild addr:$src, i16))]>;
def ILD_Fp32m80: FpI_<(outs RFP80:$dst), (ins i32mem:$src), ZeroArgFP,
                  [(set RFP80:$dst, (X86fild addr:$src, i32))]>;
def ILD_Fp64m80: FpI_<(outs RFP80:$dst), (ins i64mem:$src), ZeroArgFP,
                  [(set RFP80:$dst, (X86fild addr:$src, i64))]>;

def ST_Fp32m   : FpIf32<(outs), (ins f32mem:$op, RFP32:$src), OneArgFP,
                  [(store RFP32:$src, addr:$op)]>;
def ST_Fp64m32 : FpIf64<(outs), (ins f32mem:$op, RFP64:$src), OneArgFP,
                  [(truncstoref32 RFP64:$src, addr:$op)]>;
def ST_Fp64m   : FpIf64<(outs), (ins f64mem:$op, RFP64:$src), OneArgFP,
                  [(store RFP64:$src, addr:$op)]>;
def ST_Fp80m32 : FpI_<(outs), (ins f32mem:$op, RFP80:$src), OneArgFP,
                  [(truncstoref32 RFP80:$src, addr:$op)]>;
def ST_Fp80m64 : FpI_<(outs), (ins f64mem:$op, RFP80:$src), OneArgFP,
                  [(truncstoref64 RFP80:$src, addr:$op)]>;
// FST does not support 80-bit memory target; FSTP must be used.

let mayStore = 1, neverHasSideEffects = 1 in {
def ST_FpP32m    : FpIf32<(outs), (ins f32mem:$op, RFP32:$src), OneArgFP, []>;
def ST_FpP64m32  : FpIf64<(outs), (ins f32mem:$op, RFP64:$src), OneArgFP, []>;
def ST_FpP64m    : FpIf64<(outs), (ins f64mem:$op, RFP64:$src), OneArgFP, []>;
def ST_FpP80m32  : FpI_<(outs), (ins f32mem:$op, RFP80:$src), OneArgFP, []>;
def ST_FpP80m64  : FpI_<(outs), (ins f64mem:$op, RFP80:$src), OneArgFP, []>;
}
def ST_FpP80m    : FpI_<(outs), (ins f80mem:$op, RFP80:$src), OneArgFP,
                    [(store RFP80:$src, addr:$op)]>;
let mayStore = 1, neverHasSideEffects = 1 in {
def IST_Fp16m32  : FpIf32<(outs), (ins i16mem:$op, RFP32:$src), OneArgFP, []>;
def IST_Fp32m32  : FpIf32<(outs), (ins i32mem:$op, RFP32:$src), OneArgFP, []>;
def IST_Fp64m32  : FpIf32<(outs), (ins i64mem:$op, RFP32:$src), OneArgFP, []>;
def IST_Fp16m64  : FpIf64<(outs), (ins i16mem:$op, RFP64:$src), OneArgFP, []>;
def IST_Fp32m64  : FpIf64<(outs), (ins i32mem:$op, RFP64:$src), OneArgFP, []>;
def IST_Fp64m64  : FpIf64<(outs), (ins i64mem:$op, RFP64:$src), OneArgFP, []>;
def IST_Fp16m80  : FpI_<(outs), (ins i16mem:$op, RFP80:$src), OneArgFP, []>;
def IST_Fp32m80  : FpI_<(outs), (ins i32mem:$op, RFP80:$src), OneArgFP, []>;
def IST_Fp64m80  : FpI_<(outs), (ins i64mem:$op, RFP80:$src), OneArgFP, []>;
}

let mayLoad = 1 in {
def LD_F32m   : FPI<0xD9, MRM0m, (outs), (ins f32mem:$src), "fld{s}\t$src",
                    IIC_FLD>;
def LD_F64m   : FPI<0xDD, MRM0m, (outs), (ins f64mem:$src), "fld{l}\t$src",
                    IIC_FLD>;
def LD_F80m   : FPI<0xDB, MRM5m, (outs), (ins f80mem:$src), "fld{t}\t$src",
                    IIC_FLD80>;
def ILD_F16m  : FPI<0xDF, MRM0m, (outs), (ins i16mem:$src), "fild{s}\t$src",
                    IIC_FILD>;
def ILD_F32m  : FPI<0xDB, MRM0m, (outs), (ins i32mem:$src), "fild{l}\t$src",
                    IIC_FILD>;
def ILD_F64m  : FPI<0xDF, MRM5m, (outs), (ins i64mem:$src), "fild{ll}\t$src",
                    IIC_FILD>;
}
let mayStore = 1 in {
def ST_F32m   : FPI<0xD9, MRM2m, (outs), (ins f32mem:$dst), "fst{s}\t$dst",
                    IIC_FST>;
def ST_F64m   : FPI<0xDD, MRM2m, (outs), (ins f64mem:$dst), "fst{l}\t$dst",
                    IIC_FST>;
def ST_FP32m  : FPI<0xD9, MRM3m, (outs), (ins f32mem:$dst), "fstp{s}\t$dst",
                    IIC_FST>;
def ST_FP64m  : FPI<0xDD, MRM3m, (outs), (ins f64mem:$dst), "fstp{l}\t$dst",
                    IIC_FST>;
def ST_FP80m  : FPI<0xDB, MRM7m, (outs), (ins f80mem:$dst), "fstp{t}\t$dst",
                    IIC_FST80>;
def IST_F16m  : FPI<0xDF, MRM2m, (outs), (ins i16mem:$dst), "fist{s}\t$dst",
                    IIC_FIST>;
def IST_F32m  : FPI<0xDB, MRM2m, (outs), (ins i32mem:$dst), "fist{l}\t$dst",
                    IIC_FIST>;
def IST_FP16m : FPI<0xDF, MRM3m, (outs), (ins i16mem:$dst), "fistp{s}\t$dst",
                    IIC_FIST>;
def IST_FP32m : FPI<0xDB, MRM3m, (outs), (ins i32mem:$dst), "fistp{l}\t$dst",
                    IIC_FIST>;
def IST_FP64m : FPI<0xDF, MRM7m, (outs), (ins i64mem:$dst), "fistp{ll}\t$dst",
                    IIC_FIST>;
}

// FISTTP requires SSE3 even though it's a FPStack op.
let Predicates = [HasSSE3] in {
def ISTT_Fp16m32 : FpI_<(outs), (ins i16mem:$op, RFP32:$src), OneArgFP,
                    [(X86fp_to_i16mem RFP32:$src, addr:$op)]>;
def ISTT_Fp32m32 : FpI_<(outs), (ins i32mem:$op, RFP32:$src), OneArgFP,
                    [(X86fp_to_i32mem RFP32:$src, addr:$op)]>;
def ISTT_Fp64m32 : FpI_<(outs), (ins i64mem:$op, RFP32:$src), OneArgFP,
                    [(X86fp_to_i64mem RFP32:$src, addr:$op)]>;
def ISTT_Fp16m64 : FpI_<(outs), (ins i16mem:$op, RFP64:$src), OneArgFP,
                    [(X86fp_to_i16mem RFP64:$src, addr:$op)]>;
def ISTT_Fp32m64 : FpI_<(outs), (ins i32mem:$op, RFP64:$src), OneArgFP,
                    [(X86fp_to_i32mem RFP64:$src, addr:$op)]>;
def ISTT_Fp64m64 : FpI_<(outs), (ins i64mem:$op, RFP64:$src), OneArgFP,
                    [(X86fp_to_i64mem RFP64:$src, addr:$op)]>;
def ISTT_Fp16m80 : FpI_<(outs), (ins i16mem:$op, RFP80:$src), OneArgFP,
                    [(X86fp_to_i16mem RFP80:$src, addr:$op)]>;
def ISTT_Fp32m80 : FpI_<(outs), (ins i32mem:$op, RFP80:$src), OneArgFP,
                    [(X86fp_to_i32mem RFP80:$src, addr:$op)]>;
def ISTT_Fp64m80 : FpI_<(outs), (ins i64mem:$op, RFP80:$src), OneArgFP,
                    [(X86fp_to_i64mem RFP80:$src, addr:$op)]>;
} // Predicates = [HasSSE3]

let mayStore = 1 in {
def ISTT_FP16m : FPI<0xDF, MRM1m, (outs), (ins i16mem:$dst), "fisttp{s}\t$dst",
  IIC_FST>;
def ISTT_FP32m : FPI<0xDB, MRM1m, (outs), (ins i32mem:$dst), "fisttp{l}\t$dst",
  IIC_FST>;
def ISTT_FP64m : FPI<0xDD, MRM1m, (outs), (ins i64mem:$dst), 
  "fisttp{ll}\t$dst", IIC_FST>;
}

// FP Stack manipulation instructions.
def LD_Frr   : FPI<0xC0, AddRegFrm, (outs), (ins RST:$op), "fld\t$op",
                   IIC_FLD>, D9;
def ST_Frr   : FPI<0xD0, AddRegFrm, (outs), (ins RST:$op), "fst\t$op",
                   IIC_FST>, DD;
def ST_FPrr  : FPI<0xD8, AddRegFrm, (outs), (ins RST:$op), "fstp\t$op",
                   IIC_FST>, DD;
def XCH_F    : FPI<0xC8, AddRegFrm, (outs), (ins RST:$op), "fxch\t$op",
                   IIC_FXCH>, D9;

// Floating point constant loads.
let isReMaterializable = 1 in {
def LD_Fp032 : FpIf32<(outs RFP32:$dst), (ins), ZeroArgFP,
                [(set RFP32:$dst, fpimm0)]>;
def LD_Fp132 : FpIf32<(outs RFP32:$dst), (ins), ZeroArgFP,
                [(set RFP32:$dst, fpimm1)]>;
def LD_Fp064 : FpIf64<(outs RFP64:$dst), (ins), ZeroArgFP,
                [(set RFP64:$dst, fpimm0)]>;
def LD_Fp164 : FpIf64<(outs RFP64:$dst), (ins), ZeroArgFP,
                [(set RFP64:$dst, fpimm1)]>;
def LD_Fp080 : FpI_<(outs RFP80:$dst), (ins), ZeroArgFP,
                [(set RFP80:$dst, fpimm0)]>;
def LD_Fp180 : FpI_<(outs RFP80:$dst), (ins), ZeroArgFP,
                [(set RFP80:$dst, fpimm1)]>;
}

def LD_F0 : FPI<0xEE, RawFrm, (outs), (ins), "fldz", IIC_FLDZ>, D9;
def LD_F1 : FPI<0xE8, RawFrm, (outs), (ins), "fld1", IIC_FIST>, D9;


// Floating point compares.
def UCOM_Fpr32 : FpIf32<(outs), (ins RFP32:$lhs, RFP32:$rhs), CompareFP,
                        [(set FPSW, (trunc (X86cmp RFP32:$lhs, RFP32:$rhs)))]>;
def UCOM_Fpr64 : FpIf64<(outs), (ins RFP64:$lhs, RFP64:$rhs), CompareFP,
                        [(set FPSW, (trunc (X86cmp RFP64:$lhs, RFP64:$rhs)))]>;
def UCOM_Fpr80 : FpI_  <(outs), (ins RFP80:$lhs, RFP80:$rhs), CompareFP,
                        [(set FPSW, (trunc (X86cmp RFP80:$lhs, RFP80:$rhs)))]>;
} // Defs = [FPSW]

// CC = ST(0) cmp ST(i)
let Defs = [EFLAGS, FPSW] in {
def UCOM_FpIr32: FpIf32<(outs), (ins RFP32:$lhs, RFP32:$rhs), CompareFP,
                  [(set EFLAGS, (X86cmp RFP32:$lhs, RFP32:$rhs))]>;
def UCOM_FpIr64: FpIf64<(outs), (ins RFP64:$lhs, RFP64:$rhs), CompareFP,
                  [(set EFLAGS, (X86cmp RFP64:$lhs, RFP64:$rhs))]>;
def UCOM_FpIr80: FpI_<(outs), (ins RFP80:$lhs, RFP80:$rhs), CompareFP,
                  [(set EFLAGS, (X86cmp RFP80:$lhs, RFP80:$rhs))]>;
}

let Defs = [FPSW], Uses = [ST0] in {
def UCOM_Fr    : FPI<0xE0, AddRegFrm,    // FPSW = cmp ST(0) with ST(i)
                    (outs), (ins RST:$reg),
                    "fucom\t$reg", IIC_FUCOM>, DD;
def UCOM_FPr   : FPI<0xE8, AddRegFrm,    // FPSW = cmp ST(0) with ST(i), pop
                    (outs), (ins RST:$reg),
                    "fucomp\t$reg", IIC_FUCOM>, DD;
def UCOM_FPPr  : FPI<0xE9, RawFrm,       // cmp ST(0) with ST(1), pop, pop
                    (outs), (ins),
                    "fucompp", IIC_FUCOM>, DA;
}

let Defs = [EFLAGS, FPSW], Uses = [ST0] in {
def UCOM_FIr   : FPI<0xE8, AddRegFrm,     // CC = cmp ST(0) with ST(i)
                    (outs), (ins RST:$reg),
                    "fucomi\t$reg", IIC_FUCOMI>, DB;
def UCOM_FIPr  : FPI<0xE8, AddRegFrm,     // CC = cmp ST(0) with ST(i), pop
                    (outs), (ins RST:$reg),
                    "fucompi\t$reg", IIC_FUCOMI>, DF;
}

let Defs = [EFLAGS, FPSW] in {
def COM_FIr : FPI<0xF0, AddRegFrm, (outs), (ins RST:$reg),
                  "fcomi\t$reg", IIC_FCOMI>, DB;
def COM_FIPr : FPI<0xF0, AddRegFrm, (outs), (ins RST:$reg),
                   "fcompi\t$reg", IIC_FCOMI>, DF;
}

// Floating point flag ops.
let Defs = [AX], Uses = [FPSW] in
def FNSTSW16r : I<0xE0, RawFrm,                  // AX = fp flags
                  (outs), (ins), "fnstsw %ax",
                  [(set AX, (X86fp_stsw FPSW))], IIC_FNSTSW>, DF;

def FNSTCW16m : I<0xD9, MRM7m,                   // [mem16] = X87 control world
                  (outs), (ins i16mem:$dst), "fnstcw\t$dst",
                  [(X86fp_cwd_get16 addr:$dst)], IIC_FNSTCW>;
                  
let mayLoad = 1 in
def FLDCW16m  : I<0xD9, MRM5m,                   // X87 control world = [mem16]
                  (outs), (ins i16mem:$dst), "fldcw\t$dst", [], IIC_FLDCW>;

// FPU control instructions
let Defs = [FPSW] in
def FNINIT : I<0xE3, RawFrm, (outs), (ins), "fninit", [], IIC_FNINIT>, DB;
def FFREE : FPI<0xC0, AddRegFrm, (outs), (ins RST:$reg),
                "ffree\t$reg", IIC_FFREE>, DD;

// Clear exceptions

let Defs = [FPSW] in
def FNCLEX : I<0xE2, RawFrm, (outs), (ins), "fnclex", [], IIC_FNCLEX>, DB;

// Operandless floating-point instructions for the disassembler.
def WAIT : I<0x9B, RawFrm, (outs), (ins), "wait", [], IIC_WAIT>;

def FNOP : I<0xD0, RawFrm, (outs), (ins), "fnop", [], IIC_FNOP>, D9;
def FXAM : I<0xE5, RawFrm, (outs), (ins), "fxam", [], IIC_FXAM>, D9;
def FLDL2T : I<0xE9, RawFrm, (outs), (ins), "fldl2t", [], IIC_FLDL>, D9;
def FLDL2E : I<0xEA, RawFrm, (outs), (ins), "fldl2e", [], IIC_FLDL>, D9;
def FLDPI : I<0xEB, RawFrm, (outs), (ins), "fldpi", [], IIC_FLDL>, D9;
def FLDLG2 : I<0xEC, RawFrm, (outs), (ins), "fldlg2", [], IIC_FLDL>, D9;
def FLDLN2 : I<0xED, RawFrm, (outs), (ins), "fldln2", [], IIC_FLDL>, D9;
def F2XM1 : I<0xF0, RawFrm, (outs), (ins), "f2xm1", [], IIC_F2XM1>, D9;
def FYL2X : I<0xF1, RawFrm, (outs), (ins), "fyl2x", [], IIC_FYL2X>, D9;
def FPTAN : I<0xF2, RawFrm, (outs), (ins), "fptan", [], IIC_FPTAN>, D9;
def FPATAN : I<0xF3, RawFrm, (outs), (ins), "fpatan", [], IIC_FPATAN>, D9;
def FXTRACT : I<0xF4, RawFrm, (outs), (ins), "fxtract", [], IIC_FXTRACT>, D9;
def FPREM1 : I<0xF5, RawFrm, (outs), (ins), "fprem1", [], IIC_FPREM1>, D9;
def FDECSTP : I<0xF6, RawFrm, (outs), (ins), "fdecstp", [], IIC_FPSTP>, D9;
def FINCSTP : I<0xF7, RawFrm, (outs), (ins), "fincstp", [], IIC_FPSTP>, D9;
def FPREM : I<0xF8, RawFrm, (outs), (ins), "fprem", [], IIC_FPREM>, D9;
def FYL2XP1 : I<0xF9, RawFrm, (outs), (ins), "fyl2xp1", [], IIC_FYL2XP1>, D9;
def FSINCOS : I<0xFB, RawFrm, (outs), (ins), "fsincos", [], IIC_FSINCOS>, D9;
def FRNDINT : I<0xFC, RawFrm, (outs), (ins), "frndint", [], IIC_FRNDINT>, D9;
def FSCALE : I<0xFD, RawFrm, (outs), (ins), "fscale", [], IIC_FSCALE>, D9;
def FCOMPP : I<0xD9, RawFrm, (outs), (ins), "fcompp", [], IIC_FCOMPP>, DE;

def FXSAVE : I<0xAE, MRM0m, (outs opaque512mem:$dst), (ins),
               "fxsave\t$dst", [], IIC_FXSAVE>, TB;
def FXSAVE64 : I<0xAE, MRM0m, (outs opaque512mem:$dst), (ins),
                 "fxsaveq\t$dst", [], IIC_FXSAVE>, TB, REX_W,
                 Requires<[In64BitMode]>;
def FXRSTOR : I<0xAE, MRM1m, (outs), (ins opaque512mem:$src),
                "fxrstor\t$src", [], IIC_FXRSTOR>, TB;
def FXRSTOR64 : I<0xAE, MRM1m, (outs), (ins opaque512mem:$src),
                  "fxrstorq\t$src", [], IIC_FXRSTOR>, TB, REX_W,
                  Requires<[In64BitMode]>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// Required for RET of f32 / f64 / f80 values.
def : Pat<(X86fld addr:$src, f32), (LD_Fp32m addr:$src)>;
def : Pat<(X86fld addr:$src, f64), (LD_Fp64m addr:$src)>;
def : Pat<(X86fld addr:$src, f80), (LD_Fp80m addr:$src)>;

// Required for CALL which return f32 / f64 / f80 values.
def : Pat<(X86fst RFP32:$src, addr:$op, f32), (ST_Fp32m addr:$op, RFP32:$src)>;
def : Pat<(X86fst RFP64:$src, addr:$op, f32), (ST_Fp64m32 addr:$op, 
                                                          RFP64:$src)>;
def : Pat<(X86fst RFP64:$src, addr:$op, f64), (ST_Fp64m addr:$op, RFP64:$src)>;
def : Pat<(X86fst RFP80:$src, addr:$op, f32), (ST_Fp80m32 addr:$op, 
                                                          RFP80:$src)>;
def : Pat<(X86fst RFP80:$src, addr:$op, f64), (ST_Fp80m64 addr:$op, 
                                                          RFP80:$src)>;
def : Pat<(X86fst RFP80:$src, addr:$op, f80), (ST_FpP80m addr:$op,
                                                         RFP80:$src)>;

// Floating point constant -0.0 and -1.0
def : Pat<(f32 fpimmneg0), (CHS_Fp32 (LD_Fp032))>, Requires<[FPStackf32]>;
def : Pat<(f32 fpimmneg1), (CHS_Fp32 (LD_Fp132))>, Requires<[FPStackf32]>;
def : Pat<(f64 fpimmneg0), (CHS_Fp64 (LD_Fp064))>, Requires<[FPStackf64]>;
def : Pat<(f64 fpimmneg1), (CHS_Fp64 (LD_Fp164))>, Requires<[FPStackf64]>;
def : Pat<(f80 fpimmneg0), (CHS_Fp80 (LD_Fp080))>;
def : Pat<(f80 fpimmneg1), (CHS_Fp80 (LD_Fp180))>;

// Used to conv. i64 to f64 since there isn't a SSE version.
def : Pat<(X86fildflag addr:$src, i64), (ILD_Fp64m64 addr:$src)>;

// FP extensions map onto simple pseudo-value conversions if they are to/from
// the FP stack.
def : Pat<(f64 (fextend RFP32:$src)), (COPY_TO_REGCLASS RFP32:$src, RFP64)>,
          Requires<[FPStackf32]>;
def : Pat<(f80 (fextend RFP32:$src)), (COPY_TO_REGCLASS RFP32:$src, RFP80)>,
           Requires<[FPStackf32]>;
def : Pat<(f80 (fextend RFP64:$src)), (COPY_TO_REGCLASS RFP64:$src, RFP80)>,
           Requires<[FPStackf64]>;

// FP truncations map onto simple pseudo-value conversions if they are to/from
// the FP stack.  We have validated that only value-preserving truncations make
// it through isel.
def : Pat<(f32 (fround RFP64:$src)), (COPY_TO_REGCLASS RFP64:$src, RFP32)>,
          Requires<[FPStackf32]>;
def : Pat<(f32 (fround RFP80:$src)), (COPY_TO_REGCLASS RFP80:$src, RFP32)>,
           Requires<[FPStackf32]>;
def : Pat<(f64 (fround RFP80:$src)), (COPY_TO_REGCLASS RFP80:$src, RFP64)>,
           Requires<[FPStackf64]>;
