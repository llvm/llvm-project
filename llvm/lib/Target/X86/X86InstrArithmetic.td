//===-- X86InstrArithmetic.td - Integer Arithmetic Instrs --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the integer arithmetic instructions in the X86
// architecture.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LEA - Load Effective Address
let SchedRW = [WriteLEA] in {
let hasSideEffects = 0 in
def LEA16r   : I<0x8D, MRMSrcMem,
                 (outs GR16:$dst), (ins anymem:$src),
                 "lea{w}\t{$src|$dst}, {$dst|$src}", []>, OpSize16;
let isReMaterializable = 1 in
def LEA32r   : I<0x8D, MRMSrcMem,
                 (outs GR32:$dst), (ins anymem:$src),
                 "lea{l}\t{$src|$dst}, {$dst|$src}",
                 [(set GR32:$dst, lea32addr:$src)]>,
                 OpSize32, Requires<[Not64BitMode]>;

def LEA64_32r : I<0x8D, MRMSrcMem,
                  (outs GR32:$dst), (ins lea64_32mem:$src),
                  "lea{l}\t{$src|$dst}, {$dst|$src}",
                  [(set GR32:$dst, lea64_32addr:$src)]>,
                  OpSize32, Requires<[In64BitMode]>;

let isReMaterializable = 1 in
def LEA64r   : RI<0x8D, MRMSrcMem, (outs GR64:$dst), (ins lea64mem:$src),
                  "lea{q}\t{$src|$dst}, {$dst|$src}",
                  [(set GR64:$dst, lea64addr:$src)]>;
} // SchedRW

// Pseudo instruction for lea that prevent optimizer from eliminating
// the instruction.
let SchedRW = [WriteLEA], isPseudo = true, hasSideEffects = 1 in {
def PLEA32r   : PseudoI<(outs GR32:$dst), (ins anymem:$src), []>;
def PLEA64r   : PseudoI<(outs GR64:$dst), (ins anymem:$src), []>;
}

//===----------------------------------------------------------------------===//
// MUL/IMUL and DIV/IDIV Instructions
//
class MulDivOpR<bits<8> o, Format f, string m, X86TypeInfo t,
             X86FoldableSchedWrite sched, list<dag> p>
  : UnaryOpR<o, f, m, "$src1", t, (outs), p> {
  let SchedRW = [sched];
}

class MulDivOpM<bits<8> o, Format f, string m, X86TypeInfo t,
             X86FoldableSchedWrite sched, list<dag> p>
  : UnaryOpM<o, f, m, "$src1", t, (outs), p> {
  let SchedRW =
    [sched.Folded,
     // Memory operand.
     ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault,
     // Register reads (implicit or explicit).
     sched.ReadAfterFold, sched.ReadAfterFold];
}

multiclass Mul<bits<8> o, string m, Format RegMRM, Format MemMRM, SDPatternOperator node> {
  // AL is really implied by AX, but the registers in Defs must match the
  // SDNode results (i8, i32).
  //
  // FIXME: Used for 8-bit mul, ignore result upper 8 bits.
  // This probably ought to be moved to a def : Pat<> if the
  // syntax can be accepted.
  let Defs = [AL,EFLAGS,AX], Uses = [AL] in
  def 8r : MulDivOpR<o, RegMRM, m, Xi8, WriteIMul8,
                  [(set AL, (node AL, GR8:$src1)), (implicit EFLAGS)]>;
  let Defs = [AX,DX,EFLAGS], Uses = [AX] in
  def 16r : MulDivOpR<o, RegMRM, m, Xi16, WriteIMul16, []>, OpSize16;
  let Defs = [EAX,EDX,EFLAGS], Uses = [EAX] in
  def 32r : MulDivOpR<o, RegMRM, m, Xi32, WriteIMul32, []>, OpSize32;
  let Defs = [RAX,RDX,EFLAGS], Uses = [RAX] in
  def 64r : MulDivOpR<o, RegMRM, m, Xi64, WriteIMul64, []>;
  let Defs = [AL,EFLAGS,AX], Uses = [AL] in
  def 8m : MulDivOpM<o, MemMRM, m, Xi8, WriteIMul8,
                  [(set AL, (node AL, (loadi8 addr:$src1))), (implicit EFLAGS)]>;
  let Defs = [AX,DX,EFLAGS], Uses = [AX] in
  def 16m : MulDivOpM<o, MemMRM, m, Xi16, WriteIMul16, []>, OpSize16;
  let Defs = [EAX,EDX,EFLAGS], Uses = [EAX] in
  def 32m : MulDivOpM<o, MemMRM, m, Xi32, WriteIMul32, []>, OpSize32;
  let Defs = [RAX,RDX,EFLAGS], Uses = [RAX] in
  def 64m : MulDivOpM<o, MemMRM, m, Xi64, WriteIMul64, []>, Requires<[In64BitMode]>;
}

defm MUL : Mul<0xF7, "mul", MRM4r, MRM4m, mul>;
defm IMUL : Mul<0xF7, "imul", MRM5r, MRM5m, null_frag>;

multiclass Div<bits<8> o, string m, Format RegMRM, Format MemMRM> {
  defvar sched8 = !if(!eq(m, "div"), WriteDiv8, WriteIDiv8);
  defvar sched16 = !if(!eq(m, "div"), WriteDiv16, WriteIDiv16);
  defvar sched32 = !if(!eq(m, "div"), WriteDiv32, WriteIDiv32);
  defvar sched64 = !if(!eq(m, "div"), WriteDiv64, WriteIDiv64);
  let Defs = [AL,AH,EFLAGS], Uses = [AX] in
  def 8r  : MulDivOpR<o, RegMRM, m, Xi8, sched8, []>;
  let Defs = [AX,DX,EFLAGS], Uses = [AX,DX] in
  def 16r : MulDivOpR<o, RegMRM, m, Xi16, sched16, []>, OpSize16;
  let Defs = [EAX,EDX,EFLAGS], Uses = [EAX,EDX] in
  def 32r : MulDivOpR<o, RegMRM, m, Xi32, sched32, []>, OpSize32;
  let Defs = [RAX,RDX,EFLAGS], Uses = [RAX,RDX] in
  def 64r : MulDivOpR<o, RegMRM, m, Xi64, sched64, []>;
  let Defs = [AL,AH,EFLAGS], Uses = [AX] in
  def 8m  : MulDivOpM<o, MemMRM, m, Xi8, sched8, []>;
  let Defs = [AX,DX,EFLAGS], Uses = [AX,DX] in
  def 16m : MulDivOpM<o, MemMRM, m, Xi16, sched16, []>, OpSize16;
  let Defs = [EAX,EDX,EFLAGS], Uses = [EAX,EDX] in
  def 32m : MulDivOpM<o, MemMRM, m, Xi32, sched32, []>, OpSize32;
  let Defs = [RAX,RDX,EFLAGS], Uses = [RAX,RDX] in
  def 64m : MulDivOpM<o, MemMRM, m, Xi64, sched64, []>, Requires<[In64BitMode]>;
}
let hasSideEffects = 1 in { // so that we don't speculatively execute
defm DIV: Div<0xF7, "div", MRM6r, MRM6m>;
defm IDIV: Div<0xF7, "idiv", MRM7r, MRM7m>;
}

class IMulOpRR<X86TypeInfo t, X86FoldableSchedWrite sched>
  : BinOpRR_RF<0xAF, "imul", t, X86smul_flag>, TB {
  let Form = MRMSrcReg;
  let SchedRW = [sched];
  // X = IMUL Y, Z --> X = IMUL Z, Y
  let isCommutable = 1;
}
class IMulOpRM<X86TypeInfo t, X86FoldableSchedWrite sched>
  : BinOpRM_RF<0xAF, "imul", t, X86smul_flag>, TB {
let Form = MRMSrcMem;
let SchedRW = [sched.Folded, sched.ReadAfterFold];
}

def IMUL16rr : IMulOpRR<Xi16, WriteIMul16Reg>, OpSize16;
def IMUL32rr : IMulOpRR<Xi32, WriteIMul32Reg>, OpSize32;
def IMUL64rr : IMulOpRR<Xi64, WriteIMul64Reg>;
def IMUL16rm : IMulOpRM<Xi16, WriteIMul16Reg>, OpSize16;
def IMUL32rm : IMulOpRM<Xi32, WriteIMul32Reg>, OpSize32;
def IMUL64rm : IMulOpRM<Xi64, WriteIMul64Reg>;

class IMulOpRI8_R<X86TypeInfo t, X86FoldableSchedWrite sched>
  : BinOpRI8<0x6B, "imul", binop_ndd_args, t, MRMSrcReg,
             (outs t.RegClass:$dst)>, DefEFLAGS {
  let SchedRW = [sched];
}
class IMulOpRI_R<X86TypeInfo t, X86FoldableSchedWrite sched>
  : BinOpRI<0x69, "imul", binop_ndd_args, t, MRMSrcReg,
            (outs t.RegClass:$dst),
            [(set t.RegClass:$dst, EFLAGS, (X86smul_flag t.RegClass:$src1,
             t.ImmNoSuOperator:$src2))]>, DefEFLAGS {
  let SchedRW = [sched];
}
class IMulOpMI8_R<X86TypeInfo t, X86FoldableSchedWrite sched>
  : BinOpMI8<"imul", binop_ndd_args, t, MRMSrcMem, (outs t.RegClass:$dst)>,
    DefEFLAGS {
  let Opcode = 0x6B;
  let SchedRW = [sched.Folded];
}
class IMulOpMI_R<X86TypeInfo t, X86FoldableSchedWrite sched>
  : BinOpMI<0x69, "imul", binop_ndd_args, t, MRMSrcMem,
            (outs t.RegClass:$dst),
            [(set t.RegClass:$dst, EFLAGS, (X86smul_flag (t.LoadNode addr:$src1),
             t.ImmNoSuOperator:$src2))]>,
    DefEFLAGS {
  let SchedRW = [sched.Folded];
}
def IMUL16rri8 : IMulOpRI8_R<Xi16, WriteIMul16Imm>, OpSize16;
def IMUL32rri8 : IMulOpRI8_R<Xi32, WriteIMul32Imm>, OpSize32;
def IMUL64rri8 : IMulOpRI8_R<Xi64, WriteIMul64Imm>;
def IMUL16rri  : IMulOpRI_R<Xi16, WriteIMul16Imm>, OpSize16;
def IMUL32rri  : IMulOpRI_R<Xi32, WriteIMul32Imm>, OpSize32;
def IMUL64rri32 : IMulOpRI_R<Xi64, WriteIMul64Imm>;

def IMUL16rmi8 : IMulOpMI8_R<Xi16, WriteIMul16Imm>, OpSize16;
def IMUL32rmi8 : IMulOpMI8_R<Xi32, WriteIMul32Imm>, OpSize32;
def IMUL64rmi8 : IMulOpMI8_R<Xi64, WriteIMul64Imm>;
def IMUL16rmi  : IMulOpMI_R<Xi16, WriteIMul16Imm>, OpSize16;
def IMUL32rmi  : IMulOpMI_R<Xi32, WriteIMul32Imm>, OpSize32;
def IMUL64rmi32 : IMulOpMI_R<Xi64, WriteIMul64Imm>;

//===----------------------------------------------------------------------===//
// INC and DEC Instructions
//
class IncOpR_RF<X86TypeInfo t> : UnaryOpR_RF<0xFF, MRM0r, "inc", t, null_frag> {
  let Pattern = [(set t.RegClass:$dst, EFLAGS,
                 (X86add_flag_nocf t.RegClass:$src1, 1))];
}
class DecOpR_RF<X86TypeInfo t> : UnaryOpR_RF<0xFF, MRM1r, "dec", t, null_frag> {
  let Pattern = [(set t.RegClass:$dst, EFLAGS,
                 (X86sub_flag_nocf t.RegClass:$src1, 1))];
}
class IncOpM_M<X86TypeInfo t> : UnaryOpM_MF<0xFF, MRM0m, "inc", t, null_frag> {
  let Pattern = [(store (add (t.LoadNode addr:$src1), 1), addr:$src1),
                 (implicit EFLAGS)];
}
class DecOpM_M<X86TypeInfo t> : UnaryOpM_MF<0xFF, MRM1m, "dec", t, null_frag> {
  let Pattern = [(store (add (t.LoadNode addr:$src1), -1), addr:$src1),
                 (implicit EFLAGS)];
}
// IncDec_Alt - Instructions like "inc reg" short forms.
// Short forms only valid in 32-bit mode. Selected during MCInst lowering.
class IncDec_Alt<bits<8> o, string m, X86TypeInfo t>
  : UnaryOpR_RF<o, AddRegFrm, m, t, null_frag>, Requires<[Not64BitMode]>;

let isConvertibleToThreeAddress = 1 in {
def INC16r_alt : IncDec_Alt<0x40, "inc", Xi16>, OpSize16;
def INC32r_alt : IncDec_Alt<0x40, "inc", Xi32>, OpSize32;
def DEC16r_alt : IncDec_Alt<0x48, "dec", Xi16>, OpSize16;
def DEC32r_alt : IncDec_Alt<0x48, "dec", Xi32>, OpSize32;
def INC8r  : IncOpR_RF<Xi8>;
def INC16r : IncOpR_RF<Xi16>, OpSize16;
def INC32r : IncOpR_RF<Xi32>, OpSize32;
def INC64r : IncOpR_RF<Xi64>;
def DEC8r  : DecOpR_RF<Xi8>;
def DEC16r : DecOpR_RF<Xi16>, OpSize16;
def DEC32r : DecOpR_RF<Xi32>, OpSize32;
def DEC64r : DecOpR_RF<Xi64>;
}
let Predicates = [UseIncDec] in {
def INC8m  : IncOpM_M<Xi8>;
def INC16m : IncOpM_M<Xi16>, OpSize16;
def INC32m : IncOpM_M<Xi32>, OpSize32;
def DEC8m  : DecOpM_M<Xi8>;
def DEC16m : DecOpM_M<Xi16>, OpSize16;
def DEC32m : DecOpM_M<Xi32>, OpSize32;
}
let Predicates = [UseIncDec, In64BitMode] in {
def INC64m : IncOpM_M<Xi64>;
def DEC64m : DecOpM_M<Xi64>;
}

//===----------------------------------------------------------------------===//
// NEG and NOT Instructions
//
class NegOpR_R<X86TypeInfo t, bit ndd = 0>
  : UnaryOpR_R<0xF7, MRM3r, "neg", t, ineg, ndd>;
class NegOpR_RF<X86TypeInfo t, bit ndd = 0>
  : UnaryOpR_RF<0xF7, MRM3r, "neg", t, ineg, ndd>;
class NegOpM_M<X86TypeInfo t> : UnaryOpM_M<0xF7, MRM3m, "neg", t, null_frag>;
class NegOpM_MF<X86TypeInfo t> : UnaryOpM_MF<0xF7, MRM3m, "neg", t, ineg>;
class NegOpM_R<X86TypeInfo t> : UnaryOpM_R<0xF7, MRM3m, "neg", t, null_frag>;
class NegOpM_RF<X86TypeInfo t> : UnaryOpM_RF<0xF7, MRM3m, "neg", t, ineg>;

class NotOpR_R<X86TypeInfo t, bit ndd = 0>
  : UnaryOpR_R<0xF7, MRM2r, "not", t, not, ndd>;
class NotOpM_M<X86TypeInfo t> : UnaryOpM_M<0xF7, MRM2m, "not", t, not>;
class NotOpM_R<X86TypeInfo t> : UnaryOpM_R<0xF7, MRM2m, "not", t, not>;

let Predicates = [NoNDD] in {
def NEG8r  : NegOpR_RF<Xi8>;
def NEG16r : NegOpR_RF<Xi16>, OpSize16;
def NEG32r : NegOpR_RF<Xi32>, OpSize32;
def NEG64r : NegOpR_RF<Xi64>;
def NOT8r  : NotOpR_R<Xi8>;
def NOT16r : NotOpR_R<Xi16>, OpSize16;
def NOT32r : NotOpR_R<Xi32>, OpSize32;
def NOT64r : NotOpR_R<Xi64>;
}

let Predicates = [HasNDD, In64BitMode] in {
def NEG8r_ND  : NegOpR_RF<Xi8, 1>;
def NEG16r_ND : NegOpR_RF<Xi16, 1>, PD;
def NEG32r_ND : NegOpR_RF<Xi32, 1>;
def NEG64r_ND : NegOpR_RF<Xi64, 1>;

def NOT8r_ND  : NotOpR_R<Xi8, 1>;
def NOT16r_ND : NotOpR_R<Xi16, 1>, PD;
def NOT32r_ND : NotOpR_R<Xi32, 1>;
def NOT64r_ND : NotOpR_R<Xi64, 1>;

def NEG8r_NF_ND  : NegOpR_R<Xi8, 1>, EVEX_NF;
def NEG16r_NF_ND : NegOpR_R<Xi16, 1>, EVEX_NF, PD;
def NEG32r_NF_ND : NegOpR_R<Xi32, 1>, EVEX_NF;
def NEG64r_NF_ND : NegOpR_R<Xi64, 1>, EVEX_NF;
}

def NEG8m  : NegOpM_MF<Xi8>;
def NEG16m : NegOpM_MF<Xi16>, OpSize16;
def NEG32m : NegOpM_MF<Xi32>, OpSize32;
def NEG64m : NegOpM_MF<Xi64>, Requires<[In64BitMode]>;

let Predicates = [HasNDD, In64BitMode] in {
def NEG8m_ND  : NegOpM_RF<Xi8>;
def NEG16m_ND : NegOpM_RF<Xi16>, PD;
def NEG32m_ND : NegOpM_RF<Xi32>;
def NEG64m_ND : NegOpM_RF<Xi64>;

def NEG8m_NF_ND  : NegOpM_R<Xi8>, EVEX_NF;
def NEG16m_NF_ND : NegOpM_R<Xi16>, EVEX_NF, PD;
def NEG32m_NF_ND : NegOpM_R<Xi32>, EVEX_NF;
def NEG64m_NF_ND : NegOpM_R<Xi64>, EVEX_NF;
}

def NOT8m  : NotOpM_M<Xi8>;
def NOT16m : NotOpM_M<Xi16>, OpSize16;
def NOT32m : NotOpM_M<Xi32>, OpSize32;
def NOT64m : NotOpM_M<Xi64>, Requires<[In64BitMode]>;

let Predicates = [HasNDD, In64BitMode] in {
def NOT8m_ND  : NotOpM_R<Xi8>;
def NOT16m_ND : NotOpM_R<Xi16>, PD;
def NOT32m_ND : NotOpM_R<Xi32>;
def NOT64m_ND : NotOpM_R<Xi64>;
}

let Predicates = [In64BitMode], Pattern = [(null_frag)] in {
def NEG8r_NF  : NegOpR_R<Xi8>, NF;
def NEG16r_NF : NegOpR_R<Xi16>, NF, PD;
def NEG32r_NF : NegOpR_R<Xi32>, NF;
def NEG64r_NF : NegOpR_R<Xi64>, NF;
def NEG8m_NF  : NegOpM_M<Xi8>, NF;
def NEG16m_NF : NegOpM_M<Xi16>, NF, PD;
def NEG32m_NF : NegOpM_M<Xi32>, NF;
def NEG64m_NF : NegOpM_M<Xi64>, NF;

def NEG8r_EVEX  : NegOpR_RF<Xi8>, PL;
def NEG16r_EVEX : NegOpR_RF<Xi16>, PL, PD;
def NEG32r_EVEX : NegOpR_RF<Xi32>, PL;
def NEG64r_EVEX : NegOpR_RF<Xi64>, PL;

def NOT8r_EVEX  : NotOpR_R<Xi8>, PL;
def NOT16r_EVEX : NotOpR_R<Xi16>, PL, PD;
def NOT32r_EVEX : NotOpR_R<Xi32>, PL;
def NOT64r_EVEX : NotOpR_R<Xi64>, PL;

def NEG8m_EVEX  : NegOpM_MF<Xi8>, PL;
def NEG16m_EVEX : NegOpM_MF<Xi16>, PL, PD;
def NEG32m_EVEX : NegOpM_MF<Xi32>, PL;
def NEG64m_EVEX : NegOpM_MF<Xi64>, PL;

def NOT8m_EVEX  : NotOpM_M<Xi8>, PL;
def NOT16m_EVEX : NotOpM_M<Xi16>, PL, PD;
def NOT32m_EVEX : NotOpM_M<Xi32>, PL;
def NOT64m_EVEX : NotOpM_M<Xi64>, PL;
}

/// ArithBinOp_RF - This is an arithmetic binary operator where the pattern is
/// defined with "(set GPR:$dst, EFLAGS, (...".
///
/// It would be nice to get rid of the second and third argument here, but
/// tblgen can't handle dependent type references aggressively enough: PR8330
multiclass ArithBinOp_RF<bits<8> BaseOpc, bits<8> BaseOpc2, bits<8> BaseOpc4,
                         string mnemonic, Format RegMRM, Format MemMRM,
                         SDNode opnodeflag, SDNode opnode,
                         bit CommutableRR, bit ConvertibleToThreeAddress,
                         bit ConvertibleToThreeAddressRR> {
  let isCommutable = CommutableRR,
      isConvertibleToThreeAddress = ConvertibleToThreeAddressRR in {
    let Predicates = [NoNDD] in {
      def NAME#8rr  : BinOpRR_RF<BaseOpc, mnemonic, Xi8 , opnodeflag>;
      def NAME#16rr : BinOpRR_RF<BaseOpc, mnemonic, Xi16, opnodeflag>, OpSize16;
      def NAME#32rr : BinOpRR_RF<BaseOpc, mnemonic, Xi32, opnodeflag>, OpSize32;
      def NAME#64rr : BinOpRR_RF<BaseOpc, mnemonic, Xi64, opnodeflag>;
    }
    let Predicates = [HasNDD, In64BitMode] in {
      def NAME#8rr_ND  : BinOpRR_RF<BaseOpc, mnemonic, Xi8 , opnodeflag, 1>;
      def NAME#16rr_ND : BinOpRR_RF<BaseOpc, mnemonic, Xi16, opnodeflag, 1>, PD;
      def NAME#32rr_ND : BinOpRR_RF<BaseOpc, mnemonic, Xi32, opnodeflag, 1>;
      def NAME#64rr_ND : BinOpRR_RF<BaseOpc, mnemonic, Xi64, opnodeflag, 1>;
      def NAME#8rr_NF_ND  : BinOpRR_R<BaseOpc, mnemonic, Xi8, 1>, EVEX_NF;
      def NAME#16rr_NF_ND : BinOpRR_R<BaseOpc, mnemonic, Xi16, 1>, EVEX_NF, PD;
      def NAME#32rr_NF_ND : BinOpRR_R<BaseOpc, mnemonic, Xi32, 1>, EVEX_NF;
      def NAME#64rr_NF_ND : BinOpRR_R<BaseOpc, mnemonic, Xi64, 1>, EVEX_NF;
    }
    let Predicates = [In64BitMode] in {
      def NAME#8rr_NF  : BinOpRR_R<BaseOpc, mnemonic, Xi8>, NF;
      def NAME#16rr_NF : BinOpRR_R<BaseOpc, mnemonic, Xi16>, NF, PD;
      def NAME#32rr_NF : BinOpRR_R<BaseOpc, mnemonic, Xi32>, NF;
      def NAME#64rr_NF : BinOpRR_R<BaseOpc, mnemonic, Xi64>, NF;
      def NAME#8rr_EVEX  : BinOpRR_RF<BaseOpc, mnemonic, Xi8 , null_frag>, PL;
      def NAME#16rr_EVEX : BinOpRR_RF<BaseOpc, mnemonic, Xi16, null_frag>, PL, PD;
      def NAME#32rr_EVEX : BinOpRR_RF<BaseOpc, mnemonic, Xi32, null_frag>, PL;
      def NAME#64rr_EVEX : BinOpRR_RF<BaseOpc, mnemonic, Xi64, null_frag>, PL;
    }
  }

    def NAME#8rr_REV  : BinOpRR_RF_Rev<BaseOpc2, mnemonic, Xi8>;
    def NAME#16rr_REV : BinOpRR_RF_Rev<BaseOpc2, mnemonic, Xi16>, OpSize16;
    def NAME#32rr_REV : BinOpRR_RF_Rev<BaseOpc2, mnemonic, Xi32>, OpSize32;
    def NAME#64rr_REV : BinOpRR_RF_Rev<BaseOpc2, mnemonic, Xi64>;
    let Predicates = [In64BitMode] in {
      def NAME#8rr_EVEX_REV  : BinOpRR_RF_Rev<BaseOpc2, mnemonic, Xi8>, PL;
      def NAME#16rr_EVEX_REV : BinOpRR_RF_Rev<BaseOpc2, mnemonic, Xi16>, PL, PD;
      def NAME#32rr_EVEX_REV : BinOpRR_RF_Rev<BaseOpc2, mnemonic, Xi32>, PL;
      def NAME#64rr_EVEX_REV : BinOpRR_RF_Rev<BaseOpc2, mnemonic, Xi64>, PL;
      def NAME#8rr_ND_REV  : BinOpRR_RF_Rev<BaseOpc2, mnemonic, Xi8, 1>;
      def NAME#16rr_ND_REV : BinOpRR_RF_Rev<BaseOpc2, mnemonic, Xi16, 1>, PD;
      def NAME#32rr_ND_REV : BinOpRR_RF_Rev<BaseOpc2, mnemonic, Xi32, 1>;
      def NAME#64rr_ND_REV : BinOpRR_RF_Rev<BaseOpc2, mnemonic, Xi64, 1>;
      def NAME#8rr_NF_REV  : BinOpRR_R_Rev<BaseOpc2, mnemonic, Xi8>, NF;
      def NAME#16rr_NF_REV : BinOpRR_R_Rev<BaseOpc2, mnemonic, Xi16>, NF, PD;
      def NAME#32rr_NF_REV : BinOpRR_R_Rev<BaseOpc2, mnemonic, Xi32>, NF;
      def NAME#64rr_NF_REV : BinOpRR_R_Rev<BaseOpc2, mnemonic, Xi64>, NF;
      def NAME#8rr_NF_ND_REV  : BinOpRR_R_Rev<BaseOpc2, mnemonic, Xi8, 1>, EVEX_NF;
      def NAME#16rr_NF_ND_REV : BinOpRR_R_Rev<BaseOpc2, mnemonic, Xi16, 1>, EVEX_NF, PD;
      def NAME#32rr_NF_ND_REV : BinOpRR_R_Rev<BaseOpc2, mnemonic, Xi32, 1>, EVEX_NF;
      def NAME#64rr_NF_ND_REV : BinOpRR_R_Rev<BaseOpc2, mnemonic, Xi64, 1>, EVEX_NF;
    }

    let Predicates = [NoNDD] in {
      def NAME#8rm   : BinOpRM_RF<BaseOpc2, mnemonic, Xi8 , opnodeflag>;
      def NAME#16rm  : BinOpRM_RF<BaseOpc2, mnemonic, Xi16, opnodeflag>, OpSize16;
      def NAME#32rm  : BinOpRM_RF<BaseOpc2, mnemonic, Xi32, opnodeflag>, OpSize32;
      def NAME#64rm  : BinOpRM_RF<BaseOpc2, mnemonic, Xi64, opnodeflag>;
    }
    let Predicates = [HasNDD, In64BitMode] in {
      def NAME#8rm_ND  : BinOpRM_RF<BaseOpc2, mnemonic, Xi8 , opnodeflag, 1>;
      def NAME#16rm_ND : BinOpRM_RF<BaseOpc2, mnemonic, Xi16, opnodeflag, 1>, PD;
      def NAME#32rm_ND : BinOpRM_RF<BaseOpc2, mnemonic, Xi32, opnodeflag, 1>;
      def NAME#64rm_ND : BinOpRM_RF<BaseOpc2, mnemonic, Xi64, opnodeflag, 1>;
      def NAME#8rm_NF_ND  : BinOpRM_R<BaseOpc2, mnemonic, Xi8, 1>, EVEX_NF;
      def NAME#16rm_NF_ND : BinOpRM_R<BaseOpc2, mnemonic, Xi16, 1>, EVEX_NF, PD;
      def NAME#32rm_NF_ND : BinOpRM_R<BaseOpc2, mnemonic, Xi32, 1>, EVEX_NF;
      def NAME#64rm_NF_ND : BinOpRM_R<BaseOpc2, mnemonic, Xi64, 1>, EVEX_NF;
    }
    let Predicates = [In64BitMode] in {
      def NAME#8rm_NF  : BinOpRM_R<BaseOpc2, mnemonic, Xi8>, NF;
      def NAME#16rm_NF : BinOpRM_R<BaseOpc2, mnemonic, Xi16>, NF, PD;
      def NAME#32rm_NF : BinOpRM_R<BaseOpc2, mnemonic, Xi32>, NF;
      def NAME#64rm_NF : BinOpRM_R<BaseOpc2, mnemonic, Xi64>, NF;
      def NAME#8rm_EVEX  : BinOpRM_RF<BaseOpc2, mnemonic, Xi8 , null_frag>, PL;
      def NAME#16rm_EVEX : BinOpRM_RF<BaseOpc2, mnemonic, Xi16, null_frag>, PL, PD;
      def NAME#32rm_EVEX : BinOpRM_RF<BaseOpc2, mnemonic, Xi32, null_frag>, PL;
      def NAME#64rm_EVEX : BinOpRM_RF<BaseOpc2, mnemonic, Xi64, null_frag>, PL;
    }

    let isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
      let Predicates = [NoNDD] in {
        // NOTE: These are order specific, we want the ri8 forms to be listed
        // first so that they are slightly preferred to the ri forms.
        def NAME#16ri8 : BinOpRI8_RF<0x83, mnemonic, Xi16, RegMRM>, OpSize16;
        def NAME#32ri8 : BinOpRI8_RF<0x83, mnemonic, Xi32, RegMRM>, OpSize32;
        def NAME#64ri8 : BinOpRI8_RF<0x83, mnemonic, Xi64, RegMRM>;
        def NAME#8ri   : BinOpRI_RF<0x80, mnemonic, Xi8 , opnodeflag, RegMRM>;
        def NAME#16ri  : BinOpRI_RF<0x81, mnemonic, Xi16, opnodeflag, RegMRM>, OpSize16;
        def NAME#32ri  : BinOpRI_RF<0x81, mnemonic, Xi32, opnodeflag, RegMRM>, OpSize32;
        def NAME#64ri32: BinOpRI_RF<0x81, mnemonic, Xi64, opnodeflag, RegMRM>;
      }
      let Predicates = [HasNDD, In64BitMode] in {
        def NAME#16ri8_ND : BinOpRI8_RF<0x83, mnemonic, Xi16, RegMRM, 1>, PD;
        def NAME#32ri8_ND : BinOpRI8_RF<0x83, mnemonic, Xi32, RegMRM, 1>;
        def NAME#64ri8_ND : BinOpRI8_RF<0x83, mnemonic, Xi64, RegMRM, 1>;
        def NAME#8ri_ND   : BinOpRI_RF<0x80, mnemonic, Xi8 , opnodeflag, RegMRM, 1>;
        def NAME#16ri_ND  : BinOpRI_RF<0x81, mnemonic, Xi16, opnodeflag, RegMRM, 1>, PD;
        def NAME#32ri_ND  : BinOpRI_RF<0x81, mnemonic, Xi32, opnodeflag, RegMRM, 1>;
        def NAME#64ri32_ND: BinOpRI_RF<0x81, mnemonic, Xi64, opnodeflag, RegMRM, 1>;
        def NAME#16ri8_NF_ND : BinOpRI8_R<0x83, mnemonic, Xi16, RegMRM, 1>, EVEX_NF, PD;
        def NAME#32ri8_NF_ND : BinOpRI8_R<0x83, mnemonic, Xi32, RegMRM, 1>, EVEX_NF;
        def NAME#64ri8_NF_ND : BinOpRI8_R<0x83, mnemonic, Xi64, RegMRM, 1>, EVEX_NF;
        def NAME#8ri_NF_ND  : BinOpRI_R<0x80, mnemonic, Xi8, RegMRM, 1>, EVEX_NF;
        def NAME#16ri_NF_ND : BinOpRI_R<0x81, mnemonic, Xi16, RegMRM, 1>, EVEX_NF, PD;
        def NAME#32ri_NF_ND : BinOpRI_R<0x81, mnemonic, Xi32, RegMRM, 1>, EVEX_NF;
        def NAME#64ri32_NF_ND : BinOpRI_R<0x81, mnemonic, Xi64, RegMRM, 1>, EVEX_NF;
      }
      let Predicates = [In64BitMode] in {
        def NAME#16ri8_NF : BinOpRI8_R<0x83, mnemonic, Xi16, RegMRM>, NF, PD;
        def NAME#32ri8_NF : BinOpRI8_R<0x83, mnemonic, Xi32, RegMRM>, NF;
        def NAME#64ri8_NF : BinOpRI8_R<0x83, mnemonic, Xi64, RegMRM>, NF;
        def NAME#8ri_NF  : BinOpRI_R<0x80, mnemonic, Xi8, RegMRM>, NF;
        def NAME#16ri_NF : BinOpRI_R<0x81, mnemonic, Xi16, RegMRM>, NF, PD;
        def NAME#32ri_NF : BinOpRI_R<0x81, mnemonic, Xi32, RegMRM>, NF;
        def NAME#64ri32_NF : BinOpRI_R<0x81, mnemonic, Xi64, RegMRM>, NF;
        def NAME#16ri8_EVEX : BinOpRI8_RF<0x83, mnemonic, Xi16, RegMRM>, PL, PD;
        def NAME#32ri8_EVEX : BinOpRI8_RF<0x83, mnemonic, Xi32, RegMRM>, PL;
        def NAME#64ri8_EVEX : BinOpRI8_RF<0x83, mnemonic, Xi64, RegMRM>, PL;
        def NAME#8ri_EVEX   : BinOpRI_RF<0x80, mnemonic, Xi8 , null_frag, RegMRM>, PL;
        def NAME#16ri_EVEX  : BinOpRI_RF<0x81, mnemonic, Xi16, null_frag, RegMRM>, PL, PD;
        def NAME#32ri_EVEX  : BinOpRI_RF<0x81, mnemonic, Xi32, null_frag, RegMRM>, PL;
        def NAME#64ri32_EVEX: BinOpRI_RF<0x81, mnemonic, Xi64, null_frag, RegMRM>, PL;
      }
    }

    def NAME#8mr    : BinOpMR_MF<BaseOpc, mnemonic, Xi8 , opnode>;
    def NAME#16mr   : BinOpMR_MF<BaseOpc, mnemonic, Xi16, opnode>, OpSize16;
    def NAME#32mr   : BinOpMR_MF<BaseOpc, mnemonic, Xi32, opnode>, OpSize32;
    def NAME#64mr   : BinOpMR_MF<BaseOpc, mnemonic, Xi64, opnode>;
    let Predicates = [HasNDD, In64BitMode] in {
    def NAME#8mr_ND    : BinOpMR_RF<BaseOpc, mnemonic, Xi8 , opnode>;
    def NAME#16mr_ND   : BinOpMR_RF<BaseOpc, mnemonic, Xi16, opnode>, PD;
    def NAME#32mr_ND   : BinOpMR_RF<BaseOpc, mnemonic, Xi32, opnode>;
    def NAME#64mr_ND   : BinOpMR_RF<BaseOpc, mnemonic, Xi64, opnode>;
    def NAME#8mr_NF_ND    : BinOpMR_R<BaseOpc, mnemonic, Xi8>, EVEX_NF;
    def NAME#16mr_NF_ND   : BinOpMR_R<BaseOpc, mnemonic, Xi16>, EVEX_NF, PD;
    def NAME#32mr_NF_ND   : BinOpMR_R<BaseOpc, mnemonic, Xi32>, EVEX_NF;
    def NAME#64mr_NF_ND   : BinOpMR_R<BaseOpc, mnemonic, Xi64>, EVEX_NF;
  }
  let Predicates = [In64BitMode] in {
    def NAME#8mr_NF    : BinOpMR_M<BaseOpc, mnemonic, Xi8>, NF;
    def NAME#16mr_NF   : BinOpMR_M<BaseOpc, mnemonic, Xi16>, NF, PD;
    def NAME#32mr_NF   : BinOpMR_M<BaseOpc, mnemonic, Xi32>, NF;
    def NAME#64mr_NF   : BinOpMR_M<BaseOpc, mnemonic, Xi64>, NF;
    def NAME#8mr_EVEX    : BinOpMR_MF<BaseOpc, mnemonic, Xi8 , null_frag>, PL;
    def NAME#16mr_EVEX   : BinOpMR_MF<BaseOpc, mnemonic, Xi16, null_frag>, PL, PD;
    def NAME#32mr_EVEX   : BinOpMR_MF<BaseOpc, mnemonic, Xi32, null_frag>, PL;
    def NAME#64mr_EVEX   : BinOpMR_MF<BaseOpc, mnemonic, Xi64, null_frag>, PL;
  }

  // NOTE: These are order specific, we want the mi8 forms to be listed
  // first so that they are slightly preferred to the mi forms.
  def NAME#16mi8  : BinOpMI8_MF<mnemonic, Xi16, MemMRM>, OpSize16;
  def NAME#32mi8  : BinOpMI8_MF<mnemonic, Xi32, MemMRM>, OpSize32;
  let Predicates = [In64BitMode] in
    def NAME#64mi8  : BinOpMI8_MF<mnemonic, Xi64, MemMRM>;
  def NAME#8mi    : BinOpMI_MF<0x80, mnemonic, Xi8 , opnode, MemMRM>;
  def NAME#16mi   : BinOpMI_MF<0x81, mnemonic, Xi16, opnode, MemMRM>, OpSize16;
  def NAME#32mi   : BinOpMI_MF<0x81, mnemonic, Xi32, opnode, MemMRM>, OpSize32;
  let Predicates = [In64BitMode] in
    def NAME#64mi32 : BinOpMI_MF<0x81, mnemonic, Xi64, opnode, MemMRM>;
  let Predicates = [HasNDD, In64BitMode] in {
    def NAME#16mi8_ND  : BinOpMI8_RF<mnemonic, Xi16, MemMRM>, PD;
    def NAME#32mi8_ND  : BinOpMI8_RF<mnemonic, Xi32, MemMRM>;
    def NAME#64mi8_ND  : BinOpMI8_RF<mnemonic, Xi64, MemMRM>;
    def NAME#8mi_ND    : BinOpMI_RF<0x80, mnemonic, Xi8 , opnode, MemMRM>;
    def NAME#16mi_ND   : BinOpMI_RF<0x81, mnemonic, Xi16, opnode, MemMRM>, PD;
    def NAME#32mi_ND   : BinOpMI_RF<0x81, mnemonic, Xi32, opnode, MemMRM>;
    def NAME#64mi32_ND : BinOpMI_RF<0x81, mnemonic, Xi64, opnode, MemMRM>;
    def NAME#16mi8_NF_ND  : BinOpMI8_R<mnemonic, Xi16, MemMRM>, NF, PD;
    def NAME#32mi8_NF_ND  : BinOpMI8_R<mnemonic, Xi32, MemMRM>, NF;
    def NAME#64mi8_NF_ND  : BinOpMI8_R<mnemonic, Xi64, MemMRM>, NF;
    def NAME#8mi_NF_ND    : BinOpMI_R<0x80, mnemonic, Xi8, MemMRM>, NF;
    def NAME#16mi_NF_ND   : BinOpMI_R<0x81, mnemonic, Xi16, MemMRM>, NF, PD;
    def NAME#32mi_NF_ND   : BinOpMI_R<0x81, mnemonic, Xi32, MemMRM>, NF;
    def NAME#64mi32_NF_ND : BinOpMI_R<0x81, mnemonic, Xi64, MemMRM>, NF;
  }
  let Predicates = [In64BitMode] in {
    def NAME#16mi8_NF  : BinOpMI8_M<mnemonic, Xi16, MemMRM>, NF, PD;
    def NAME#32mi8_NF  : BinOpMI8_M<mnemonic, Xi32, MemMRM>, NF;
    def NAME#64mi8_NF  : BinOpMI8_M<mnemonic, Xi64, MemMRM>, NF;
    def NAME#8mi_NF    : BinOpMI_M<0x80, mnemonic, Xi8, MemMRM>, NF;
    def NAME#16mi_NF   : BinOpMI_M<0x81, mnemonic, Xi16, MemMRM>, NF, PD;
    def NAME#32mi_NF   : BinOpMI_M<0x81, mnemonic, Xi32, MemMRM>, NF;
    def NAME#64mi32_NF : BinOpMI_M<0x81, mnemonic, Xi64, MemMRM>, NF;
    def NAME#16mi8_EVEX  : BinOpMI8_MF<mnemonic, Xi16, MemMRM>, PL, PD;
    def NAME#32mi8_EVEX  : BinOpMI8_MF<mnemonic, Xi32, MemMRM>, PL;
    def NAME#64mi8_EVEX  : BinOpMI8_MF<mnemonic, Xi64, MemMRM>, PL;
    def NAME#8mi_EVEX    : BinOpMI_MF<0x80, mnemonic, Xi8 , null_frag, MemMRM>, PL;
    def NAME#16mi_EVEX   : BinOpMI_MF<0x81, mnemonic, Xi16, null_frag, MemMRM>, PL, PD;
    def NAME#32mi_EVEX   : BinOpMI_MF<0x81, mnemonic, Xi32, null_frag, MemMRM>, PL;
    def NAME#64mi32_EVEX : BinOpMI_MF<0x81, mnemonic, Xi64, null_frag, MemMRM>, PL;
  }

  // These are for the disassembler since 0x82 opcode behaves like 0x80, but
  // not in 64-bit mode.
  let Predicates = [Not64BitMode] in {
  def NAME#8ri8 : BinOpRI8_RF<0x82, mnemonic, Xi8, RegMRM>, DisassembleOnly;
  def NAME#8mi8 : BinOpMI8_MF<mnemonic, Xi8, MemMRM>, DisassembleOnly;
  }

  def NAME#8i8 : BinOpAI_AF<BaseOpc4, mnemonic, Xi8 , AL,
                            "{$src, %al|al, $src}">;
  def NAME#16i16 : BinOpAI_AF<BaseOpc4, mnemonic, Xi16, AX,
                              "{$src, %ax|ax, $src}">, OpSize16;
  def NAME#32i32 : BinOpAI_AF<BaseOpc4, mnemonic, Xi32, EAX,
                              "{$src, %eax|eax, $src}">, OpSize32;
  def NAME#64i32 : BinOpAI_AF<BaseOpc4, mnemonic, Xi64, RAX,
                              "{$src, %rax|rax, $src}">;
}

/// ArithBinOp_RFF - This is an arithmetic binary operator where the pattern is
/// defined with "(set GPR:$dst, EFLAGS, (node LHS, RHS, EFLAGS))" like ADC and
/// SBB.
///
/// It would be nice to get rid of the second and third argument here, but
/// tblgen can't handle dependent type references aggressively enough: PR8330
multiclass ArithBinOp_RFF<bits<8> BaseOpc, bits<8> BaseOpc2, bits<8> BaseOpc4,
                          string mnemonic, Format RegMRM, Format MemMRM,
                          SDNode opnode, bit CommutableRR,
                           bit ConvertibleToThreeAddress> {
  let isCommutable = CommutableRR in {
    let Predicates = [NoNDD] in {
      def NAME#8rr  : BinOpRRF_RF<BaseOpc, mnemonic, Xi8 , opnode>;
      let isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
        def NAME#16rr : BinOpRRF_RF<BaseOpc, mnemonic, Xi16, opnode>, OpSize16;
        def NAME#32rr : BinOpRRF_RF<BaseOpc, mnemonic, Xi32, opnode>, OpSize32;
        def NAME#64rr : BinOpRRF_RF<BaseOpc, mnemonic, Xi64, opnode>;
      }
    }
    let Predicates = [HasNDD, In64BitMode] in {
      def NAME#8rr_ND  : BinOpRRF_RF<BaseOpc, mnemonic, Xi8 , opnode, 1>;
      let isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
        def NAME#16rr_ND : BinOpRRF_RF<BaseOpc, mnemonic, Xi16, opnode, 1>, PD;
        def NAME#32rr_ND : BinOpRRF_RF<BaseOpc, mnemonic, Xi32, opnode, 1>;
        def NAME#64rr_ND : BinOpRRF_RF<BaseOpc, mnemonic, Xi64, opnode, 1>;
      }
    }
  } // isCommutable

  let Predicates = [In64BitMode] in {
    def NAME#8rr_EVEX  : BinOpRRF_RF<BaseOpc, mnemonic, Xi8 , null_frag>, PL;
    def NAME#16rr_EVEX : BinOpRRF_RF<BaseOpc, mnemonic, Xi16, null_frag>, PL, PD;
    def NAME#32rr_EVEX : BinOpRRF_RF<BaseOpc, mnemonic, Xi32, null_frag>, PL;
    def NAME#64rr_EVEX : BinOpRRF_RF<BaseOpc, mnemonic, Xi64, null_frag>, PL;
  }

  def NAME#8rr_REV  : BinOpRRF_RF_Rev<BaseOpc2, mnemonic, Xi8>;
  def NAME#16rr_REV : BinOpRRF_RF_Rev<BaseOpc2, mnemonic, Xi16>, OpSize16;
  def NAME#32rr_REV : BinOpRRF_RF_Rev<BaseOpc2, mnemonic, Xi32>, OpSize32;
  def NAME#64rr_REV : BinOpRRF_RF_Rev<BaseOpc2, mnemonic, Xi64>;
  let Predicates = [In64BitMode] in {
    def NAME#8rr_ND_REV  : BinOpRRF_RF_Rev<BaseOpc2, mnemonic, Xi8, 1>;
    def NAME#16rr_ND_REV : BinOpRRF_RF_Rev<BaseOpc2, mnemonic, Xi16, 1>, PD;
    def NAME#32rr_ND_REV : BinOpRRF_RF_Rev<BaseOpc2, mnemonic, Xi32, 1>;
    def NAME#64rr_ND_REV : BinOpRRF_RF_Rev<BaseOpc2, mnemonic, Xi64, 1>;
    def NAME#8rr_EVEX_REV  : BinOpRRF_RF_Rev<BaseOpc2, mnemonic, Xi8>, PL;
    def NAME#16rr_EVEX_REV : BinOpRRF_RF_Rev<BaseOpc2, mnemonic, Xi16>, PL, PD;
    def NAME#32rr_EVEX_REV : BinOpRRF_RF_Rev<BaseOpc2, mnemonic, Xi32>, PL;
    def NAME#64rr_EVEX_REV : BinOpRRF_RF_Rev<BaseOpc2, mnemonic, Xi64>, PL;
  }

  let Predicates = [NoNDD] in {
    def NAME#8rm   : BinOpRMF_RF<BaseOpc2, mnemonic, Xi8 , opnode>;
    def NAME#16rm  : BinOpRMF_RF<BaseOpc2, mnemonic, Xi16, opnode>, OpSize16;
    def NAME#32rm  : BinOpRMF_RF<BaseOpc2, mnemonic, Xi32, opnode>, OpSize32;
    def NAME#64rm  : BinOpRMF_RF<BaseOpc2, mnemonic, Xi64, opnode>;
  }
  let Predicates = [HasNDD, In64BitMode] in {
    def NAME#8rm_ND   : BinOpRMF_RF<BaseOpc2, mnemonic, Xi8 , opnode, 1>;
    def NAME#16rm_ND  : BinOpRMF_RF<BaseOpc2, mnemonic, Xi16, opnode, 1>, PD;
    def NAME#32rm_ND  : BinOpRMF_RF<BaseOpc2, mnemonic, Xi32, opnode, 1>;
    def NAME#64rm_ND  : BinOpRMF_RF<BaseOpc2, mnemonic, Xi64, opnode, 1>;
  }
  let Predicates = [In64BitMode] in {
    def NAME#8rm_EVEX   : BinOpRMF_RF<BaseOpc2, mnemonic, Xi8 , opnode>, PL;
    def NAME#16rm_EVEX  : BinOpRMF_RF<BaseOpc2, mnemonic, Xi16, opnode>, PL, PD;
    def NAME#32rm_EVEX  : BinOpRMF_RF<BaseOpc2, mnemonic, Xi32, opnode>, PL;
    def NAME#64rm_EVEX  : BinOpRMF_RF<BaseOpc2, mnemonic, Xi64, opnode>, PL;
  }

  let Predicates = [NoNDD] in {
    def NAME#8ri   : BinOpRIF_RF<0x80, mnemonic, Xi8 , opnode, RegMRM>;
    let isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
      // NOTE: These are order specific, we want the ri8 forms to be listed
      // first so that they are slightly preferred to the ri forms.
      def NAME#16ri8 : BinOpRI8F_RF<0x83, mnemonic, Xi16, RegMRM>, OpSize16;
      def NAME#32ri8 : BinOpRI8F_RF<0x83, mnemonic, Xi32, RegMRM>, OpSize32;
      def NAME#64ri8 : BinOpRI8F_RF<0x83, mnemonic, Xi64, RegMRM>;

      def NAME#16ri  : BinOpRIF_RF<0x81, mnemonic, Xi16, opnode, RegMRM>, OpSize16;
      def NAME#32ri  : BinOpRIF_RF<0x81, mnemonic, Xi32, opnode, RegMRM>, OpSize32;
      def NAME#64ri32: BinOpRIF_RF<0x81, mnemonic, Xi64, opnode, RegMRM>;
    }
  }

  let Predicates = [HasNDD, In64BitMode] in {
    def NAME#8ri_ND   : BinOpRIF_RF<0x80, mnemonic, Xi8 , opnode, RegMRM, 1>;
    let isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
      def NAME#16ri8_ND : BinOpRI8F_RF<0x83, mnemonic, Xi16, RegMRM, 1>, PD;
      def NAME#32ri8_ND : BinOpRI8F_RF<0x83, mnemonic, Xi32, RegMRM, 1>;
      def NAME#64ri8_ND : BinOpRI8F_RF<0x83, mnemonic, Xi64, RegMRM, 1>;
      def NAME#16ri_ND  : BinOpRIF_RF<0x81, mnemonic, Xi16, opnode, RegMRM, 1>, PD;
      def NAME#32ri_ND  : BinOpRIF_RF<0x81, mnemonic, Xi32, opnode, RegMRM, 1>;
      def NAME#64ri32_ND: BinOpRIF_RF<0x81, mnemonic, Xi64, opnode, RegMRM, 1>;
    }
  }
  let Predicates = [In64BitMode] in {
    def NAME#8ri_EVEX   : BinOpRIF_RF<0x80, mnemonic, Xi8 , opnode, RegMRM>, PL;
    def NAME#16ri8_EVEX : BinOpRI8F_RF<0x83, mnemonic, Xi16, RegMRM>, PL, PD;
    def NAME#32ri8_EVEX : BinOpRI8F_RF<0x83, mnemonic, Xi32, RegMRM>, PL;
    def NAME#64ri8_EVEX : BinOpRI8F_RF<0x83, mnemonic, Xi64, RegMRM>, PL;
    def NAME#16ri_EVEX  : BinOpRIF_RF<0x81, mnemonic, Xi16, opnode, RegMRM>, PL, PD;
    def NAME#32ri_EVEX  : BinOpRIF_RF<0x81, mnemonic, Xi32, opnode, RegMRM>, PL;
    def NAME#64ri32_EVEX: BinOpRIF_RF<0x81, mnemonic, Xi64, opnode, RegMRM>, PL;
  }

  def NAME#8mr    : BinOpMRF_MF<BaseOpc, mnemonic, Xi8 , opnode>;
  def NAME#16mr   : BinOpMRF_MF<BaseOpc, mnemonic, Xi16, opnode>, OpSize16;
  def NAME#32mr   : BinOpMRF_MF<BaseOpc, mnemonic, Xi32, opnode>, OpSize32;
  def NAME#64mr   : BinOpMRF_MF<BaseOpc, mnemonic, Xi64, opnode>;
  let Predicates = [HasNDD, In64BitMode] in {
    def NAME#8mr_ND    : BinOpMRF_RF<BaseOpc, mnemonic, Xi8 , opnode>;
    def NAME#16mr_ND   : BinOpMRF_RF<BaseOpc, mnemonic, Xi16, opnode>, PD;
    def NAME#32mr_ND   : BinOpMRF_RF<BaseOpc, mnemonic, Xi32, opnode>;
    def NAME#64mr_ND   : BinOpMRF_RF<BaseOpc, mnemonic, Xi64, opnode>;
  }
  let Predicates = [In64BitMode] in {
    def NAME#8mr_EVEX    : BinOpMRF_MF<BaseOpc, mnemonic, Xi8 , null_frag>, PL;
    def NAME#16mr_EVEX   : BinOpMRF_MF<BaseOpc, mnemonic, Xi16, null_frag>, PL, PD;
    def NAME#32mr_EVEX   : BinOpMRF_MF<BaseOpc, mnemonic, Xi32, null_frag>, PL;
    def NAME#64mr_EVEX   : BinOpMRF_MF<BaseOpc, mnemonic, Xi64, null_frag>, PL;
  }

  // NOTE: These are order specific, we want the mi8 forms to be listed
  // first so that they are slightly preferred to the mi forms.
  def NAME#8mi    : BinOpMIF_MF<0x80, mnemonic, Xi8 , opnode, MemMRM>;
  def NAME#16mi8  : BinOpMI8F_MF<mnemonic, Xi16, MemMRM>, OpSize16;
  def NAME#32mi8  : BinOpMI8F_MF<mnemonic, Xi32, MemMRM>, OpSize32;
  let Predicates = [In64BitMode] in
    def NAME#64mi8  : BinOpMI8F_MF<mnemonic, Xi64, MemMRM>;
  def NAME#16mi   : BinOpMIF_MF<0x81, mnemonic, Xi16, opnode, MemMRM>, OpSize16;
  def NAME#32mi   : BinOpMIF_MF<0x81, mnemonic, Xi32, opnode, MemMRM>, OpSize32;
  let Predicates = [In64BitMode] in
    def NAME#64mi32 : BinOpMIF_MF<0x81, mnemonic, Xi64, opnode, MemMRM>;

  let Predicates = [HasNDD, In64BitMode] in {
    def NAME#8mi_ND    : BinOpMIF_RF<0x80, mnemonic, Xi8 , opnode, MemMRM>;
    def NAME#16mi8_ND  : BinOpMI8F_RF<mnemonic, Xi16, MemMRM>, PD;
    def NAME#32mi8_ND  : BinOpMI8F_RF<mnemonic, Xi32, MemMRM>;
    def NAME#64mi8_ND  : BinOpMI8F_RF<mnemonic, Xi64, MemMRM>;
    def NAME#16mi_ND   : BinOpMIF_RF<0x81, mnemonic, Xi16, opnode, MemMRM>, PD;
    def NAME#32mi_ND   : BinOpMIF_RF<0x81, mnemonic, Xi32, opnode, MemMRM>;
    def NAME#64mi32_ND : BinOpMIF_RF<0x81, mnemonic, Xi64, opnode, MemMRM>;
  }
  let Predicates = [In64BitMode] in {
    def NAME#8mi_EVEX    : BinOpMIF_MF<0x80, mnemonic, Xi8 , opnode, MemMRM>, PL;
    def NAME#16mi8_EVEX  : BinOpMI8F_MF<mnemonic, Xi16, MemMRM>, PL, PD;
    def NAME#32mi8_EVEX  : BinOpMI8F_MF<mnemonic, Xi32, MemMRM>, PL;
    def NAME#64mi8_EVEX  : BinOpMI8F_MF<mnemonic, Xi64, MemMRM>, PL;
    def NAME#16mi_EVEX   : BinOpMIF_MF<0x81, mnemonic, Xi16, opnode, MemMRM>, PL, PD;
    def NAME#32mi_EVEX   : BinOpMIF_MF<0x81, mnemonic, Xi32, opnode, MemMRM>, PL;
    def NAME#64mi32_EVEX : BinOpMIF_MF<0x81, mnemonic, Xi64, opnode, MemMRM>, PL;
  }

  // These are for the disassembler since 0x82 opcode behaves like 0x80, but
  // not in 64-bit mode.
  let Predicates = [Not64BitMode]  in {
    def NAME#8ri8 : BinOpRI8F_RF<0x82, mnemonic, Xi8, RegMRM>, DisassembleOnly;
  def NAME#8mi8 : BinOpMI8F_MF<mnemonic, Xi8, MemMRM>, DisassembleOnly;
  }

  def NAME#8i8 : BinOpAIF_AF<BaseOpc4, mnemonic, Xi8 , AL,
                             "{$src, %al|al, $src}">;
  def NAME#16i16 : BinOpAIF_AF<BaseOpc4, mnemonic, Xi16, AX,
                               "{$src, %ax|ax, $src}">, OpSize16;
  def NAME#32i32 : BinOpAIF_AF<BaseOpc4, mnemonic, Xi32, EAX,
                               "{$src, %eax|eax, $src}">, OpSize32;
  def NAME#64i32 : BinOpAIF_AF<BaseOpc4, mnemonic, Xi64, RAX,
                               "{$src, %rax|rax, $src}">;
}

/// ArithBinOp_F - This is an arithmetic binary operator where the pattern is
/// defined with "(set EFLAGS, (...".  It would be really nice to find a way
/// to factor this with the other ArithBinOp_*.
///
multiclass ArithBinOp_F<bits<8> BaseOpc, bits<8> BaseOpc2, bits<8> BaseOpc4,
                        string mnemonic, Format RegMRM, Format MemMRM,
                        SDNode opnode, bit CommutableRR,
                        bit ConvertibleToThreeAddress> {
  let isCommutable = CommutableRR in {
  def NAME#8rr  : BinOpRR_F<BaseOpc, mnemonic, Xi8 , opnode>;
    let isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
    def NAME#16rr : BinOpRR_F<BaseOpc, mnemonic, Xi16, opnode>, OpSize16;
    def NAME#32rr : BinOpRR_F<BaseOpc, mnemonic, Xi32, opnode>, OpSize32;
    def NAME#64rr : BinOpRR_F<BaseOpc, mnemonic, Xi64, opnode>;
    } // isConvertibleToThreeAddress
  } // isCommutable

  def NAME#8rr_REV  : BinOpRR_F_Rev<BaseOpc2, mnemonic, Xi8>;
  def NAME#16rr_REV : BinOpRR_F_Rev<BaseOpc2, mnemonic, Xi16>, OpSize16;
  def NAME#32rr_REV : BinOpRR_F_Rev<BaseOpc2, mnemonic, Xi32>, OpSize32;
  def NAME#64rr_REV : BinOpRR_F_Rev<BaseOpc2, mnemonic, Xi64>;

  def NAME#8rm   : BinOpRM_F<BaseOpc2, mnemonic, Xi8 , opnode>;
  def NAME#16rm  : BinOpRM_F<BaseOpc2, mnemonic, Xi16, opnode>, OpSize16;
  def NAME#32rm  : BinOpRM_F<BaseOpc2, mnemonic, Xi32, opnode>, OpSize32;
  def NAME#64rm  : BinOpRM_F<BaseOpc2, mnemonic, Xi64, opnode>;

  def NAME#8ri   : BinOpRI_F<0x80, mnemonic, Xi8 , opnode, RegMRM>;

  let isConvertibleToThreeAddress = ConvertibleToThreeAddress in {
  // NOTE: These are order specific, we want the ri8 forms to be listed
  // first so that they are slightly preferred to the ri forms.
  def NAME#16ri8 : BinOpRI8_F<0x83, mnemonic, Xi16, RegMRM>, OpSize16;
  def NAME#32ri8 : BinOpRI8_F<0x83, mnemonic, Xi32, RegMRM>, OpSize32;
  def NAME#64ri8 : BinOpRI8_F<0x83, mnemonic, Xi64, RegMRM>;

  def NAME#16ri  : BinOpRI_F<0x81, mnemonic, Xi16, opnode, RegMRM>, OpSize16;
  def NAME#32ri  : BinOpRI_F<0x81, mnemonic, Xi32, opnode, RegMRM>, OpSize32;
  def NAME#64ri32: BinOpRI_F<0x81, mnemonic, Xi64, opnode, RegMRM>;
  }

  def NAME#8mr    : BinOpMR_F<BaseOpc, mnemonic, Xi8 , opnode>;
  def NAME#16mr   : BinOpMR_F<BaseOpc, mnemonic, Xi16, opnode>, OpSize16;
  def NAME#32mr   : BinOpMR_F<BaseOpc, mnemonic, Xi32, opnode>, OpSize32;
  def NAME#64mr   : BinOpMR_F<BaseOpc, mnemonic, Xi64, opnode>;

  // NOTE: These are order specific, we want the mi8 forms to be listed
  // first so that they are slightly preferred to the mi forms.
  def NAME#16mi8  : BinOpMI8_F<mnemonic, Xi16, MemMRM>, OpSize16;
  def NAME#32mi8  : BinOpMI8_F<mnemonic, Xi32, MemMRM>, OpSize32;
  let Predicates = [In64BitMode] in
  def NAME#64mi8  : BinOpMI8_F<mnemonic, Xi64, MemMRM>;

  def NAME#8mi    : BinOpMI_F<0x80, mnemonic, Xi8 , opnode, MemMRM>;
  def NAME#16mi   : BinOpMI_F<0x81, mnemonic, Xi16, opnode, MemMRM>, OpSize16;
  def NAME#32mi   : BinOpMI_F<0x81, mnemonic, Xi32, opnode, MemMRM>, OpSize32;
  let Predicates = [In64BitMode] in
  def NAME#64mi32 : BinOpMI_F<0x81, mnemonic, Xi64, opnode, MemMRM>;

  // These are for the disassembler since 0x82 opcode behaves like 0x80, but
  // not in 64-bit mode.
  let Predicates = [Not64BitMode] in {
  def NAME#8ri8 : BinOpRI8_F<0x82, mnemonic, Xi8, RegMRM>, DisassembleOnly;
    let mayLoad = 1 in
    def NAME#8mi8 : BinOpMI8_F<mnemonic, Xi8, MemMRM>;
  }

  def NAME#8i8 : BinOpAI_F<BaseOpc4, mnemonic, Xi8 , AL,
                           "{$src, %al|al, $src}">;
  def NAME#16i16 : BinOpAI_F<BaseOpc4, mnemonic, Xi16, AX,
                           "{$src, %ax|ax, $src}">, OpSize16;
  def NAME#32i32 : BinOpAI_F<BaseOpc4, mnemonic, Xi32, EAX,
                           "{$src, %eax|eax, $src}">, OpSize32;
  def NAME#64i32 : BinOpAI_F<BaseOpc4, mnemonic, Xi64, RAX,
                           "{$src, %rax|rax, $src}">;
}


defm AND : ArithBinOp_RF<0x21, 0x23, 0x25, "and", MRM4r, MRM4m,
                         X86and_flag, and, 1, 0, 0>;
defm OR  : ArithBinOp_RF<0x09, 0x0B, 0x0D, "or", MRM1r, MRM1m,
                         X86or_flag, or, 1, 0, 0>;
defm XOR : ArithBinOp_RF<0x31, 0x33, 0x35, "xor", MRM6r, MRM6m,
                         X86xor_flag, xor, 1, 0, 0>;
defm ADD : ArithBinOp_RF<0x01, 0x03, 0x05, "add", MRM0r, MRM0m,
                         X86add_flag, add, 1, 1, 1>;
let isCompare = 1 in {
defm SUB : ArithBinOp_RF<0x29, 0x2B, 0x2D, "sub", MRM5r, MRM5m,
                         X86sub_flag, sub, 0, 1, 0>;
}

// Version of XOR8rr_NOREX that use GR8_NOREX. This is used by the handling of
// __builtin_parity where the last step xors an h-register with an l-register.
let isCodeGenOnly = 1, hasSideEffects = 0, Constraints = "$src1 = $dst",
    Defs = [EFLAGS], isCommutable = 1 in
def XOR8rr_NOREX : I<0x30, MRMDestReg, (outs GR8_NOREX:$dst),
                     (ins GR8_NOREX:$src1, GR8_NOREX:$src2),
                     "xor{b}\t{$src2, $dst|$dst, $src2}", []>,
                     Sched<[WriteALU]>;

// Arithmetic.
defm ADC : ArithBinOp_RFF<0x11, 0x13, 0x15, "adc", MRM2r, MRM2m, X86adc_flag,
                          1, 0>;
defm SBB : ArithBinOp_RFF<0x19, 0x1B, 0x1D, "sbb", MRM3r, MRM3m, X86sbb_flag,
                          0, 0>;

let isCompare = 1 in {
defm CMP : ArithBinOp_F<0x39, 0x3B, 0x3D, "cmp", MRM7r, MRM7m, X86cmp, 0, 0>;
}

// Patterns to recognize loads on the LHS of an ADC. We can't make X86adc_flag
// commutable since it has EFLAGs as an input.
def : Pat<(X86adc_flag (loadi8 addr:$src2), GR8:$src1, EFLAGS),
          (ADC8rm GR8:$src1, addr:$src2)>;
def : Pat<(X86adc_flag (loadi16 addr:$src2), GR16:$src1, EFLAGS),
          (ADC16rm GR16:$src1, addr:$src2)>;
def : Pat<(X86adc_flag (loadi32 addr:$src2), GR32:$src1, EFLAGS),
          (ADC32rm GR32:$src1, addr:$src2)>;
def : Pat<(X86adc_flag (loadi64 addr:$src2), GR64:$src1, EFLAGS),
          (ADC64rm GR64:$src1, addr:$src2)>;

// Patterns to recognize RMW ADC with loads in operand 1.
def : Pat<(store (X86adc_flag GR8:$src, (loadi8 addr:$dst), EFLAGS),
                 addr:$dst),
          (ADC8mr addr:$dst, GR8:$src)>;
def : Pat<(store (X86adc_flag GR16:$src, (loadi16 addr:$dst), EFLAGS),
                 addr:$dst),
          (ADC16mr addr:$dst, GR16:$src)>;
def : Pat<(store (X86adc_flag GR32:$src, (loadi32 addr:$dst), EFLAGS),
                 addr:$dst),
          (ADC32mr addr:$dst, GR32:$src)>;
def : Pat<(store (X86adc_flag GR64:$src, (loadi64 addr:$dst), EFLAGS),
                 addr:$dst),
          (ADC64mr addr:$dst, GR64:$src)>;

// Patterns for basic arithmetic ops with relocImm for the immediate field.
multiclass ArithBinOp_RF_relocImm_Pats<SDNode OpNodeFlag, SDNode OpNode> {
  def : Pat<(OpNodeFlag GR8:$src1, relocImm8_su:$src2),
            (!cast<Instruction>(NAME#"8ri") GR8:$src1, relocImm8_su:$src2)>;
  def : Pat<(OpNodeFlag GR16:$src1, relocImm16_su:$src2),
            (!cast<Instruction>(NAME#"16ri") GR16:$src1, relocImm16_su:$src2)>;
  def : Pat<(OpNodeFlag GR32:$src1, relocImm32_su:$src2),
            (!cast<Instruction>(NAME#"32ri") GR32:$src1, relocImm32_su:$src2)>;
  def : Pat<(OpNodeFlag GR64:$src1, i64relocImmSExt32_su:$src2),
            (!cast<Instruction>(NAME#"64ri32") GR64:$src1, i64relocImmSExt32_su:$src2)>;

  def : Pat<(store (OpNode (load addr:$dst), relocImm8_su:$src), addr:$dst),
            (!cast<Instruction>(NAME#"8mi") addr:$dst, relocImm8_su:$src)>;
  def : Pat<(store (OpNode (load addr:$dst), relocImm16_su:$src), addr:$dst),
            (!cast<Instruction>(NAME#"16mi") addr:$dst, relocImm16_su:$src)>;
  def : Pat<(store (OpNode (load addr:$dst), relocImm32_su:$src), addr:$dst),
            (!cast<Instruction>(NAME#"32mi") addr:$dst, relocImm32_su:$src)>;
  def : Pat<(store (OpNode (load addr:$dst), i64relocImmSExt32_su:$src), addr:$dst),
            (!cast<Instruction>(NAME#"64mi32") addr:$dst, i64relocImmSExt32_su:$src)>;
}

multiclass ArithBinOp_RFF_relocImm_Pats<SDNode OpNodeFlag> {
  def : Pat<(OpNodeFlag GR8:$src1, relocImm8_su:$src2, EFLAGS),
            (!cast<Instruction>(NAME#"8ri") GR8:$src1, relocImm8_su:$src2)>;
  def : Pat<(OpNodeFlag GR16:$src1, relocImm16_su:$src2, EFLAGS),
            (!cast<Instruction>(NAME#"16ri") GR16:$src1, relocImm16_su:$src2)>;
  def : Pat<(OpNodeFlag GR32:$src1, relocImm32_su:$src2, EFLAGS),
            (!cast<Instruction>(NAME#"32ri") GR32:$src1, relocImm32_su:$src2)>;
  def : Pat<(OpNodeFlag GR64:$src1, i64relocImmSExt32_su:$src2, EFLAGS),
            (!cast<Instruction>(NAME#"64ri32") GR64:$src1, i64relocImmSExt32_su:$src2)>;

  def : Pat<(store (OpNodeFlag (load addr:$dst), relocImm8_su:$src, EFLAGS), addr:$dst),
            (!cast<Instruction>(NAME#"8mi") addr:$dst, relocImm8_su:$src)>;
  def : Pat<(store (OpNodeFlag (load addr:$dst), relocImm16_su:$src, EFLAGS), addr:$dst),
            (!cast<Instruction>(NAME#"16mi") addr:$dst, relocImm16_su:$src)>;
  def : Pat<(store (OpNodeFlag (load addr:$dst), relocImm32_su:$src, EFLAGS), addr:$dst),
            (!cast<Instruction>(NAME#"32mi") addr:$dst, relocImm32_su:$src)>;
  def : Pat<(store (OpNodeFlag (load addr:$dst), i64relocImmSExt32_su:$src, EFLAGS), addr:$dst),
            (!cast<Instruction>(NAME#"64mi32") addr:$dst, i64relocImmSExt32_su:$src)>;
}

multiclass ArithBinOp_F_relocImm_Pats<SDNode OpNodeFlag> {
  def : Pat<(OpNodeFlag GR8:$src1, relocImm8_su:$src2),
            (!cast<Instruction>(NAME#"8ri") GR8:$src1, relocImm8_su:$src2)>;
  def : Pat<(OpNodeFlag GR16:$src1, relocImm16_su:$src2),
            (!cast<Instruction>(NAME#"16ri") GR16:$src1, relocImm16_su:$src2)>;
  def : Pat<(OpNodeFlag GR32:$src1, relocImm32_su:$src2),
            (!cast<Instruction>(NAME#"32ri") GR32:$src1, relocImm32_su:$src2)>;
  def : Pat<(OpNodeFlag GR64:$src1, i64relocImmSExt32_su:$src2),
            (!cast<Instruction>(NAME#"64ri32") GR64:$src1, i64relocImmSExt32_su:$src2)>;

  def : Pat<(OpNodeFlag (loadi8 addr:$src1), relocImm8_su:$src2),
            (!cast<Instruction>(NAME#"8mi") addr:$src1, relocImm8_su:$src2)>;
  def : Pat<(OpNodeFlag (loadi16 addr:$src1), relocImm16_su:$src2),
            (!cast<Instruction>(NAME#"16mi") addr:$src1, relocImm16_su:$src2)>;
  def : Pat<(OpNodeFlag (loadi32 addr:$src1), relocImm32_su:$src2),
            (!cast<Instruction>(NAME#"32mi") addr:$src1, relocImm32_su:$src2)>;
  def : Pat<(OpNodeFlag (loadi64 addr:$src1), i64relocImmSExt32_su:$src2),
            (!cast<Instruction>(NAME#"64mi32") addr:$src1, i64relocImmSExt32_su:$src2)>;
}

defm AND : ArithBinOp_RF_relocImm_Pats<X86and_flag, and>;
defm OR  : ArithBinOp_RF_relocImm_Pats<X86or_flag, or>;
defm XOR : ArithBinOp_RF_relocImm_Pats<X86xor_flag, xor>;
defm ADD : ArithBinOp_RF_relocImm_Pats<X86add_flag, add>;
defm SUB : ArithBinOp_RF_relocImm_Pats<X86sub_flag, sub>;

defm ADC : ArithBinOp_RFF_relocImm_Pats<X86adc_flag>;
defm SBB : ArithBinOp_RFF_relocImm_Pats<X86sbb_flag>;

defm CMP : ArithBinOp_F_relocImm_Pats<X86cmp>;

// ADC is commutable, but we can't indicate that to tablegen. So manually
// reverse the operands.
def : Pat<(X86adc_flag GR8:$src1, relocImm8_su:$src2, EFLAGS),
          (ADC8ri relocImm8_su:$src2, GR8:$src1)>;
def : Pat<(X86adc_flag i16relocImmSExt8_su:$src2, GR16:$src1, EFLAGS),
          (ADC16ri8 GR16:$src1, i16relocImmSExt8_su:$src2)>;
def : Pat<(X86adc_flag relocImm16_su:$src2, GR16:$src1, EFLAGS),
          (ADC16ri GR16:$src1, relocImm16_su:$src2)>;
def : Pat<(X86adc_flag i32relocImmSExt8_su:$src2, GR32:$src1, EFLAGS),
          (ADC32ri8 GR32:$src1, i32relocImmSExt8_su:$src2)>;
def : Pat<(X86adc_flag relocImm32_su:$src2, GR32:$src1, EFLAGS),
          (ADC32ri GR32:$src1, relocImm32_su:$src2)>;
def : Pat<(X86adc_flag i64relocImmSExt8_su:$src2, GR64:$src1, EFLAGS),
          (ADC64ri8 GR64:$src1, i64relocImmSExt8_su:$src2)>;
def : Pat<(X86adc_flag i64relocImmSExt32_su:$src2, GR64:$src1, EFLAGS),
          (ADC64ri32 GR64:$src1, i64relocImmSExt32_su:$src2)>;

def : Pat<(store (X86adc_flag relocImm8_su:$src, (load addr:$dst), EFLAGS), addr:$dst),
          (ADC8mi addr:$dst, relocImm8_su:$src)>;
def : Pat<(store (X86adc_flag i16relocImmSExt8_su:$src, (load addr:$dst), EFLAGS), addr:$dst),
          (ADC16mi8 addr:$dst, i16relocImmSExt8_su:$src)>;
def : Pat<(store (X86adc_flag relocImm16_su:$src, (load addr:$dst), EFLAGS), addr:$dst),
          (ADC16mi addr:$dst, relocImm16_su:$src)>;
def : Pat<(store (X86adc_flag i32relocImmSExt8_su:$src, (load addr:$dst), EFLAGS), addr:$dst),
          (ADC32mi8 addr:$dst, i32relocImmSExt8_su:$src)>;
def : Pat<(store (X86adc_flag relocImm32_su:$src, (load addr:$dst), EFLAGS), addr:$dst),
          (ADC32mi addr:$dst, relocImm32_su:$src)>;
def : Pat<(store (X86adc_flag i64relocImmSExt8_su:$src, (load addr:$dst), EFLAGS), addr:$dst),
          (ADC64mi8 addr:$dst, i64relocImmSExt8_su:$src)>;
def : Pat<(store (X86adc_flag i64relocImmSExt32_su:$src, (load addr:$dst), EFLAGS), addr:$dst),
          (ADC64mi32 addr:$dst, i64relocImmSExt32_su:$src)>;

//===----------------------------------------------------------------------===//
// Semantically, test instructions are similar like AND, except they don't
// generate a result.  From an encoding perspective, they are very different:
// they don't have all the usual imm8 and REV forms, and are encoded into a
// different space.
let isCompare = 1 in {
  let isCommutable = 1 in {
  // Avoid selecting these and instead use a test+and. Post processing will
  // combine them. This gives bunch of other patterns that start with
  // and a chance to match.
  def TEST8rr  : BinOpRR_F<0x84, "test", Xi8 , null_frag>;
  def TEST16rr : BinOpRR_F<0x85, "test", Xi16, null_frag>, OpSize16;
  def TEST32rr : BinOpRR_F<0x85, "test", Xi32, null_frag>, OpSize32;
  def TEST64rr : BinOpRR_F<0x85, "test", Xi64, null_frag>;
  } // isCommutable

def TEST8mr    : BinOpMR_F<0x84, "test", Xi8 , null_frag>;
def TEST16mr   : BinOpMR_F<0x85, "test", Xi16, null_frag>, OpSize16;
def TEST32mr   : BinOpMR_F<0x85, "test", Xi32, null_frag>, OpSize32;
def TEST64mr   : BinOpMR_F<0x85, "test", Xi64, null_frag>;

def TEST8ri    : BinOpRI_F<0xF6, "test", Xi8 , X86testpat, MRM0r>;
def TEST16ri   : BinOpRI_F<0xF7, "test", Xi16, X86testpat, MRM0r>, OpSize16;
def TEST32ri   : BinOpRI_F<0xF7, "test", Xi32, X86testpat, MRM0r>, OpSize32;
def TEST64ri32 : BinOpRI_F<0xF7, "test", Xi64, X86testpat, MRM0r>;

def TEST8mi    : BinOpMI_F<0xF6, "test", Xi8 , X86testpat, MRM0m>;
def TEST16mi   : BinOpMI_F<0xF7, "test", Xi16, X86testpat, MRM0m>, OpSize16;
def TEST32mi   : BinOpMI_F<0xF7, "test", Xi32, X86testpat, MRM0m>, OpSize32;

  let Predicates = [In64BitMode] in
  def TEST64mi32 : BinOpMI_F<0xF7, "test", Xi64, X86testpat, MRM0m>;

def TEST8i8 : BinOpAI_F<0xA8, "test", Xi8 , AL, "{$src, %al|al, $src}">;
def TEST16i16 : BinOpAI_F<0xA9, "test", Xi16, AX, "{$src, %ax|ax, $src}">, OpSize16;
def TEST32i32 : BinOpAI_F<0xA9, "test", Xi32, EAX, "{$src, %eax|eax, $src}">, OpSize32;
def TEST64i32 : BinOpAI_F<0xA9, "test", Xi64, RAX, "{$src, %rax|rax, $src}">;
} // isCompare

// Patterns to match a relocImm into the immediate field.
def : Pat<(X86testpat GR8:$src1, relocImm8_su:$src2),
          (TEST8ri GR8:$src1, relocImm8_su:$src2)>;
def : Pat<(X86testpat GR16:$src1, relocImm16_su:$src2),
          (TEST16ri GR16:$src1, relocImm16_su:$src2)>;
def : Pat<(X86testpat GR32:$src1, relocImm32_su:$src2),
          (TEST32ri GR32:$src1, relocImm32_su:$src2)>;
def : Pat<(X86testpat GR64:$src1, i64relocImmSExt32_su:$src2),
          (TEST64ri32 GR64:$src1, i64relocImmSExt32_su:$src2)>;

def : Pat<(X86testpat (loadi8 addr:$src1), relocImm8_su:$src2),
          (TEST8mi addr:$src1, relocImm8_su:$src2)>;
def : Pat<(X86testpat (loadi16 addr:$src1), relocImm16_su:$src2),
          (TEST16mi addr:$src1, relocImm16_su:$src2)>;
def : Pat<(X86testpat (loadi32 addr:$src1), relocImm32_su:$src2),
          (TEST32mi addr:$src1, relocImm32_su:$src2)>;
def : Pat<(X86testpat (loadi64 addr:$src1), i64relocImmSExt32_su:$src2),
          (TEST64mi32 addr:$src1, i64relocImmSExt32_su:$src2)>;

//===----------------------------------------------------------------------===//
// ANDN Instruction
//
multiclass AndN<X86TypeInfo t, string suffix> {
  defvar andn_rr_p =
    [(set t.RegClass:$dst, EFLAGS, (X86and_flag (not t.RegClass:$src1),
     t.RegClass:$src2))];
  defvar andn_rm_p =
    [(set t.RegClass:$dst, EFLAGS, (X86and_flag (not t.RegClass:$src1),
     (t.LoadNode addr:$src2)))];
  def rr#suffix : ITy<0xF2, MRMSrcReg, t, (outs t.RegClass:$dst),
                      (ins t.RegClass:$src1, t.RegClass:$src2), "andn",
                      binop_ndd_args, andn_rr_p>, VVVV, Sched<[WriteALU]>,
                     T8, DefEFLAGS;
  def rm#suffix : ITy<0xF2, MRMSrcMem, t, (outs t.RegClass:$dst),
                       (ins t.RegClass:$src1, t.MemOperand:$src2), "andn",
                       binop_ndd_args, andn_rm_p>, VVVV,
                       Sched<[WriteALU.Folded, WriteALU.ReadAfterFold]>,
                       T8, DefEFLAGS;
}

// Complexity is reduced to give and with immediate a chance to match first.
let AddedComplexity = -6 in {
defm ANDN32 : AndN<Xi32, "">, VEX, Requires<[HasBMI, NoEGPR]>;
defm ANDN64 : AndN<Xi64, "">, VEX, REX_W, Requires<[HasBMI, NoEGPR]>;
defm ANDN32 : AndN<Xi32, "_EVEX">, EVEX, Requires<[HasBMI, HasEGPR, In64BitMode]>;
defm ANDN64 : AndN<Xi64, "_EVEX">, EVEX, REX_W, Requires<[HasBMI, HasEGPR, In64BitMode]>;
}

let Predicates = [HasBMI], AddedComplexity = -6 in {
  def : Pat<(and (not GR32:$src1), GR32:$src2),
            (ANDN32rr GR32:$src1, GR32:$src2)>;
  def : Pat<(and (not GR64:$src1), GR64:$src2),
            (ANDN64rr GR64:$src1, GR64:$src2)>;
  def : Pat<(and (not GR32:$src1), (loadi32 addr:$src2)),
            (ANDN32rm GR32:$src1, addr:$src2)>;
  def : Pat<(and (not GR64:$src1), (loadi64 addr:$src2)),
            (ANDN64rm GR64:$src1, addr:$src2)>;
}

//===----------------------------------------------------------------------===//
// MULX Instruction
//
multiclass MulX<X86TypeInfo t, X86FoldableSchedWrite sched> {
  defvar mulx_args = "{$src, $dst2, $dst1|$dst1, $dst2, $src}";
  defvar mulx_rm_sched =
    [WriteIMulHLd, sched.Folded,
     // Memory operand.
     ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault,
     // Implicit read of EDX/RDX
     sched.ReadAfterFold];

  def rr : ITy<0xF6, MRMSrcReg, t, (outs t.RegClass:$dst1, t.RegClass:$dst2),
               (ins t.RegClass:$src), "mulx", mulx_args, []>, T8, XD, VEX,
           VVVV, Sched<[WriteIMulH, sched]>;
  let mayLoad = 1 in
  def rm : ITy<0xF6, MRMSrcMem, t, (outs t.RegClass:$dst1, t.RegClass:$dst2),
               (ins t.MemOperand:$src), "mulx", mulx_args, []>, T8, XD, VEX,
               VVVV, Sched<mulx_rm_sched>;

  let Predicates = [In64BitMode] in {
  def rr_EVEX : ITy<0xF6, MRMSrcReg, t,
                    (outs t.RegClass:$dst1, t.RegClass:$dst2),
                    (ins t.RegClass:$src), "mulx", mulx_args, []>, T8, XD,
                EVEX, VVVV, Sched<[WriteIMulH, sched]>;
    let mayLoad = 1 in
    def rm_EVEX : ITy<0xF6, MRMSrcMem, t,
                      (outs t.RegClass:$dst1, t.RegClass:$dst2),
                      (ins t.MemOperand:$src), "mulx", mulx_args, []>, T8, XD,
                  EVEX, VVVV, Sched<mulx_rm_sched>;
  }
  // Pseudo instructions to be used when the low result isn't used. The
  // instruction is defined to keep the high if both destinations are the same.
  def Hrr : PseudoI<(outs t.RegClass:$dst), (ins t.RegClass:$src), []>,
            Sched<[sched]>;
  let mayLoad = 1 in
  def Hrm : PseudoI<(outs t.RegClass:$dst), (ins t.MemOperand:$src), []>,
            Sched<[sched.Folded]>;
}

let Uses = [EDX] in
defm MULX32 : MulX<Xi32, WriteMULX32>;

let Uses = [RDX] in
defm MULX64 : MulX<Xi64, WriteMULX64>, REX_W;

//===----------------------------------------------------------------------===//
// ADCX and ADOX Instructions
//
// We don't have patterns for these as there is no advantage over ADC for
// most code.
let Form = MRMSrcReg in {
def ADCX32rr : BinOpRRF_RF<0xF6, "adcx", Xi32, null_frag>, T8, PD;
def ADCX64rr : BinOpRRF_RF<0xF6, "adcx", Xi64, null_frag>, T8, PD;
def ADOX32rr : BinOpRRF_RF<0xF6, "adox", Xi32, null_frag>, T8, XS;
def ADOX64rr : BinOpRRF_RF<0xF6, "adox", Xi64, null_frag>, T8, XS;
}
let Form = MRMSrcMem in {
def ADCX32rm : BinOpRMF_RF<0xF6, "adcx", Xi32, null_frag>, T8, PD;
def ADCX64rm : BinOpRMF_RF<0xF6, "adcx", Xi64, null_frag>, T8, PD;
def ADOX32rm : BinOpRMF_RF<0xF6, "adox", Xi32, null_frag>, T8, XS;
def ADOX64rm : BinOpRMF_RF<0xF6, "adox", Xi64, null_frag>, T8, XS;
}
