//===- SystemZInstrInfo.td - SystemZ Instruction defs ---------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the SystemZ instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SystemZ Instruction Predicate Definitions.
def IsZ10 : Predicate<"Subtarget.isZ10()">;

include "SystemZInstrFormats.td"

//===----------------------------------------------------------------------===//
// Type Constraints.
//===----------------------------------------------------------------------===//
class SDTCisI8<int OpNum> : SDTCisVT<OpNum, i8>;
class SDTCisI16<int OpNum> : SDTCisVT<OpNum, i16>;
class SDTCisI32<int OpNum> : SDTCisVT<OpNum, i32>;
class SDTCisI64<int OpNum> : SDTCisVT<OpNum, i64>;

//===----------------------------------------------------------------------===//
// Type Profiles.
//===----------------------------------------------------------------------===//
def SDT_SystemZCall         : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;
def SDT_SystemZCallSeqStart : SDCallSeqStart<[SDTCisI64<0>]>;
def SDT_SystemZCallSeqEnd   : SDCallSeqEnd<[SDTCisI64<0>, SDTCisI64<1>]>;
def SDT_CmpTest             : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;
def SDT_BrCond              : SDTypeProfile<0, 2,
                                           [SDTCisVT<0, OtherVT>,
                                            SDTCisI8<1>]>;
def SDT_SelectCC            : SDTypeProfile<1, 3,
                                           [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
                                            SDTCisI8<3>]>;
def SDT_Address             : SDTypeProfile<1, 1,
                                            [SDTCisSameAs<0, 1>, SDTCisPtrTy<0>]>;

//===----------------------------------------------------------------------===//
// SystemZ Specific Node Definitions.
//===----------------------------------------------------------------------===//
def SystemZretflag : SDNode<"SystemZISD::RET_FLAG", SDTNone,
                     [SDNPHasChain, SDNPOptInFlag]>;
def SystemZcall    : SDNode<"SystemZISD::CALL", SDT_SystemZCall,
                     [SDNPHasChain, SDNPOutFlag, SDNPOptInFlag]>;
def SystemZcallseq_start :
                 SDNode<"ISD::CALLSEQ_START", SDT_SystemZCallSeqStart,
                        [SDNPHasChain, SDNPOutFlag]>;
def SystemZcallseq_end :
                 SDNode<"ISD::CALLSEQ_END",   SDT_SystemZCallSeqEnd,
                        [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;
def SystemZcmp     : SDNode<"SystemZISD::CMP", SDT_CmpTest, [SDNPOutFlag]>;
def SystemZucmp    : SDNode<"SystemZISD::UCMP", SDT_CmpTest, [SDNPOutFlag]>;
def SystemZbrcond  : SDNode<"SystemZISD::BRCOND", SDT_BrCond,
                            [SDNPHasChain, SDNPInFlag]>;
def SystemZselect  : SDNode<"SystemZISD::SELECT", SDT_SelectCC, [SDNPInFlag]>;
def SystemZpcrelwrapper : SDNode<"SystemZISD::PCRelativeWrapper", SDT_Address, []>;


include "SystemZOperands.td"

//===----------------------------------------------------------------------===//
// Instruction list..

def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i64imm:$amt),
                              "#ADJCALLSTACKDOWN",
                              [(SystemZcallseq_start timm:$amt)]>;
def ADJCALLSTACKUP   : Pseudo<(outs), (ins i64imm:$amt1, i64imm:$amt2),
                              "#ADJCALLSTACKUP",
                              [(SystemZcallseq_end timm:$amt1, timm:$amt2)]>;

let usesCustomDAGSchedInserter = 1 in {
  def Select32 : Pseudo<(outs GR32:$dst), (ins GR32:$src1, GR32:$src2, i8imm:$cc),
                        "# Select32 PSEUDO",
                        [(set GR32:$dst,
                              (SystemZselect GR32:$src1, GR32:$src2, imm:$cc))]>;
  def Select64 : Pseudo<(outs GR64:$dst), (ins GR64:$src1, GR64:$src2, i8imm:$cc),
                        "# Select64 PSEUDO",
                        [(set GR64:$dst,
                              (SystemZselect GR64:$src1, GR64:$src2, imm:$cc))]>;
}


//===----------------------------------------------------------------------===//
//  Control Flow Instructions...
//

// FIXME: Provide proper encoding!
let isReturn = 1, isTerminator = 1, isBarrier = 1, hasCtrlDep = 1 in {
  def RET : Pseudo<(outs), (ins), "br\t%r14", [(SystemZretflag)]>;
}

let isBranch = 1, isTerminator = 1 in {
  let isBarrier = 1 in {
    def JMP  : Pseudo<(outs), (ins brtarget:$dst), "j\t{$dst}", [(br bb:$dst)]>;

    let isIndirectBranch = 1 in
      def JMPr   : Pseudo<(outs), (ins GR64:$dst), "br\t{$dst}", [(brind GR64:$dst)]>;
  }

  let Uses = [PSW] in {
    def JO  : Pseudo<(outs), (ins brtarget:$dst),
                     "jo\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_O)]>;
    def JH  : Pseudo<(outs), (ins brtarget:$dst),
                     "jh\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_H)]>;
    def JNLE: Pseudo<(outs), (ins brtarget:$dst),
                     "jnle\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_NLE)]>;
    def JL  : Pseudo<(outs), (ins brtarget:$dst),
                     "jl\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_L)]>;
    def JNHE: Pseudo<(outs), (ins brtarget:$dst),
                     "jnhe\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_NHE)]>;
    def JLH : Pseudo<(outs), (ins brtarget:$dst),
                     "jlh\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_LH)]>;
    def JNE : Pseudo<(outs), (ins brtarget:$dst),
                     "jne\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_NE)]>;
    def JE  : Pseudo<(outs), (ins brtarget:$dst),
                     "je\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_E)]>;
    def JNLH: Pseudo<(outs), (ins brtarget:$dst),
                     "jnlh\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_NLH)]>;
    def JHE : Pseudo<(outs), (ins brtarget:$dst),
                     "jhe\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_HE)]>;
    def JNL : Pseudo<(outs), (ins brtarget:$dst),
                     "jnl\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_NL)]>;
    def JLE : Pseudo<(outs), (ins brtarget:$dst),
                     "jle\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_LE)]>;
    def JNH : Pseudo<(outs), (ins brtarget:$dst),
                     "jnh\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_NH)]>;
    def JNO : Pseudo<(outs), (ins brtarget:$dst),
                     "jno\t$dst",
                     [(SystemZbrcond bb:$dst, SYSTEMZ_COND_NO)]>;
  } // Uses = [PSW]
} // isBranch = 1

//===----------------------------------------------------------------------===//
//  Call Instructions...
//

let isCall = 1 in
  // All calls clobber the non-callee saved registers. Uses for argument
  // registers are added manually.
  let Defs = [R0D, R1D, R2D, R3D, R4D, R5D, R14D,
              F0L, F1L, F2L, F3L, F4L, F5L, F6L, F7L] in {
    def CALLi     : Pseudo<(outs), (ins imm_pcrel:$dst, variable_ops),
                           "brasl\t%r14, $dst", [(SystemZcall imm:$dst)]>;
    def CALLr     : Pseudo<(outs), (ins ADDR64:$dst, variable_ops),
                           "basr\t%r14, $dst", [(SystemZcall ADDR64:$dst)]>;
  }

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//

let isReMaterializable = 1 in
// FIXME: Provide imm12 variant
// FIXME: Address should be halfword aligned...
def LA64r  : RXI<0x47,
                 (outs GR64:$dst), (ins laaddr:$src),
                 "lay\t{$dst, $src}",
                 [(set GR64:$dst, laaddr:$src)]>;
def LA64rm : RXYI<0x71E3,
                  (outs GR64:$dst), (ins i64imm:$src),
                  "larl\t{$dst, $src}",
                  [(set GR64:$dst,
                        (SystemZpcrelwrapper tglobaladdr:$src))]>;

let neverHasSideEffects = 1 in
def NOP : Pseudo<(outs), (ins), "# no-op", []>;

//===----------------------------------------------------------------------===//
// Move Instructions

let neverHasSideEffects = 1 in {
def MOV32rr : RRI<0x18,
                  (outs GR32:$dst), (ins GR32:$src),
                  "lr\t{$dst, $src}",
                  []>;
def MOV64rr : RREI<0xB904,
                   (outs GR64:$dst), (ins GR64:$src),
                   "lgr\t{$dst, $src}",
                   []>;
def MOV128rr : Pseudo<(outs GR128:$dst), (ins GR128:$src),
                     "# MOV128 PSEUDO!\n"
                     "\tlgr\t${dst:subreg_odd}, ${src:subreg_odd}\n"
                     "\tlgr\t${dst:subreg_even}, ${src:subreg_even}",
                     []>;
def MOV64rrP : Pseudo<(outs GR64P:$dst), (ins GR64P:$src),
                     "# MOV64P PSEUDO!\n"
                     "\tlr\t${dst:subreg_odd}, ${src:subreg_odd}\n"
                     "\tlr\t${dst:subreg_even}, ${src:subreg_even}",
                     []>;
}

def MOVSX64rr32 : RREI<0xB914,
                       (outs GR64:$dst), (ins GR32:$src),
                       "lgfr\t{$dst, $src}",
                       [(set GR64:$dst, (sext GR32:$src))]>;
def MOVZX64rr32 : RREI<0xB916,
                       (outs GR64:$dst), (ins GR32:$src),
                       "llgfr\t{$dst, $src}",
                       [(set GR64:$dst, (zext GR32:$src))]>;

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def MOV32ri16 : RII<0x8A7,
                    (outs GR32:$dst), (ins s16imm:$src),
                    "lhi\t{$dst, $src}",
                    [(set GR32:$dst, immSExt16:$src)]>;
def MOV64ri16 : RII<0x9A7,
                    (outs GR64:$dst), (ins s16imm64:$src),
                    "lghi\t{$dst, $src}",
                    [(set GR64:$dst, immSExt16:$src)]>;

def MOV64rill16 : RII<0xFA5,
                      (outs GR64:$dst), (ins i64imm:$src),
                      "llill\t{$dst, $src}",
                      [(set GR64:$dst, i64ll16:$src)]>;
def MOV64rilh16 : RII<0xEA5,
                      (outs GR64:$dst), (ins i64imm:$src),
                      "llilh\t{$dst, $src}",
                      [(set GR64:$dst, i64lh16:$src)]>;
def MOV64rihl16 : RII<0xDA5,
                      (outs GR64:$dst), (ins i64imm:$src),
                      "llihl\t{$dst, $src}",
                      [(set GR64:$dst, i64hl16:$src)]>;
def MOV64rihh16 : RII<0xCA5,
                      (outs GR64:$dst), (ins i64imm:$src),
                      "llihh\t{$dst, $src}",
                      [(set GR64:$dst, i64hh16:$src)]>;

def MOV64ri32 : RILI<0x1C0,
                     (outs GR64:$dst), (ins s32imm64:$src),
                     "lgfi\t{$dst, $src}",
                     [(set GR64:$dst, immSExt32:$src)]>;
def MOV64rilo32 : RILI<0xFC0,
                       (outs GR64:$dst), (ins i64imm:$src),
                       "llilf\t{$dst, $src}",
                       [(set GR64:$dst, i64lo32:$src)]>;
def MOV64rihi32 : RILI<0xEC0, (outs GR64:$dst), (ins i64imm:$src),
                       "llihf\t{$dst, $src}",
                       [(set GR64:$dst, i64hi32:$src)]>;
}

let canFoldAsLoad = 1, isReMaterializable = 1, mayHaveSideEffects = 1 in {
def MOV32rm  : RXI<0x58,
                   (outs GR32:$dst), (ins rriaddr12:$src),
                   "l\t{$dst, $src}",
                   [(set GR32:$dst, (load rriaddr12:$src))]>;
def MOV32rmy : RXYI<0x58E3,
                    (outs GR32:$dst), (ins rriaddr:$src),
                    "ly\t{$dst, $src}",
                    [(set GR32:$dst, (load rriaddr:$src))]>;
def MOV64rm  : RXYI<0x04E3,
                    (outs GR64:$dst), (ins rriaddr:$src),
                    "lg\t{$dst, $src}",
                    [(set GR64:$dst, (load rriaddr:$src))]>;
def MOV64Prm : Pseudo<(outs GR64P:$dst), (ins rriaddr12:$src),
                      "# MOV64P PSEUDO!\n"
                      "\tl\t${dst:subreg_odd},  $src\n"
                      "\tl\t${dst:subreg_even}, 4+$src",
                      [(set GR64P:$dst, (load rriaddr12:$src))]>;
def MOV64Prmy : Pseudo<(outs GR64P:$dst), (ins rriaddr:$src),
                       "# MOV64P PSEUDO!\n"
                       "\tly\t${dst:subreg_odd},  $src\n"
                       "\tly\t${dst:subreg_even}, 4+$src",
                       [(set GR64P:$dst, (load rriaddr:$src))]>;
def MOV128rm : Pseudo<(outs GR128:$dst), (ins rriaddr:$src),
                      "# MOV128 PSEUDO!\n"
                      "\tlg\t${dst:subreg_odd},  $src\n"
                      "\tlg\t${dst:subreg_even}, 8+$src",
                      [(set GR128:$dst, (load rriaddr:$src))]>;
}

def MOV32mr  : RXI<0x50,
                   (outs), (ins rriaddr12:$dst, GR32:$src),
                   "st\t{$src, $dst}",
                   [(store GR32:$src, rriaddr12:$dst)]>;
def MOV32mry : RXYI<0x50E3,
                    (outs), (ins rriaddr:$dst, GR32:$src),
                    "sty\t{$src, $dst}",
                    [(store GR32:$src, rriaddr:$dst)]>;
def MOV64mr  : RXYI<0x24E3,
                    (outs), (ins rriaddr:$dst, GR64:$src),
                    "stg\t{$src, $dst}",
                    [(store GR64:$src, rriaddr:$dst)]>;
def MOV64Pmr : Pseudo<(outs), (ins rriaddr12:$dst, GR64P:$src),
                      "# MOV64P PSEUDO!\n"
                      "\tst\t${src:subreg_odd}, $dst\n"
                      "\tst\t${src:subreg_even}, 4+$dst",
                      [(store GR64P:$src, rriaddr12:$dst)]>;
def MOV64Pmry : Pseudo<(outs), (ins rriaddr:$dst, GR64P:$src),
                       "# MOV64P PSEUDO!\n"
                       "\tsty\t${src:subreg_odd}, $dst\n"
                       "\tsty\t${src:subreg_even}, 4+$dst",
                       [(store GR64P:$src, rriaddr:$dst)]>;
def MOV128mr : Pseudo<(outs), (ins rriaddr:$dst, GR128:$src),
                      "# MOV128 PSEUDO!\n"
                      "\tstg\t${src:subreg_odd}, $dst\n"
                      "\tstg\t${src:subreg_even}, 8+$dst",
                      [(store GR128:$src, rriaddr:$dst)]>;

def MOV8mi    : SII<0x92,
                    (outs), (ins riaddr12:$dst, i32i8imm:$src),
                    "mvi\t{$dst, $src}",
                    [(truncstorei8 (i32 i32immSExt8:$src), riaddr12:$dst)]>;
def MOV8miy   : SIYI<0x52EB,
                     (outs), (ins riaddr:$dst, i32i8imm:$src),
                     "mviy\t{$dst, $src}",
                     [(truncstorei8 (i32 i32immSExt8:$src), riaddr:$dst)]>;

let AddedComplexity = 2 in {
def MOV16mi   : SILI<0xE544,
                     (outs), (ins riaddr12:$dst, s16imm:$src),
                     "mvhhi\t{$dst, $src}",
                     [(truncstorei16 (i32 i32immSExt16:$src), riaddr12:$dst)]>,
                     Requires<[IsZ10]>;
def MOV32mi16 : SILI<0xE54C,
                     (outs), (ins riaddr12:$dst, s32imm:$src),
                     "mvhi\t{$dst, $src}",
                     [(store (i32 immSExt16:$src), riaddr12:$dst)]>,
                     Requires<[IsZ10]>;
def MOV64mi16 : SILI<0xE548,
                     (outs), (ins riaddr12:$dst, s32imm64:$src),
                     "mvghi\t{$dst, $src}",
                     [(store (i64 immSExt16:$src), riaddr12:$dst)]>,
                     Requires<[IsZ10]>;
}

// sexts
def MOVSX32rr8  : RREI<0xB926,
                       (outs GR32:$dst), (ins GR32:$src),
                       "lbr\t{$dst, $src}",
                       [(set GR32:$dst, (sext_inreg GR32:$src, i8))]>;
def MOVSX64rr8  : RREI<0xB906,
                       (outs GR64:$dst), (ins GR64:$src),
                       "lgbr\t{$dst, $src}",
                       [(set GR64:$dst, (sext_inreg GR64:$src, i8))]>;
def MOVSX32rr16 : RREI<0xB927,
                       (outs GR32:$dst), (ins GR32:$src),
                       "lhr\t{$dst, $src}",
                       [(set GR32:$dst, (sext_inreg GR32:$src, i16))]>;
def MOVSX64rr16 : RREI<0xB907,
                       (outs GR64:$dst), (ins GR64:$src),
                       "lghr\t{$dst, $src}",
                       [(set GR64:$dst, (sext_inreg GR64:$src, i16))]>;

// extloads
def MOVSX32rm8   : RXYI<0x76E3,
                        (outs GR32:$dst), (ins rriaddr:$src),
                        "lb\t{$dst, $src}",
                        [(set GR32:$dst, (sextloadi32i8 rriaddr:$src))]>;
def MOVSX32rm16  : RXI<0x48,
                       (outs GR32:$dst), (ins rriaddr12:$src),
                       "lh\t{$dst, $src}",
                       [(set GR32:$dst, (sextloadi32i16 rriaddr12:$src))]>;
def MOVSX32rm16y : RXYI<0x78E3,
                        (outs GR32:$dst), (ins rriaddr:$src),
                        "lhy\t{$dst, $src}",
                        [(set GR32:$dst, (sextloadi32i16 rriaddr:$src))]>;
def MOVSX64rm8   : RXYI<0x77E3,
                        (outs GR64:$dst), (ins rriaddr:$src),
                        "lgb\t{$dst, $src}",
                        [(set GR64:$dst, (sextloadi64i8 rriaddr:$src))]>;
def MOVSX64rm16  : RXYI<0x15E3,
                        (outs GR64:$dst), (ins rriaddr:$src),
                        "lgh\t{$dst, $src}",
                        [(set GR64:$dst, (sextloadi64i16 rriaddr:$src))]>;
def MOVSX64rm32  : RXYI<0x14E3,
                        (outs GR64:$dst), (ins rriaddr:$src),
                        "lgf\t{$dst, $src}",
                        [(set GR64:$dst, (sextloadi64i32 rriaddr:$src))]>;

def MOVZX32rm8  : RXYI<0x94E3,
                       (outs GR32:$dst), (ins rriaddr:$src),
                       "llc\t{$dst, $src}",
                       [(set GR32:$dst, (zextloadi32i8 rriaddr:$src))]>;
def MOVZX32rm16 : RXYI<0x95E3,
                       (outs GR32:$dst), (ins rriaddr:$src),
                       "llh\t{$dst, $src}",
                       [(set GR32:$dst, (zextloadi32i16 rriaddr:$src))]>;
def MOVZX64rm8  : RXYI<0x90E3,
                       (outs GR64:$dst), (ins rriaddr:$src),
                       "llgc\t{$dst, $src}",
                       [(set GR64:$dst, (zextloadi64i8 rriaddr:$src))]>;
def MOVZX64rm16 : RXYI<0x91E3,
                       (outs GR64:$dst), (ins rriaddr:$src),
                       "llgh\t{$dst, $src}",
                       [(set GR64:$dst, (zextloadi64i16 rriaddr:$src))]>;
def MOVZX64rm32 : RXYI<0x16E3,
                       (outs GR64:$dst), (ins rriaddr:$src),
                       "llgf\t{$dst, $src}",
                       [(set GR64:$dst, (zextloadi64i32 rriaddr:$src))]>;

// truncstores
def MOV32m8r   : RXI<0x42,
                     (outs), (ins rriaddr12:$dst, GR32:$src),
                     "stc\t{$src, $dst}",
                     [(truncstorei8 GR32:$src, rriaddr12:$dst)]>;

def MOV32m8ry  : RXYI<0x72E3,
                      (outs), (ins rriaddr:$dst, GR32:$src),
                      "stcy\t{$src, $dst}",
                      [(truncstorei8 GR32:$src, rriaddr:$dst)]>;

def MOV32m16r  : RXI<0x40,
                     (outs), (ins rriaddr12:$dst, GR32:$src),
                     "sth\t{$src, $dst}",
                     [(truncstorei16 GR32:$src, rriaddr12:$dst)]>;

def MOV32m16ry : RXYI<0x70E3,
                      (outs), (ins rriaddr:$dst, GR32:$src),
                      "sthy\t{$src, $dst}",
                      [(truncstorei16 GR32:$src, rriaddr:$dst)]>;

def MOV64m8r   : RXI<0x42,
                     (outs), (ins rriaddr12:$dst, GR64:$src),
                     "stc\t{$src, $dst}",
                     [(truncstorei8 GR64:$src, rriaddr12:$dst)]>;

def MOV64m8ry  : RXYI<0x72E3,
                      (outs), (ins rriaddr:$dst, GR64:$src),
                      "stcy\t{$src, $dst}",
                      [(truncstorei8 GR64:$src, rriaddr:$dst)]>;

def MOV64m16r  : RXI<0x40,
                     (outs), (ins rriaddr12:$dst, GR64:$src),
                     "sth\t{$src, $dst}",
                     [(truncstorei16 GR64:$src, rriaddr12:$dst)]>;

def MOV64m16ry : RXYI<0x70E3,
                      (outs), (ins rriaddr:$dst, GR64:$src),
                      "sthy\t{$src, $dst}",
                      [(truncstorei16 GR64:$src, rriaddr:$dst)]>;

def MOV64m32r  : RXI<0x50,
                     (outs), (ins rriaddr12:$dst, GR64:$src),
                     "st\t{$src, $dst}",
                     [(truncstorei32 GR64:$src, rriaddr12:$dst)]>;

def MOV64m32ry : RXYI<0x50E3,
                      (outs), (ins rriaddr:$dst, GR64:$src),
                      "sty\t{$src, $dst}",
                      [(truncstorei32 GR64:$src, rriaddr:$dst)]>;

// multiple regs moves
// FIXME: should we use multiple arg nodes?
def MOV32mrm  : RSYI<0x90EB,
                     (outs), (ins riaddr:$dst, GR32:$from, GR32:$to),
                     "stmy\t{$from, $to, $dst}",
                     []>;
def MOV64mrm  : RSYI<0x24EB,
                     (outs), (ins riaddr:$dst, GR64:$from, GR64:$to),
                     "stmg\t{$from, $to, $dst}",
                     []>;
def MOV32rmm  : RSYI<0x90EB,
                     (outs GR32:$from, GR32:$to), (ins riaddr:$dst),
                     "lmy\t{$from, $to, $dst}",
                     []>;
def MOV64rmm  : RSYI<0x04EB,
                     (outs GR64:$from, GR64:$to), (ins riaddr:$dst),
                     "lmg\t{$from, $to, $dst}",
                     []>;

let isReMaterializable = 1, isAsCheapAsAMove = 1, isTwoAddress = 1 in {
def MOV64Pr0_even : Pseudo<(outs GR64P:$dst), (ins GR64P:$src),
                           "lhi\t${dst:subreg_even}, 0",
                           []>;
def MOV128r0_even : Pseudo<(outs GR128:$dst), (ins GR128:$src),
                           "lghi\t${dst:subreg_even}, 0",
                           []>;
}

// Byte swaps
def BSWAP32rr : RREI<0xB91F,
                     (outs GR32:$dst), (ins GR32:$src),
                     "lrvr\t{$dst, $src}",
                     [(set GR32:$dst, (bswap GR32:$src))]>;
def BSWAP64rr : RREI<0xB90F,
                     (outs GR64:$dst), (ins GR64:$src),
                     "lrvgr\t{$dst, $src}",
                     [(set GR64:$dst, (bswap GR64:$src))]>;

// FIXME: this is invalid pattern for big-endian
//def BSWAP16rm : RXYI<0x1FE3, (outs GR32:$dst), (ins rriaddr:$src),
//                     "lrvh\t{$dst, $src}",
//                     [(set GR32:$dst, (bswap (extloadi32i16 rriaddr:$src)))]>;
def BSWAP32rm : RXYI<0x1EE3, (outs GR32:$dst), (ins rriaddr:$src),
                     "lrv\t{$dst, $src}",
                     [(set GR32:$dst, (bswap (load rriaddr:$src)))]>;
def BSWAP64rm : RXYI<0x0FE3, (outs GR64:$dst), (ins rriaddr:$src),
                     "lrvg\t{$dst, $src}",
                     [(set GR64:$dst, (bswap (load rriaddr:$src)))]>;

//def BSWAP16mr : RXYI<0xE33F, (outs), (ins rriaddr:$dst, GR32:$src),
//                     "strvh\t{$src, $dst}",
//                     [(truncstorei16 (bswap GR32:$src), rriaddr:$dst)]>;
def BSWAP32mr : RXYI<0xE33E, (outs), (ins rriaddr:$dst, GR32:$src),
                     "strv\t{$src, $dst}",
                     [(store (bswap GR32:$src), rriaddr:$dst)]>;
def BSWAP64mr : RXYI<0xE32F, (outs), (ins rriaddr:$dst, GR64:$src),
                     "strvg\t{$src, $dst}",
                     [(store (bswap GR64:$src), rriaddr:$dst)]>;

//===----------------------------------------------------------------------===//
// Arithmetic Instructions

let Defs = [PSW] in {
def NEG32rr : RRI<0x13,
                  (outs GR32:$dst), (ins GR32:$src),
                  "lcr\t{$dst, $src}",
                  [(set GR32:$dst, (ineg GR32:$src)),
                   (implicit PSW)]>;
def NEG64rr : RREI<0xB903, (outs GR64:$dst), (ins GR64:$src),
                   "lcgr\t{$dst, $src}",
                   [(set GR64:$dst, (ineg GR64:$src)),
                    (implicit PSW)]>;
def NEG64rr32 : RREI<0xB913, (outs GR64:$dst), (ins GR32:$src),
                     "lcgfr\t{$dst, $src}",
                     [(set GR64:$dst, (ineg (sext GR32:$src))),
                      (implicit PSW)]>;
}

let isTwoAddress = 1 in {

let Defs = [PSW] in {

let isCommutable = 1 in { // X = ADD Y, Z  == X = ADD Z, Y
def ADD32rr : RRI<0x1A, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "ar\t{$dst, $src2}",
                  [(set GR32:$dst, (add GR32:$src1, GR32:$src2)),
                   (implicit PSW)]>;
def ADD64rr : RREI<0xB908, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "agr\t{$dst, $src2}",
                   [(set GR64:$dst, (add GR64:$src1, GR64:$src2)),
                    (implicit PSW)]>;
}

def ADD32rm   : RXI<0x5A, (outs GR32:$dst), (ins GR32:$src1, rriaddr12:$src2),
                    "a\t{$dst, $src2}",
                    [(set GR32:$dst, (add GR32:$src1, (load rriaddr12:$src2))),
                     (implicit PSW)]>;
def ADD32rmy  : RXYI<0xE35A, (outs GR32:$dst), (ins GR32:$src1, rriaddr:$src2),
                     "ay\t{$dst, $src2}",
                     [(set GR32:$dst, (add GR32:$src1, (load rriaddr:$src2))),
                      (implicit PSW)]>;
def ADD64rm   : RXYI<0xE308, (outs GR64:$dst), (ins GR64:$src1, rriaddr:$src2),
                     "ag\t{$dst, $src2}",
                     [(set GR64:$dst, (add GR64:$src1, (load rriaddr:$src2))),
                      (implicit PSW)]>;


def ADD32ri16 : RII<0xA7A,
                    (outs GR32:$dst), (ins GR32:$src1, s16imm:$src2),
                    "ahi\t{$dst, $src2}",
                    [(set GR32:$dst, (add GR32:$src1, immSExt16:$src2)),
                     (implicit PSW)]>;
def ADD32ri   : RILI<0xC29,
                     (outs GR32:$dst), (ins GR32:$src1, s32imm:$src2),
                     "afi\t{$dst, $src2}",
                     [(set GR32:$dst, (add GR32:$src1, imm:$src2)),
                      (implicit PSW)]>;
def ADD64ri16 : RILI<0xA7B,
                     (outs GR64:$dst), (ins GR64:$src1, s16imm64:$src2),
                     "aghi\t{$dst, $src2}",
                     [(set GR64:$dst, (add GR64:$src1, immSExt16:$src2)),
                      (implicit PSW)]>;
def ADD64ri32 : RILI<0xC28,
                     (outs GR64:$dst), (ins GR64:$src1, s32imm64:$src2),
                     "agfi\t{$dst, $src2}",
                     [(set GR64:$dst, (add GR64:$src1, immSExt32:$src2)),
                      (implicit PSW)]>;

let isCommutable = 1 in { // X = ADC Y, Z  == X = ADC Z, Y
def ADC32rr : RRI<0x1E, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "alr\t{$dst, $src2}",
                  [(set GR32:$dst, (addc GR32:$src1, GR32:$src2))]>;
def ADC64rr : RREI<0xB90A, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "algr\t{$dst, $src2}",
                   [(set GR64:$dst, (addc GR64:$src1, GR64:$src2))]>;
}

def ADC32ri   : RILI<0xC2B,
                     (outs GR32:$dst), (ins GR32:$src1, s32imm:$src2),
                     "alfi\t{$dst, $src2}",
                     [(set GR32:$dst, (addc GR32:$src1, imm:$src2))]>;
def ADC64ri32 : RILI<0xC2A,
                     (outs GR64:$dst), (ins GR64:$src1, s32imm64:$src2),
                     "algfi\t{$dst, $src2}",
                     [(set GR64:$dst, (addc GR64:$src1, immSExt32:$src2))]>;

let Uses = [PSW] in {
def ADDE32rr : RREI<0xB998, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                    "alcr\t{$dst, $src2}",
                    [(set GR32:$dst, (adde GR32:$src1, GR32:$src2)),
                     (implicit PSW)]>;
def ADDE64rr : RREI<0xB988, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                    "alcgr\t{$dst, $src2}",
                    [(set GR64:$dst, (adde GR64:$src1, GR64:$src2)),
                     (implicit PSW)]>;
}

let isCommutable = 1 in { // X = AND Y, Z  == X = AND Z, Y
def AND32rr : RRI<0x14,
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "nr\t{$dst, $src2}",
                  [(set GR32:$dst, (and GR32:$src1, GR32:$src2))]>;
def AND64rr : RREI<0xB980,
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "ngr\t{$dst, $src2}",
                   [(set GR64:$dst, (and GR64:$src1, GR64:$src2))]>;
}

def AND32rm   : RXI<0x54, (outs GR32:$dst), (ins GR32:$src1, rriaddr12:$src2),
                    "n\t{$dst, $src2}",
                    [(set GR32:$dst, (and GR32:$src1, (load rriaddr12:$src2))),
                     (implicit PSW)]>;
def AND32rmy  : RXYI<0xE354, (outs GR32:$dst), (ins GR32:$src1, rriaddr:$src2),
                     "ny\t{$dst, $src2}",
                     [(set GR32:$dst, (and GR32:$src1, (load rriaddr:$src2))),
                      (implicit PSW)]>;
def AND64rm   : RXYI<0xE360, (outs GR64:$dst), (ins GR64:$src1, rriaddr:$src2),
                     "ng\t{$dst, $src2}",
                     [(set GR64:$dst, (and GR64:$src1, (load rriaddr:$src2))),
                      (implicit PSW)]>;

def AND32rill16 : RII<0xA57,
                      (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                      "nill\t{$dst, $src2}",
                      [(set GR32:$dst, (and GR32:$src1, i32ll16c:$src2))]>;
def AND64rill16 : RII<0xA57,
                      (outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                      "nill\t{$dst, $src2}",
                      [(set GR64:$dst, (and GR64:$src1, i64ll16c:$src2))]>;

def AND32rilh16 : RII<0xA56,
                      (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                      "nilh\t{$dst, $src2}",
                      [(set GR32:$dst, (and GR32:$src1, i32lh16c:$src2))]>;
def AND64rilh16 : RII<0xA56,
                      (outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                      "nilh\t{$dst, $src2}",
                      [(set GR64:$dst, (and GR64:$src1, i64lh16c:$src2))]>;

def AND64rihl16 : RII<0xA55,
                      (outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                      "nihl\t{$dst, $src2}",
                      [(set GR64:$dst, (and GR64:$src1, i64hl16c:$src2))]>;
def AND64rihh16 : RII<0xA54,
                      (outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                      "nihh\t{$dst, $src2}",
                      [(set GR64:$dst, (and GR64:$src1, i64hh16c:$src2))]>;

def AND32ri     : RILI<0xC0B,
                       (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                       "nilf\t{$dst, $src2}",
                       [(set GR32:$dst, (and GR32:$src1, imm:$src2))]>;
def AND64rilo32 : RILI<0xC0B,
                       (outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                       "nilf\t{$dst, $src2}",
                       [(set GR64:$dst, (and GR64:$src1, i64lo32c:$src2))]>;
def AND64rihi32 : RILI<0xC0A,
                       (outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                       "nihf\t{$dst, $src2}",
                       [(set GR64:$dst, (and GR64:$src1, i64hi32c:$src2))]>;

let isCommutable = 1 in { // X = OR Y, Z  == X = OR Z, Y
def OR32rr : RRI<0x16,
                 (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                 "or\t{$dst, $src2}",
                 [(set GR32:$dst, (or GR32:$src1, GR32:$src2))]>;
def OR64rr : RREI<0xB981,
                  (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                  "ogr\t{$dst, $src2}",
                  [(set GR64:$dst, (or GR64:$src1, GR64:$src2))]>;
}

def OR32rm   : RXI<0x56, (outs GR32:$dst), (ins GR32:$src1, rriaddr12:$src2),
                   "o\t{$dst, $src2}",
                   [(set GR32:$dst, (or GR32:$src1, (load rriaddr12:$src2))),
                    (implicit PSW)]>;
def OR32rmy  : RXYI<0xE356, (outs GR32:$dst), (ins GR32:$src1, rriaddr:$src2),
                    "oy\t{$dst, $src2}",
                    [(set GR32:$dst, (or GR32:$src1, (load rriaddr:$src2))),
                     (implicit PSW)]>;
def OR64rm   : RXYI<0xE381, (outs GR64:$dst), (ins GR64:$src1, rriaddr:$src2),
                    "og\t{$dst, $src2}",
                    [(set GR64:$dst, (or GR64:$src1, (load rriaddr:$src2))),
                     (implicit PSW)]>;

 // FIXME: Provide proper encoding!
def OR32ri16  : RII<0xA5B,
                    (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                    "oill\t{$dst, $src2}",
                    [(set GR32:$dst, (or GR32:$src1, i32ll16:$src2))]>;
def OR32ri16h : RII<0xA5A,
                    (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                    "oilh\t{$dst, $src2}",
                    [(set GR32:$dst, (or GR32:$src1, i32lh16:$src2))]>;
def OR32ri : RILI<0xC0D,
                  (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                  "oilf\t{$dst, $src2}",
                  [(set GR32:$dst, (or GR32:$src1, imm:$src2))]>;

def OR64rill16 : RII<0xA5B,
                     (outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                     "oill\t{$dst, $src2}",
                     [(set GR64:$dst, (or GR64:$src1, i64ll16:$src2))]>;
def OR64rilh16 : RII<0xA5A,
                     (outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                     "oilh\t{$dst, $src2}",
                     [(set GR64:$dst, (or GR64:$src1, i64lh16:$src2))]>;
def OR64rihl16 : RII<0xA59,
                     (outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                     "oihl\t{$dst, $src2}",
                     [(set GR64:$dst, (or GR64:$src1, i64hl16:$src2))]>;
def OR64rihh16 : RII<0xA58,
                     (outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                     "oihh\t{$dst, $src2}",
                     [(set GR64:$dst, (or GR64:$src1, i64hh16:$src2))]>;

def OR64rilo32 : RILI<0xC0D,
                      (outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                      "oilf\t{$dst, $src2}",
                      [(set GR64:$dst, (or GR64:$src1, i64lo32:$src2))]>;
def OR64rihi32 : RILI<0xC0C,
                      (outs GR64:$dst), (ins GR64:$src1, i64imm:$src2),
                      "oihf\t{$dst, $src2}",
                      [(set GR64:$dst, (or GR64:$src1, i64hi32:$src2))]>;

def SUB32rr : RRI<0x1B,
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "sr\t{$dst, $src2}",
                  [(set GR32:$dst, (sub GR32:$src1, GR32:$src2))]>;
def SUB64rr : RREI<0xB909,
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "sgr\t{$dst, $src2}",
                   [(set GR64:$dst, (sub GR64:$src1, GR64:$src2))]>;

def SUB32rm   : RXI<0x5B, (outs GR32:$dst), (ins GR32:$src1, rriaddr12:$src2),
                    "s\t{$dst, $src2}",
                    [(set GR32:$dst, (sub GR32:$src1, (load rriaddr12:$src2))),
                     (implicit PSW)]>;
def SUB32rmy  : RXYI<0xE35B, (outs GR32:$dst), (ins GR32:$src1, rriaddr:$src2),
                     "sy\t{$dst, $src2}",
                     [(set GR32:$dst, (sub GR32:$src1, (load rriaddr:$src2))),
                      (implicit PSW)]>;
def SUB64rm   : RXYI<0xE309, (outs GR64:$dst), (ins GR64:$src1, rriaddr:$src2),
                     "sg\t{$dst, $src2}",
                     [(set GR64:$dst, (sub GR64:$src1, (load rriaddr:$src2))),
                      (implicit PSW)]>;
 
def SBC32rr : RRI<0x1F,
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "slr\t{$dst, $src2}",
                  [(set GR32:$dst, (subc GR32:$src1, GR32:$src2))]>;
def SBC64rr : RREI<0xB90B,
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "slgr\t{$dst, $src2}",
                   [(set GR64:$dst, (subc GR64:$src1, GR64:$src2))]>;

def SBC32ri   : RILI<0xC25,
                     (outs GR32:$dst), (ins GR32:$src1, s32imm:$src2),
                     "sllfi\t{$dst, $src2}",
                     [(set GR32:$dst, (subc GR32:$src1, imm:$src2))]>;
def SBC64ri32 : RILI<0xC24,
                     (outs GR64:$dst), (ins GR64:$src1, s32imm64:$src2),
                     "slgfi\t{$dst, $src2}",
                     [(set GR64:$dst, (subc GR64:$src1, immSExt32:$src2))]>;

let Uses = [PSW] in {
def SUBE32rr : RREI<0xB999, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                    "slbr\t{$dst, $src2}",
                    [(set GR32:$dst, (sube GR32:$src1, GR32:$src2)),
                     (implicit PSW)]>;
def SUBE64rr : RREI<0xB989, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                    "slbgr\t{$dst, $src2}",
                    [(set GR64:$dst, (sube GR64:$src1, GR64:$src2)),
                     (implicit PSW)]>;
}

let isCommutable = 1 in { // X = XOR Y, Z  == X = XOR Z, Y
def XOR32rr : RRI<0x17,
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "xr\t{$dst, $src2}",
                  [(set GR32:$dst, (xor GR32:$src1, GR32:$src2))]>;
def XOR64rr : RREI<0xB982,
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "xgr\t{$dst, $src2}",
                   [(set GR64:$dst, (xor GR64:$src1, GR64:$src2))]>;
}

def XOR32rm   : RXI<0x57,(outs GR32:$dst), (ins GR32:$src1, rriaddr12:$src2),
                    "x\t{$dst, $src2}",
                    [(set GR32:$dst, (xor GR32:$src1, (load rriaddr12:$src2))),
                     (implicit PSW)]>;
def XOR32rmy  : RXYI<0xE357, (outs GR32:$dst), (ins GR32:$src1, rriaddr:$src2),
                     "xy\t{$dst, $src2}",
                     [(set GR32:$dst, (xor GR32:$src1, (load rriaddr:$src2))),
                      (implicit PSW)]>;
def XOR64rm   : RXYI<0xE382, (outs GR64:$dst), (ins GR64:$src1, rriaddr:$src2),
                     "xg\t{$dst, $src2}",
                     [(set GR64:$dst, (xor GR64:$src1, (load rriaddr:$src2))),
                      (implicit PSW)]>;

def XOR32ri : RILI<0xC07,
                   (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                   "xilf\t{$dst, $src2}",
                   [(set GR32:$dst, (xor GR32:$src1, imm:$src2))]>;

} // Defs = [PSW]

let isCommutable = 1 in { // X = MUL Y, Z == X = MUL Z, Y
def MUL32rr : RREI<0xB252,
                   (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                   "msr\t{$dst, $src2}",
                   [(set GR32:$dst, (mul GR32:$src1, GR32:$src2))]>;
def MUL64rr : RREI<0xB90C,
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "msgr\t{$dst, $src2}",
                   [(set GR64:$dst, (mul GR64:$src1, GR64:$src2))]>;
}

def MUL64rrP   : RRI<0x1C,
                     (outs GR64P:$dst), (ins GR64P:$src1, GR32:$src2),
                     "mr\t{$dst, $src2}",
                     []>;
def UMUL64rrP  : RREI<0xB996,
                      (outs GR64P:$dst), (ins GR64P:$src1, GR32:$src2),
                      "mlr\t{$dst, $src2}",
                      []>;
def UMUL128rrP : RREI<0xB986,
                      (outs GR128:$dst), (ins GR128:$src1, GR64:$src2),
                      "mlgr\t{$dst, $src2}",
                      []>;

def MUL32ri16   : RII<0xA7C,
                      (outs GR32:$dst), (ins GR32:$src1, s16imm:$src2),
                      "mhi\t{$dst, $src2}",
                      [(set GR32:$dst, (mul GR32:$src1, i32immSExt16:$src2))]>;
def MUL64ri16   : RII<0xA7D,
                      (outs GR64:$dst), (ins GR64:$src1, s16imm64:$src2),
                      "mghi\t{$dst, $src2}",
                      [(set GR64:$dst, (mul GR64:$src1, immSExt16:$src2))]>;

let AddedComplexity = 2 in {
def MUL32ri     : RILI<0xC21,
                       (outs GR32:$dst), (ins GR32:$src1, s32imm:$src2),
                       "msfi\t{$dst, $src2}",
                       [(set GR32:$dst, (mul GR32:$src1, imm:$src2))]>,
                       Requires<[IsZ10]>;
def MUL64ri32   : RILI<0xC20,
                       (outs GR64:$dst), (ins GR64:$src1, s32imm64:$src2),
                       "msgfi\t{$dst, $src2}",
                       [(set GR64:$dst, (mul GR64:$src1, i64immSExt32:$src2))]>,
                       Requires<[IsZ10]>;
}

def MUL32rm : RXI<0x71,
                  (outs GR32:$dst), (ins GR32:$src1, rriaddr12:$src2),
                  "ms\t{$dst, $src2}",
                  [(set GR32:$dst, (mul GR32:$src1, (load rriaddr12:$src2)))]>;
def MUL32rmy : RXYI<0xE351,
                    (outs GR32:$dst), (ins GR32:$src1, rriaddr:$src2),
                    "msy\t{$dst, $src2}",
                    [(set GR32:$dst, (mul GR32:$src1, (load rriaddr:$src2)))]>;
def MUL64rm  : RXYI<0xE30C,
                    (outs GR64:$dst), (ins GR64:$src1, rriaddr:$src2),
                    "msg\t{$dst, $src2}",
                    [(set GR64:$dst, (mul GR64:$src1, (load rriaddr:$src2)))]>;

def MULSX64rr32 : RREI<0xB91C,
                       (outs GR64:$dst), (ins GR64:$src1, GR32:$src2),
                       "msgfr\t{$dst, $src2}",
                       [(set GR64:$dst, (mul GR64:$src1, (sext GR32:$src2)))]>;

def SDIVREM32r : RREI<0xB91D,
                      (outs GR128:$dst), (ins GR128:$src1, GR32:$src2),
                      "dsgfr\t{$dst, $src2}",
                      []>;
def SDIVREM64r : RREI<0xB90D,
                      (outs GR128:$dst), (ins GR128:$src1, GR64:$src2),
                      "dsgr\t{$dst, $src2}",
                      []>;

def UDIVREM32r : RREI<0xB997,
                      (outs GR64P:$dst), (ins GR64P:$src1, GR32:$src2),
                      "dlr\t{$dst, $src2}",
                      []>;
def UDIVREM64r : RREI<0xB987,
                      (outs GR128:$dst), (ins GR128:$src1, GR64:$src2),
                      "dlgr\t{$dst, $src2}",
                      []>;
let mayLoad = 1 in {
def SDIVREM32m : RXYI<0xE31D,
                      (outs GR128:$dst), (ins GR128:$src1, rriaddr:$src2),
                      "dsgf\t{$dst, $src2}",
                      []>;
def SDIVREM64m : RXYI<0xE30D,
                      (outs GR128:$dst), (ins GR128:$src1, rriaddr:$src2),
                      "dsg\t{$dst, $src2}",
                      []>;

def UDIVREM32m : RXYI<0xE397, (outs GR64P:$dst), (ins GR64P:$src1, rriaddr:$src2),
                      "dl\t{$dst, $src2}",
                      []>;
def UDIVREM64m : RXYI<0xE387, (outs GR128:$dst), (ins GR128:$src1, rriaddr:$src2),
                      "dlg\t{$dst, $src2}",
                      []>;
} // mayLoad
} // isTwoAddress = 1

//===----------------------------------------------------------------------===//
// Shifts

let isTwoAddress = 1 in
def SRL32rri : RSI<0x88,
                   (outs GR32:$dst), (ins GR32:$src, riaddr32:$amt),
                   "srl\t{$src, $amt}",
                   [(set GR32:$dst, (srl GR32:$src, riaddr32:$amt))]>;
def SRL64rri : RSYI<0xEB0C,
                    (outs GR64:$dst), (ins GR64:$src, riaddr:$amt),
                    "srlg\t{$dst, $src, $amt}",
                    [(set GR64:$dst, (srl GR64:$src, riaddr:$amt))]>;

let isTwoAddress = 1 in
def SHL32rri : RSI<0x89,
                   (outs GR32:$dst), (ins GR32:$src, riaddr32:$amt),
                   "sll\t{$src, $amt}",
                   [(set GR32:$dst, (shl GR32:$src, riaddr32:$amt))]>;
def SHL64rri : RSYI<0xEB0D,
                    (outs GR64:$dst), (ins GR64:$src, riaddr:$amt),
                    "sllg\t{$dst, $src, $amt}",
                    [(set GR64:$dst, (shl GR64:$src, riaddr:$amt))]>;

let Defs = [PSW] in {
let isTwoAddress = 1 in
def SRA32rri : RSI<0x8A,
                   (outs GR32:$dst), (ins GR32:$src, riaddr32:$amt),
                   "sra\t{$src, $amt}",
                   [(set GR32:$dst, (sra GR32:$src, riaddr32:$amt)),
                    (implicit PSW)]>;

def SRA64rri : RSYI<0xEB0A,
                    (outs GR64:$dst), (ins GR64:$src, riaddr:$amt),
                    "srag\t{$dst, $src, $amt}",
                    [(set GR64:$dst, (sra GR64:$src, riaddr:$amt)),
                     (implicit PSW)]>;
} // Defs = [PSW]

def ROTL32rri : RSYI<0xEB1D,
                     (outs GR32:$dst), (ins GR32:$src, riaddr32:$amt),
                     "rll\t{$dst, $src, $amt}",
                     [(set GR32:$dst, (rotl GR32:$src, riaddr32:$amt))]>;
def ROTL64rri : RSYI<0xEB1C,
                     (outs GR64:$dst), (ins GR64:$src, riaddr:$amt),
                     "rllg\t{$dst, $src, $amt}",
                     [(set GR64:$dst, (rotl GR64:$src, riaddr:$amt))]>;

//===----------------------------------------------------------------------===//
// Test instructions (like AND but do not produce any result)

// Integer comparisons
let Defs = [PSW] in {
def CMP32rr : RRI<0x19,
                  (outs), (ins GR32:$src1, GR32:$src2),
                  "cr\t$src1, $src2",
                  [(SystemZcmp GR32:$src1, GR32:$src2), 
                   (implicit PSW)]>;
def CMP64rr : RREI<0xB920,
                   (outs), (ins GR64:$src1, GR64:$src2),
                   "cgr\t$src1, $src2",
                   [(SystemZcmp GR64:$src1, GR64:$src2), 
                    (implicit PSW)]>;

def CMP32ri   : RILI<0xC2D,
                     (outs), (ins GR32:$src1, s32imm:$src2),
                     "cfi\t$src1, $src2",
                     [(SystemZcmp GR32:$src1, imm:$src2), 
                      (implicit PSW)]>;
def CMP64ri32 : RILI<0xC2C,
                     (outs), (ins GR64:$src1, s32imm64:$src2),
                     "cgfi\t$src1, $src2",
                     [(SystemZcmp GR64:$src1, i64immSExt32:$src2),
                      (implicit PSW)]>;

def CMP32rm : RXI<0x59,
                  (outs), (ins GR32:$src1, rriaddr12:$src2),
                  "c\t$src1, $src2",
                  [(SystemZcmp GR32:$src1, (load rriaddr12:$src2)),
                   (implicit PSW)]>;
def CMP32rmy : RXYI<0xE359,
                    (outs), (ins GR32:$src1, rriaddr:$src2),
                    "cy\t$src1, $src2",
                    [(SystemZcmp GR32:$src1, (load rriaddr:$src2)),
                     (implicit PSW)]>;
def CMP64rm  : RXYI<0xE320,
                    (outs), (ins GR64:$src1, rriaddr:$src2),
                    "cg\t$src1, $src2",
                    [(SystemZcmp GR64:$src1, (load rriaddr:$src2)),
                     (implicit PSW)]>;

def UCMP32rr : RRI<0x15,
                   (outs), (ins GR32:$src1, GR32:$src2),
                   "clr\t$src1, $src2",
                   [(SystemZucmp GR32:$src1, GR32:$src2),
                    (implicit PSW)]>;
def UCMP64rr : RREI<0xB921,
                    (outs), (ins GR64:$src1, GR64:$src2),
                    "clgr\t$src1, $src2",
                    [(SystemZucmp GR64:$src1, GR64:$src2), 
                     (implicit PSW)]>;

def UCMP32ri   : RILI<0xC2F,
                      (outs), (ins GR32:$src1, i32imm:$src2),
                      "clfi\t$src1, $src2",
                      [(SystemZucmp GR32:$src1, imm:$src2),
                       (implicit PSW)]>;
def UCMP64ri32 : RILI<0xC2E,
                      (outs), (ins GR64:$src1, i64i32imm:$src2),
                      "clgfi\t$src1, $src2",
                      [(SystemZucmp GR64:$src1, i64immZExt32:$src2),
                       (implicit PSW)]>;

def UCMP32rm  : RXI<0x55,
                    (outs), (ins GR32:$src1, rriaddr12:$src2),
                    "cl\t$src1, $src2",
                    [(SystemZucmp GR32:$src1, (load rriaddr12:$src2)),
                     (implicit PSW)]>;
def UCMP32rmy : RXYI<0xE355,
                     (outs), (ins GR32:$src1, rriaddr:$src2),
                     "cly\t$src1, $src2",
                     [(SystemZucmp GR32:$src1, (load rriaddr:$src2)),
                      (implicit PSW)]>;
def UCMP64rm  : RXYI<0xE351,
                     (outs), (ins GR64:$src1, rriaddr:$src2),
                     "clg\t$src1, $src2",
                     [(SystemZucmp GR64:$src1, (load rriaddr:$src2)),
                      (implicit PSW)]>;

def CMPSX64rr32  : RREI<0xB930,
                        (outs), (ins GR64:$src1, GR32:$src2),
                        "cgfr\t$src1, $src2",
                        [(SystemZucmp GR64:$src1, (sext GR32:$src2)),
                         (implicit PSW)]>;
def UCMPZX64rr32 : RREI<0xB931,
                        (outs), (ins GR64:$src1, GR32:$src2),
                        "clgfr\t$src1, $src2",
                        [(SystemZucmp GR64:$src1, (zext GR32:$src2)),
                         (implicit PSW)]>;

def CMPSX64rm32   : RXYI<0xE330,
                         (outs), (ins GR64:$src1, rriaddr:$src2),
                         "cgf\t$src1, $src2",
                         [(SystemZucmp GR64:$src1, (sextloadi64i32 rriaddr:$src2)),
                          (implicit PSW)]>;
def UCMPZX64rm32  : RXYI<0xE331,
                         (outs), (ins GR64:$src1, rriaddr:$src2),
                         "clgf\t$src1, $src2",
                         [(SystemZucmp GR64:$src1, (zextloadi64i32 rriaddr:$src2)),
                          (implicit PSW)]>;

// FIXME: Add other crazy ucmp forms

} // Defs = [PSW]

//===----------------------------------------------------------------------===//
// Other crazy stuff
let Defs = [PSW] in {
def FLOGR64 : RREI<0xB983,
                   (outs GR128:$dst), (ins GR64:$src),
                   "flogr\t{$dst, $src}",
                   []>;
} // Defs = [PSW]

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns.
//===----------------------------------------------------------------------===//

// ConstPools, JumpTables
def : Pat<(SystemZpcrelwrapper tjumptable:$src), (LA64rm tjumptable:$src)>;
def : Pat<(SystemZpcrelwrapper tconstpool:$src), (LA64rm tconstpool:$src)>;

// anyext
def : Pat<(i64 (anyext GR32:$src)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GR32:$src, subreg_32bit)>;

// calls
def : Pat<(SystemZcall (i64 tglobaladdr:$dst)), (CALLi tglobaladdr:$dst)>;
def : Pat<(SystemZcall (i64 texternalsym:$dst)), (CALLi texternalsym:$dst)>;

//===----------------------------------------------------------------------===//
// Peepholes.
//===----------------------------------------------------------------------===//

// FIXME: use add/sub tricks with 32678/-32768

// Arbitrary immediate support.
def : Pat<(i32 imm:$src),
          (EXTRACT_SUBREG (MOV64ri32 (i64 imm:$src)), subreg_32bit)>;

// Implement in terms of LLIHF/OILF.
def : Pat<(i64 imm:$imm),
          (OR64rilo32 (MOV64rihi32 (HI32 imm:$imm)), (LO32 imm:$imm))>;

// trunc patterns
def : Pat<(i32 (trunc GR64:$src)),
          (EXTRACT_SUBREG GR64:$src, subreg_32bit)>;

// sext_inreg patterns
def : Pat<(sext_inreg GR64:$src, i32),
          (MOVSX64rr32 (EXTRACT_SUBREG GR64:$src, subreg_32bit))>;

// extload patterns
def : Pat<(extloadi32i8  rriaddr:$src), (MOVZX32rm8  rriaddr:$src)>;
def : Pat<(extloadi32i16 rriaddr:$src), (MOVZX32rm16 rriaddr:$src)>;
def : Pat<(extloadi64i8  rriaddr:$src), (MOVZX64rm8  rriaddr:$src)>;
def : Pat<(extloadi64i16 rriaddr:$src), (MOVZX64rm16 rriaddr:$src)>;
def : Pat<(extloadi64i32 rriaddr:$src), (MOVZX64rm32 rriaddr:$src)>;

// muls
def : Pat<(mulhs GR32:$src1, GR32:$src2),
          (EXTRACT_SUBREG (MUL64rrP (INSERT_SUBREG (v2i32 (IMPLICIT_DEF)),
                                                   GR32:$src1, subreg_odd32),
                                    GR32:$src2),
                          subreg_even32)>;

def : Pat<(mulhu GR32:$src1, GR32:$src2),
          (EXTRACT_SUBREG (UMUL64rrP (INSERT_SUBREG (v2i32 (IMPLICIT_DEF)),
                                                    GR32:$src1, subreg_odd32),
                                     GR32:$src2),
                          subreg_even32)>;
def : Pat<(mulhu GR64:$src1, GR64:$src2),
          (EXTRACT_SUBREG (UMUL128rrP (INSERT_SUBREG (v2i64 (IMPLICIT_DEF)),
                                                     GR64:$src1, subreg_odd),
                                      GR64:$src2),
                          subreg_even)>;

def : Pat<(ctlz GR64:$src),
          (EXTRACT_SUBREG (FLOGR64 GR64:$src), subreg_even)>;
