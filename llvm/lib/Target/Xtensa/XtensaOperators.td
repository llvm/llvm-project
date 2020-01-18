//===- XtensaOperators.td - Xtensa-specific operators ---------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===-------------------------------------------------------------------===//
// Type profiles
//===-------------------------------------------------------------------===//

def SDT_XtensaCallSeqStart        : SDCallSeqStart<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDT_XtensaCallSeqEnd          : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDT_XtensaCall                : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;
def SDT_XtensaWrapPtr             : SDTypeProfile<1, 1,
                                                 [SDTCisSameAs<0, 1>,
                                                  SDTCisPtrTy<0>]>;
def SDT_XtensaSelectCC            : SDTypeProfile<1, 5,
                                                 [SDTCisSameAs<0, 1>,
                                                  SDTCisSameAs<2, 3>, 
                                                  SDTCisVT<5, i32>]>;
def SDT_XtensaMOVSP               : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>, SDTCisVT<0, i32>]>;
def SDT_XtensaBrCC               : SDTypeProfile<0, 2, [SDTCisVT<0, i1>, SDTCisVT<1, OtherVT>]>;
def SDT_XtensaCmp                 : SDTypeProfile<1, 2, [SDTCisVT<0, i1>, SDTCisVT<1, f32>, SDTCisVT<2, f32>]>;
def SDT_XtensaMADD               : SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>, SDTCisVT<0, f32>]>;
def SDT_XtensaMOVS               : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>, SDTCisVT<0, f32>]>;
def SDT_XtensaSelectCCFP         : SDTypeProfile<1, 5, [SDTCisSameAs<0, 3>, SDTCisSameAs<1, 2>, SDTCisSameAs<3, 4>, SDTCisVT<5, i32>]>;
def SDT_XtensaBrJT               : SDTypeProfile<0, 2, [SDTCisPtrTy<0>, SDTCisVT<1, i32>]>;
def SDT_XtensaSHL                : SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDT_XtensaSRA                : SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDT_XtensaSRL                : SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDT_XtensaSRC                : SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisVT<1, i32>, SDTCisVT<2, i32>]>;
def SDT_XtensaSSL                : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;
def SDT_XtensaSSR                : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;
def SDT_XtensaMEMBARRIER         : SDTypeProfile<0, 0, []>;
def SDT_XtensaRUR                : SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;

//===-------------------------------------------------------------------===//
// Node definitions
//===-------------------------------------------------------------------===//

def Xtensa_call: SDNode<"XtensaISD::CALL", SDT_XtensaCall,
                       [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue, SDNPVariadic]>;

def Xtensa_retflag: SDNode<"XtensaISD::RET_FLAG", SDTNone,
                          [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def Xtensa_retWflag: SDNode<"XtensaISD::RETW_FLAG", SDTNone,
                          [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
	 
def Xtensa_callseq_start: SDNode<"ISD::CALLSEQ_START", SDT_XtensaCallSeqStart,
                                [SDNPHasChain, SDNPSideEffect, SDNPOutGlue]>;
def Xtensa_callseq_end  : SDNode<"ISD::CALLSEQ_END",   SDT_XtensaCallSeqEnd,
                                [SDNPHasChain, SDNPSideEffect, SDNPOptInGlue,
                                 SDNPOutGlue]>;

def Xtensa_pcrel_wrapper: SDNode<"XtensaISD::PCREL_WRAPPER", SDT_XtensaWrapPtr, []>;

def Xtensa_select : SDNode<"XtensaISD::SELECT", SDTSelect>;
def Xtensa_select_cc: SDNode<"XtensaISD::SELECT_CC", SDT_XtensaSelectCC,
                            [SDNPInGlue]>;

def Xtensa_movsp: SDNode<"XtensaISD::MOVSP", SDT_XtensaMOVSP,
                            [SDNPInGlue]>;

def Xtensa_brcc_t    : SDNode<"XtensaISD::BR_CC_T", SDT_XtensaBrCC,
                            [SDNPHasChain, SDNPInGlue]>;
def Xtensa_brcc_f    : SDNode<"XtensaISD::BR_CC_F", SDT_XtensaBrCC,
                            [SDNPHasChain, SDNPInGlue]>;

def Xtensa_cmpoeq     : SDNode<"XtensaISD::CMPOEQ", SDT_XtensaCmp, [SDNPOutGlue]>;
def Xtensa_cmpolt     : SDNode<"XtensaISD::CMPOLT", SDT_XtensaCmp, [SDNPOutGlue]>;
def Xtensa_cmpole     : SDNode<"XtensaISD::CMPOLE", SDT_XtensaCmp, [SDNPOutGlue]>;
def Xtensa_cmpueq     : SDNode<"XtensaISD::CMPUEQ", SDT_XtensaCmp, [SDNPOutGlue]>;
def Xtensa_cmpult     : SDNode<"XtensaISD::CMPULT", SDT_XtensaCmp, [SDNPOutGlue]>;
def Xtensa_cmpule     : SDNode<"XtensaISD::CMPULE", SDT_XtensaCmp, [SDNPOutGlue]>;
def Xtensa_cmpuo      : SDNode<"XtensaISD::CMPUO", SDT_XtensaCmp, [SDNPOutGlue]>;

def Xtensa_madd: SDNode<"XtensaISD::MADD", SDT_XtensaMADD,
                            [SDNPInGlue]>;
def Xtensa_msub: SDNode<"XtensaISD::MSUB", SDT_XtensaMADD,
                            [SDNPInGlue]>;
def Xtensa_movs: SDNode<"XtensaISD::MOVS", SDT_XtensaMOVS,
                            [SDNPInGlue]>;

def Xtensa_shl: SDNode<"XtensaISD::SHL", SDT_XtensaSHL,
                            [SDNPInGlue]>;
def Xtensa_sra: SDNode<"XtensaISD::SRA", SDT_XtensaSRA,
                            [SDNPInGlue]>;
def Xtensa_srl: SDNode<"XtensaISD::SRL", SDT_XtensaSRL,
                            [SDNPInGlue]>;
def Xtensa_src: SDNode<"XtensaISD::SRC", SDT_XtensaSRC,
                            [SDNPInGlue]>;
def Xtensa_ssl: SDNode<"XtensaISD::SSL", SDT_XtensaSSL,
                            [SDNPOutGlue]>;
def Xtensa_ssr: SDNode<"XtensaISD::SSR", SDT_XtensaSSR,
                            [SDNPOutGlue]>;
def Xtensa_select_cc_fp: SDNode<"XtensaISD::SELECT_CC_FP", SDT_XtensaSelectCCFP,
                            [SDNPInGlue]>;

def Xtensa_brjt: SDNode<"XtensaISD::BR_JT", SDT_XtensaBrJT,
                              [SDNPHasChain]>;
def Xtensa_callw: SDNode<"XtensaISD::CALLW", SDT_XtensaCall,
                     [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue, SDNPVariadic]>;
def Xtensa_mem_barrier: SDNode<"XtensaISD::MEMW", SDT_XtensaMEMBARRIER,
                             [SDNPHasChain, SDNPSideEffect]>;

def Xtensa_rur: SDNode<"XtensaISD::RUR", SDT_XtensaRUR,
                            [SDNPInGlue]>;

