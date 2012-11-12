//===-- X86InstrVMX.td - TSX Instruction Set Extension -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the instructions that make up the Intel TSX instruction
// set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TSX instructions

let usesCustomInserter = 1 in
def XBEGIN : I<0, Pseudo, (outs GR32:$dst), (ins),
               "# XBEGIN", [(set GR32:$dst, (int_x86_xbegin))]>,
             Requires<[HasRTM]>;

let isBranch = 1, isTerminator = 1, Defs = [EAX] in
def XBEGIN_4 : Ii32PCRel<0xc7, MRM_F8, (outs), (ins brtarget:$dst),
                         "xbegin\t$dst", []>;

def XEND : I<0x01, MRM_D5, (outs), (ins),
             "xend", [(int_x86_xend)]>, TB, Requires<[HasRTM]>;

def XABORT : Ii8<0xc6, MRM_F8, (outs), (ins i8imm:$imm),
                 "xabort\t$imm",
                 [(int_x86_xabort imm:$imm)]>, Requires<[HasRTM]>;
