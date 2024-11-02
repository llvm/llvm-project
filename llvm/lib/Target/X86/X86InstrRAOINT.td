//===---- X86InstrRAOINT.td -------------------------------*- tablegen -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the instructions that make up the Intel RAO-INT
// instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RAO-INT instructions

def SDTRAOBinaryArith : SDTypeProfile<0, 2, [SDTCisPtrTy<0>, SDTCisInt<1>]>;

def X86rao_add  : SDNode<"X86ISD::AADD", SDTRAOBinaryArith,
                         [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def X86rao_or   : SDNode<"X86ISD::AOR",  SDTRAOBinaryArith,
                         [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def X86rao_xor  : SDNode<"X86ISD::AXOR", SDTRAOBinaryArith,
                         [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def X86rao_and  : SDNode<"X86ISD::AAND", SDTRAOBinaryArith,
                         [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

multiclass RAOINT_BASE<string OpcodeStr> {
  let Predicates = [HasRAOINT] in
    def 32mr : I<0xfc, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                 !strconcat("a", OpcodeStr, "{l}\t{$src, $dst|$dst, $src}"),
                 [(!cast<SDNode>("X86rao_" # OpcodeStr) addr:$dst, GR32:$src)]>,
               Sched<[WriteALURMW]>;

  let Predicates = [HasRAOINT, In64BitMode] in
    def 64mr : I<0xfc, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                 !strconcat("a", OpcodeStr, "{q}\t{$src, $dst|$dst, $src}"),
                 [(!cast<SDNode>("X86rao_" # OpcodeStr) addr:$dst, GR64:$src)]>,
               Sched<[WriteALURMW]>, REX_W;
}

defm AADD : RAOINT_BASE<"add">, T8PS;
defm AAND : RAOINT_BASE<"and">, T8PD;
defm AOR  : RAOINT_BASE<"or" >, T8XD;
defm AXOR : RAOINT_BASE<"xor">, T8XS;
