//===---------------------------*-tablegen-*-------------------------------===//
//===------------- X86InstrKL.td - KL Instruction Set Extension -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the instructions that make up the Intel key locker
// instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Key Locker instructions
class Encodekey<bits<8> opcode, string mnemonic>
  : I<opcode, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src), mnemonic#"\t{$src, $dst|$dst, $src}", []>,
    NoCD8, XS;

class Aesencdec<bits<8> opcode, string mnemonic, SDNode node>
  : I<opcode, MRMSrcMem, (outs VR128:$dst), (ins VR128:$src1, opaquemem:$src2),
      mnemonic#"\t{$src2, $src1|$src1, $src2}",
      [(set VR128:$dst, EFLAGS, (node VR128:$src1, addr:$src2))]>, NoCD8, XS;

let SchedRW = [WriteSystem] in {
  let Uses = [XMM0, EAX], Defs = [EFLAGS], Predicates = [HasKL] in {
    def LOADIWKEY : I<0xDC, MRMSrcReg, (outs), (ins VR128:$src1, VR128:$src2),
                      "loadiwkey\t{$src2, $src1|$src1, $src2}",
                      [(int_x86_loadiwkey XMM0, VR128:$src1, VR128:$src2, EAX)]>, T8, XS;
  }

  let Predicates = [HasKL, NoEGPR] in {
  let Uses = [XMM0], Defs = [XMM0, XMM1, XMM2, XMM4, XMM5, XMM6, EFLAGS] in {
    def ENCODEKEY128 : Encodekey<0xFA, "encodekey128">, T8;
  }

  let Uses = [XMM0, XMM1], Defs = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, EFLAGS] in {
    def ENCODEKEY256 : Encodekey<0xFB, "encodekey256">, T8;
  }

  let Constraints = "$src1 = $dst", Defs = [EFLAGS], Predicates = [NoEGPR] in {
   def AESENC128KL : Aesencdec<0xDC, "aesenc128kl", X86aesenc128kl>, T8;

   def AESDEC128KL : Aesencdec<0xDD, "aesdec128kl", X86aesdec128kl>, T8;

   def AESENC256KL : Aesencdec<0xDE, "aesenc256kl", X86aesenc256kl>, T8;

   def AESDEC256KL : Aesencdec<0xDF, "aesdec256kl", X86aesdec256kl>, T8;
  }
  }

  let Predicates = [HasKL, HasEGPR, In64BitMode] in {
  let Uses = [XMM0], Defs = [XMM0, XMM1, XMM2, XMM4, XMM5, XMM6, EFLAGS] in {
    def ENCODEKEY128_EVEX : Encodekey<0xDA, "encodekey128">, EVEX, T_MAP4;
  }

  let Uses = [XMM0, XMM1], Defs = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, EFLAGS] in {
    def ENCODEKEY256_EVEX : Encodekey<0xDB, "encodekey256">, EVEX, T_MAP4;
  }

  let Constraints = "$src1 = $dst", Defs = [EFLAGS], Predicates = [HasEGPR, In64BitMode] in {
   def AESENC128KL_EVEX : Aesencdec<0xDC, "aesenc128kl", X86aesenc128kl>, EVEX, T_MAP4;

   def AESDEC128KL_EVEX : Aesencdec<0xDD, "aesdec128kl", X86aesdec128kl>, EVEX, T_MAP4;

   def AESENC256KL_EVEX : Aesencdec<0xDE, "aesenc256kl", X86aesenc256kl>, EVEX, T_MAP4;

   def AESDEC256KL_EVEX : Aesencdec<0xDF, "aesdec256kl", X86aesdec256kl>, EVEX, T_MAP4;
  }
  }
} // SchedRW

class Aesencdecwide<bits<8> opcode, Format f, string mnemonic>
  : I<opcode, f, (outs), (ins opaquemem:$src), mnemonic#"\t$src", []>, NoCD8, XS;

let SchedRW = [WriteSystem] in {
  let Uses = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7],
      Defs = [EFLAGS, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7],
      mayLoad = 1 in {
  let Predicates = [HasWIDEKL, NoEGPR] in {
    def AESENCWIDE128KL : Aesencdecwide<0xD8, MRM0m, "aesencwide128kl">, T8;
    def AESDECWIDE128KL : Aesencdecwide<0xD8, MRM1m, "aesdecwide128kl">, T8;
    def AESENCWIDE256KL : Aesencdecwide<0xD8, MRM2m, "aesencwide256kl">, T8;
    def AESDECWIDE256KL : Aesencdecwide<0xD8, MRM3m, "aesdecwide256kl">, T8;
  }

  let Predicates = [HasWIDEKL, HasEGPR, In64BitMode] in {
    def AESENCWIDE128KL_EVEX : Aesencdecwide<0xD8, MRM0m, "aesencwide128kl">, EVEX, T_MAP4;
    def AESDECWIDE128KL_EVEX : Aesencdecwide<0xD8, MRM1m, "aesdecwide128kl">, EVEX, T_MAP4;
    def AESENCWIDE256KL_EVEX : Aesencdecwide<0xD8, MRM2m, "aesencwide256kl">, EVEX, T_MAP4;
    def AESDECWIDE256KL_EVEX : Aesencdecwide<0xD8, MRM3m, "aesdecwide256kl">, EVEX, T_MAP4;
  }
  }

} // SchedRW, Predicates
