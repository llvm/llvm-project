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

let SchedRW = [WriteSystem], Predicates = [HasKL] in {
  let Uses = [XMM0, EAX], Defs = [EFLAGS] in {
    def LOADIWKEY : I<0xDC, MRMSrcReg, (outs), (ins VR128:$src1, VR128:$src2),
                      "loadiwkey\t{$src2, $src1|$src1, $src2}",
                      [(int_x86_loadiwkey XMM0, VR128:$src1, VR128:$src2, EAX)]>, T8, XS;
  }

  let Uses = [XMM0], Defs = [XMM0, XMM1, XMM2, XMM4, XMM5, XMM6, EFLAGS] in {
    def ENCODEKEY128 : I<0xFA, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                         "encodekey128\t{$src, $dst|$dst, $src}", []>, T8, XS;
  }

  let Uses = [XMM0, XMM1], Defs = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, EFLAGS] in {
    def ENCODEKEY256 : I<0xFB, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                         "encodekey256\t{$src, $dst|$dst, $src}", []>, T8, XS;
  }

  let Constraints = "$src1 = $dst",
      Defs = [EFLAGS] in {
   def AESENC128KL : I<0xDC, MRMSrcMem, (outs VR128:$dst), (ins VR128:$src1, opaquemem:$src2),
                       "aesenc128kl\t{$src2, $src1|$src1, $src2}",
                       [(set VR128:$dst, EFLAGS,
                         (X86aesenc128kl VR128:$src1, addr:$src2))]>, T8, XS;

   def AESDEC128KL : I<0xDD, MRMSrcMem, (outs VR128:$dst), (ins VR128:$src1, opaquemem:$src2),
                       "aesdec128kl\t{$src2, $src1|$src1, $src2}",
                       [(set VR128:$dst, EFLAGS,
                         (X86aesdec128kl VR128:$src1, addr:$src2))]>, T8, XS;

   def AESENC256KL : I<0xDE, MRMSrcMem, (outs VR128:$dst), (ins VR128:$src1, opaquemem:$src2),
                       "aesenc256kl\t{$src2, $src1|$src1, $src2}",
                       [(set VR128:$dst, EFLAGS,
                         (X86aesenc256kl VR128:$src1, addr:$src2))]>, T8, XS;

   def AESDEC256KL : I<0xDF, MRMSrcMem, (outs VR128:$dst), (ins VR128:$src1, opaquemem:$src2),
                       "aesdec256kl\t{$src2, $src1|$src1, $src2}",
                       [(set VR128:$dst, EFLAGS,
                         (X86aesdec256kl VR128:$src1, addr:$src2))]>, T8, XS;
  }

} // SchedRW, Predicates

let SchedRW = [WriteSystem], Predicates = [HasWIDEKL] in {
  let Uses = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7],
      Defs = [EFLAGS, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7],
      mayLoad = 1 in {
    def AESENCWIDE128KL : I<0xD8, MRM0m, (outs), (ins opaquemem:$src),
                            "aesencwide128kl\t$src", []>, T8, XS;
    def AESDECWIDE128KL : I<0xD8, MRM1m, (outs), (ins opaquemem:$src),
                            "aesdecwide128kl\t$src", []>, T8, XS;
    def AESENCWIDE256KL : I<0xD8, MRM2m, (outs), (ins opaquemem:$src),
                            "aesencwide256kl\t$src", []>, T8, XS;
    def AESDECWIDE256KL : I<0xD8, MRM3m, (outs), (ins opaquemem:$src),
                            "aesdecwide256kl\t$src", []>, T8, XS;
  }

} // SchedRW, Predicates
