; RUN: llc < %s -mtriple=x86_64 -mcpu=skx | FileCheck %s --check-prefixes=SKX

define <16 x i32> @test_v16f64_ogt(<16 x i32> %a, <16 x i32> %b, <16 x double> %f1, <16 x double> %f2) #0 {
; SKX-LABEL: test_v16f64_ogt
; SKX:       # %bb.0:
; SKX-NEXT:  pushq	%rbp
; SKX-NEXT:  movq	%rsp, %rbp
; SKX-NEXT:  andq	$-32, %rsp
; SKX-NEXT:  subq	$32, %rsp
; SKX-NEXT:  vcmpgtpd	80(%rbp), %ymm6, %k0
; SKX-NEXT:  vcmpgtpd	112(%rbp), %ymm7, %k1
; SKX-NEXT:  kshiftlb	$4, %k1, %k1
; SKX-NEXT:  korb	%k1, %k0, %k1
; SKX-NEXT:  vcmpgtpd	16(%rbp), %ymm4, %k0
; SKX-NEXT:  vcmpgtpd	48(%rbp), %ymm5, %k2
; SKX-NEXT:  kshiftlb	$4, %k2, %k2
; SKX-NEXT:  korb	%k2, %k0, %k2
; SKX-NEXT:  vpblendmd	%ymm0, %ymm2, %ymm0 {%k2}
; SKX-NEXT:  vpblendmd	%ymm1, %ymm3, %ymm1 {%k1}
; SKX-NEXT:  movq	%rbp, %rsp
; SKX-NEXT:  popq	%rbp
; SKX-NEXT:  retq
  %cond = tail call <16 x i1> @llvm.experimental.constrained.fcmps.v16f64(
    <16 x double> %f1, <16 x double> %f2, metadata !"ogt", metadata !"fpexcept.maytrap")
  %res = select <16 x i1> %cond, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %res
}

declare <16 x i1> @llvm.experimental.constrained.fcmps.v16f64(<16 x double>, <16 x double>, metadata, metadata)

attributes #0 = { nounwind strictfp "min-legal-vector-width"="0" }
