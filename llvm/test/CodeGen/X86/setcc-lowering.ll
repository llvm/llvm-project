; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+avx < %s | FileCheck %s

; Verify that we don't crash during codegen due to a wrong lowering
; of a setcc node with illegal operand types and return type.

define <8 x i16> @pr25080(<8 x i32> %a) {
; CHECK-LABEL: pr25080:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    vandps {{.*}}(%rip), %ymm0, %ymm0
; CHECK-NEXT:    vextractf128 $1, %ymm0, %xmm1
; CHECK-NEXT:    vpxor %xmm2, %xmm2, %xmm2
; CHECK-NEXT:    vpcmpeqd %xmm2, %xmm1, %xmm1
; CHECK-NEXT:    vmovdqa {{.*#+}} xmm3 = [0,1,4,5,8,9,12,13,8,9,12,13,12,13,14,15]
; CHECK-NEXT:    vpshufb %xmm3, %xmm1, %xmm1
; CHECK-NEXT:    vpcmpeqd %xmm2, %xmm0, %xmm0
; CHECK-NEXT:    vpshufb %xmm3, %xmm0, %xmm0
; CHECK-NEXT:    vpunpcklqdq {{.*#+}} xmm0 = xmm0[0],xmm1[0]
; CHECK-NEXT:    vpor {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    vpsllw $15, %xmm0, %xmm0
; CHECK-NEXT:    vpsraw $15, %xmm0, %xmm0
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
entry:
  %0 = trunc <8 x i32> %a to <8 x i23>
  %1 = icmp eq <8 x i23> %0, zeroinitializer
  %2 = or <8 x i1> %1, <i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false>
  %3 = sext <8 x i1> %2 to <8 x i16>
  ret <8 x i16> %3
}
