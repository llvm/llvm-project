; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

define <4 x float> @foo(<4 x float> %val, <4 x float> %test) nounwind {
; CHECK-LABEL: LCPI0_0:
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-LABEL: foo:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    cmpeqps %xmm1, %xmm0
; CHECK-NEXT:    andps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %cmp = fcmp oeq <4 x float> %val, %test
  %ext = zext <4 x i1> %cmp to <4 x i32>
  %result = sitofp <4 x i32> %ext to <4 x float>
  ret <4 x float> %result
}

; Make sure the operation doesn't try to get folded when the sizes don't match,
; as that ends up crashing later when trying to form a bitcast operation for
; the folded nodes.
define void @foo1(<4 x float> %val, <4 x float> %test, ptr %p) nounwind {
; CHECK-LABEL: LCPI1_0:
; CHECK-NEXT: .long 1                       ## 0x1
; CHECK-NEXT: .long 1                       ## 0x1
; CHECK-NEXT: .long 1                       ## 0x1
; CHECK-NEXT: .long 1                       ## 0x1
; CHECK-LABEL: foo1:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    cmpeqps %xmm1, %xmm0
; CHECK-NEXT:    andps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    pshufd {{.*#+}} xmm1 = xmm0[2,3,2,3]
; CHECK-NEXT:    cvtdq2pd %xmm1, %xmm1
; CHECK-NEXT:    cvtdq2pd %xmm0, %xmm0
; CHECK-NEXT:    movaps %xmm0, (%rdi)
; CHECK-NEXT:    movaps %xmm1, 16(%rdi)
; CHECK-NEXT:    retq
  %cmp = fcmp oeq <4 x float> %val, %test
  %ext = zext <4 x i1> %cmp to <4 x i32>
  %result = sitofp <4 x i32> %ext to <4 x double>
  store <4 x double> %result, ptr %p
  ret void
}

; Also test the general purpose constant folding of int->fp.
define void @foo2(ptr noalias %result) nounwind {
; CHECK-LABEL: LCPI2_0:
; CHECK-NEXT: .long 0x40800000              ## float 4
; CHECK-NEXT: .long 0x40a00000              ## float 5
; CHECK-NEXT: .long 0x40c00000              ## float 6
; CHECK-NEXT: .long 0x40e00000              ## float 7
; CHECK-LABEL: foo2:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    movaps {{.*#+}} xmm0 = [4.0E+0,5.0E+0,6.0E+0,7.0E+0]
; CHECK-NEXT:    movaps %xmm0, (%rdi)
; CHECK-NEXT:    retq
  %val = uitofp <4 x i32> <i32 4, i32 5, i32 6, i32 7> to <4 x float>
  store <4 x float> %val, ptr %result
  ret void
}

; Fold explicit AND operations when the constant isn't a splat of a single
; scalar value like what the zext creates.
define <4 x float> @foo3(<4 x float> %val, <4 x float> %test) nounwind {
; CHECK-LABEL: LCPI3_0:
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 0                       ## 0x0
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 0                       ## 0x0
; CHECK-LABEL: foo3:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    cmpeqps %xmm1, %xmm0
; CHECK-NEXT:    andps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %cmp = fcmp oeq <4 x float> %val, %test
  %ext = zext <4 x i1> %cmp to <4 x i32>
  %and = and <4 x i32> %ext, <i32 255, i32 256, i32 257, i32 258>
  %result = sitofp <4 x i32> %and to <4 x float>
  ret <4 x float> %result
}

; Test the general purpose constant folding of uint->fp.
define void @foo4(ptr noalias %result) nounwind {
; CHECK-LABEL: LCPI4_0:
; CHECK-NEXT: .long 0x3f800000              ## float 1
; CHECK-NEXT: .long 0x42fe0000              ## float 127
; CHECK-NEXT: .long 0x43000000              ## float 128
; CHECK-NEXT: .long 0x437f0000              ## float 255
; CHECK-LABEL: foo4:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    movaps {{.*#+}} xmm0 = [1.0E+0,1.27E+2,1.28E+2,2.55E+2]
; CHECK-NEXT:    movaps %xmm0, (%rdi)
; CHECK-NEXT:    retq
  %val = uitofp <4 x i8> <i8 1, i8 127, i8 -128, i8 -1> to <4 x float>
  store <4 x float> %val, ptr %result
  ret void
}

; Test when we're masking against a sign extended setcc.
define <4 x float> @foo5(<4 x i32> %a0, <4 x i32> %a1) {
; CHECK-LABEL: LCPI5_0:
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 0                       ## 0x0
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 0                       ## 0x0
; CHECK:       ## %bb.0:
; CHECK-NEXT:    pcmpgtd %xmm1, %xmm0
; CHECK-NEXT:    pand {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %1 = icmp sgt <4 x i32> %a0, %a1
  %2 = sext <4 x i1> %1 to <4 x i32>
  %3 = and <4 x i32> %2, <i32 1, i32 0, i32 1, i32 0>
  %4 = uitofp <4 x i32> %3 to <4 x float>
  ret <4 x float> %4
}

; Test when we're masking against mask arithmetic, not the setcc's directly.
define <4 x float> @foo6(<4 x i32> %a0, <4 x i32> %a1) {
; CHECK-LABEL: LCPI6_0:
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 0                       ## 0x0
; CHECK-NEXT: .long 1065353216              ## 0x3f800000
; CHECK-NEXT: .long 0                       ## 0x0
; CHECK:       ## %bb.0:
; CHECK-NEXT:    movdqa %xmm0, %xmm2
; CHECK-NEXT:    pcmpgtd %xmm1, %xmm2
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pcmpgtd %xmm1, %xmm0
; CHECK-NEXT:    pand %xmm2, %xmm0
; CHECK-NEXT:    pand {{.*}}(%rip), %xmm0
; CHECK-NEXT:    retq
  %1 = icmp sgt <4 x i32> %a0, %a1
  %2 = icmp sgt <4 x i32> %a0, zeroinitializer
  %3 = and <4 x i1> %1, %2
  %4 = sext <4 x i1> %3 to <4 x i32>
  %5 = and <4 x i32> %4, <i32 1, i32 0, i32 1, i32 0>
  %6 = uitofp <4 x i32> %5 to <4 x float>
  ret <4 x float> %6
}

define <4 x float> @foo7(<4 x i64> %a) {
; CHECK-LABEL: LCPI7_0:
; CHECK-NEXT:  .byte   0                       ## 0x0
; CHECK-NEXT:  .byte   255                     ## 0xff
; CHECK-NEXT:  .byte   0                       ## 0x0
; CHECK-NEXT:  .byte   0                       ## 0x0
; CHECK-NEXT:  .byte   0                       ## 0x0
; CHECK-NEXT:  .byte   255                     ## 0xff
; CHECK-NEXT:  .byte   0                       ## 0x0
; CHECK-NEXT:  .byte   0                       ## 0x0
; CHECK-NEXT:  .byte   0                       ## 0x0
; CHECK-NEXT:  .byte   255                     ## 0xff
; CHECK-NEXT:  .byte   0                       ## 0x0
; CHECK-NEXT:  .byte   0                       ## 0x0
; CHECK-NEXT:  .byte   0                       ## 0x0
; CHECK-NEXT:  .byte   255                     ## 0xff
; CHECK-NEXT:  .byte   0                       ## 0x0
; CHECK-NEXT:  .byte   0                       ## 0x0
; CHECK-LABEL: foo7:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    shufps {{.*#+}} xmm0 = xmm0[0,2],xmm1[0,2]
; CHECK-NEXT:    andps {{.*}}(%rip), %xmm0
; CHECK-NEXT:    cvtdq2ps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %b = and <4 x i64> %a, <i64 4278255360, i64 4278255360, i64 4278255360, i64 4278255360>
  %c = and <4 x i64> %b, <i64 65535, i64 65535, i64 65535, i64 65535>
  %d = uitofp <4 x i64> %c to <4 x float>
  ret <4 x float> %d
}
