; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=avx < %s | FileCheck %s

define float @test1(float %a) #0 {
; CHECK-LABEL: test1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fadd nnan reassoc nsz float %a, %a
  %r = fadd nnan reassoc nsz float %t1, %t1
  ret float %r
}

define float @test2(float %a) #0 {
; CHECK-LABEL: test2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fmul nnan reassoc nsz float 4.0, %a
  %t2 = fadd nnan reassoc nsz float %a, %a
  %r = fadd nnan reassoc nsz float %t1, %t2
  ret float %r
}

define float @test3(float %a) #0 {
; CHECK-LABEL: test3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fmul nnan reassoc nsz float %a, 4.0
  %t2 = fadd nnan reassoc nsz float %a, %a
  %r = fadd nnan reassoc nsz float %t1, %t2
  ret float %r
}

define float @test4(float %a) #0 {
; CHECK-LABEL: test4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fadd nnan reassoc nsz float %a, %a
  %t2 = fmul nnan reassoc nsz float 4.0, %a
  %r = fadd nnan reassoc nsz float %t1, %t2
  ret float %r
}

define float @test5(float %a) #0 {
; CHECK-LABEL: test5:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmulss {{.*}}(%rip), %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fadd nnan reassoc nsz float %a, %a
  %t2 = fmul nnan reassoc nsz float %a, 4.0
  %r = fadd nnan reassoc nsz float %t1, %t2
  ret float %r
}

define float @test6(float %a) #0 {
; CHECK-LABEL: test6:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fmul nnan reassoc nsz float 2.0, %a
  %t2 = fadd nnan reassoc nsz float %a, %a
  %r = fsub nnan reassoc nsz float %t1, %t2
  ret float %r
}

define float @test7(float %a) #0 {
; CHECK-LABEL: test7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fmul nnan reassoc nsz float %a, 2.0
  %t2 = fadd nnan reassoc nsz float %a, %a
  %r = fsub nnan reassoc nsz float %t1, %t2
  ret float %r
}

define float @test8(float %a) #0 {
; CHECK-LABEL: test8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %t1 = fmul nsz float %a, 0.0
  %t2 = fadd nnan reassoc nsz float %a, %t1
  ret float %t2
}

define float @test9(float %a) #0 {
; CHECK-LABEL: test9:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %t1 = fmul nsz float 0.0, %a
  %t2 = fadd nnan reassoc nsz float %t1, %a
  ret float %t2
}

define float @test10(float %a) #0 {
; CHECK-LABEL: test10:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vxorps %xmm0, %xmm0, %xmm0
; CHECK-NEXT:    retq
  %t1 = fsub nsz float -0.0, %a
  %t2 = fadd nnan reassoc nsz float %a, %t1
  ret float %t2
}

