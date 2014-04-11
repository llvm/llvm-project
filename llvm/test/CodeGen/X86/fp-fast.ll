; RUN: llc -march=x86-64 -mcpu=corei7-avx -enable-unsafe-fp-math < %s | FileCheck %s

; CHECK-LABEL: test1
define float @test1(float %a) {
; CHECK-NOT: addss
; CHECK: mulss
; CHECK-NOT: addss
; CHECK: ret
  %t1 = fadd float %a, %a
  %r = fadd float %t1, %t1
  ret float %r
}

; CHECK-LABEL: test2
define float @test2(float %a) {
; CHECK-NOT: addss
; CHECK: mulss
; CHECK-NOT: addss
; CHECK: ret
  %t1 = fmul float 4.0, %a
  %t2 = fadd float %a, %a
  %r = fadd float %t1, %t2
  ret float %r
}

; CHECK-LABEL: test3
define float @test3(float %a) {
; CHECK-NOT: addss
; CHECK: mulss
; CHECK-NOT: addss
; CHECK: ret
  %t1 = fmul float %a, 4.0
  %t2 = fadd float %a, %a
  %r = fadd float %t1, %t2
  ret float %r
}

; CHECK-LABEL: test4
define float @test4(float %a) {
; CHECK-NOT: addss
; CHECK: mulss
; CHECK-NOT: addss
; CHECK: ret
  %t1 = fadd float %a, %a
  %t2 = fmul float 4.0, %a
  %r = fadd float %t1, %t2
  ret float %r
}

; CHECK-LABEL: test5
define float @test5(float %a) {
; CHECK-NOT: addss
; CHECK: mulss
; CHECK-NOT: addss
; CHECK: ret
  %t1 = fadd float %a, %a
  %t2 = fmul float %a, 4.0
  %r = fadd float %t1, %t2
  ret float %r
}

; CHECK-LABEL: test6
define float @test6(float %a) {
; CHECK-NOT: addss
; CHECK: xorps
; CHECK-NOT: addss
; CHECK: ret
  %t1 = fmul float 2.0, %a
  %t2 = fadd float %a, %a
  %r = fsub float %t1, %t2
  ret float %r
}

; CHECK-LABEL: test7
define float @test7(float %a) {
; CHECK-NOT: addss
; CHECK: xorps
; CHECK-NOT: addss
; CHECK: ret
  %t1 = fmul float %a, 2.0
  %t2 = fadd float %a, %a
  %r = fsub float %t1, %t2
  ret float %r
}

; CHECK-LABEL: test8
define float @test8(float %a) {
; CHECK-NOT: fma
; CHECK-NOT: mul
; CHECK-NOT: add
; CHECK: ret
  %t1 = fmul float %a, 0.0
  %t2 = fadd float %a, %t1
  ret float %t2
}

; CHECK-LABEL: test9
define float @test9(float %a) {
; CHECK-NOT: fma
; CHECK-NOT: mul
; CHECK-NOT: add
; CHECK: ret
  %t1 = fmul float 0.0, %a
  %t2 = fadd float %t1, %a
  ret float %t2
}

; CHECK-LABEL: test10
define float @test10(float %a) {
; CHECK-NOT: add
; CHECK: vxorps
; CHECK: ret
  %t1 = fsub float -0.0, %a
  %t2 = fadd float %a, %t1
  ret float %t2
}

; CHECK-LABEL: test11
define float @test11(float %a) {
; CHECK-NOT: add
; CHECK: vxorps
; CHECK: ret
  %t1 = fsub float -0.0, %a
  %t2 = fadd float %a, %t1
  ret float %t2
}
