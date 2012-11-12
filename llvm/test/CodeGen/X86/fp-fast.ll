; RUN: llc -march=x86-64 -mattr=+avx,-fma4 -mtriple=x86_64-apple-darwin -enable-unsafe-fp-math < %s | FileCheck %s

; CHECK: test1
define float @test1(float %a) {
; CHECK-NOT: addss
; CHECK: mulss
; CHECK-NOT: addss
; CHECK: ret
  %t1 = fadd float %a, %a
  %r = fadd float %t1, %t1
  ret float %r
}

; CHECK: test2
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

; CHECK: test3
define float @test3(float %a) {
; CHECK-NOT: addss
; CHECK: xorps
; CHECK-NOT: addss
; CHECK: ret
  %t1 = fmul float 2.0, %a
  %t2 = fadd float %a, %a
  %r = fsub float %t1, %t2
  ret float %r
}

; CHECK: test4
define float @test4(float %a) {
; CHECK-NOT: fma
; CHECK-NOT mul
; CHECK-NOT: add
; CHECK: ret
  %t1 = fmul float %a, 0.0
  %t2 = fadd float %a, %t1
  ret float %t2
}

; CHECK: test5
define float @test5(float %a) {
; CHECK-NOT: add
; CHECK: vxorps
; CHECK: ret
  %t1 = fsub float -0.0, %a
  %t2 = fadd float %a, %t1
  ret float %t2
}
