; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Tests folding constants from two similar selects that feed a add
define float @test1a(i1 zeroext %arg) #0 {
  %tmp = select i1 %arg, float 5.000000e+00, float 6.000000e+00
  %tmp1 = select i1 %arg, float 1.000000e+00, float 9.000000e+00
  %tmp2 = fadd float %tmp, %tmp1
  ret float %tmp2
; CHECK-LABEL: @test1a(
; CHECK: %tmp2 = select i1 %arg, float 6.000000e+00, float 1.500000e+01
; CHECK-NOT: fadd
; CHECK: ret float %tmp2
}

; Tests folding multiple expression constants from similar selects that feed a adds
define float @test1b(i1 zeroext %arg) #0 {
  %tmp = select i1 %arg, float 5.000000e+00, float 6.000000e+00
  %tmp1 = select i1 %arg, float 1.000000e+00, float 9.000000e+00
  %tmp2 = select i1 %arg, float 2.500000e-01, float 4.000000e+00
  %tmp3 = fadd float %tmp, %tmp1
  %tmp4 = fadd float %tmp2, %tmp1
  %tmp5 = fadd float %tmp4, %tmp3
  ret float %tmp5
; CHECK-LABEL: @test1b(
; CHECK: %tmp5 = select i1 %arg, float 7.250000e+00, float 2.800000e+01
; CHECK-NOT: fadd
; CHECK: ret float %tmp5
}

; Tests folding constants from two similar selects that feed a sub
define float @test2(i1 zeroext %arg) #0 {
  %tmp = select i1 %arg, float 5.000000e+00, float 6.000000e+00
  %tmp1 = select i1 %arg, float 1.000000e+00, float 9.000000e+00
  %tmp2 = fsub float %tmp, %tmp1
  ret float %tmp2
; CHECK-LABEL: @test2(
; CHECK: %tmp2 = select i1 %arg, float 4.000000e+00, float -3.000000e+00
; CHECK-NOT: fsub
; CHECK: ret float %tmp2
}

; Tests folding constants from two similar selects that feed a mul
define float @test3(i1 zeroext %arg) #0 {
  %tmp = select i1 %arg, float 5.000000e+00, float 6.000000e+00
  %tmp1 = select i1 %arg, float 1.000000e+00, float 9.000000e+00
  %tmp2 = fmul float %tmp, %tmp1
  ret float %tmp2
; CHECK-LABEL: @test3(
; CHECK: %tmp2 = select i1 %arg, float 5.000000e+00, float 5.400000e+01
; CHECK-NOT: fmul
; CHECK: ret float %tmp2
}

declare void @use_float(float)

; Tests folding constants if the selects have multiple uses but
; we can fold away the binary op with a select.
define float @test4(i1 zeroext %arg) #0 {
  %tmp = select i1 %arg, float 5.000000e+00, float 6.000000e+00
  %tmp1 = select i1 %arg, float 1.000000e+00, float 9.000000e+00
  %tmp2 = fmul float %tmp, %tmp1
  call void @use_float(float %tmp)
  ret float %tmp2
; CHECK-LABEL: @test4(
; CHECK: select i1 %arg, float 5.000000e+00, float 6.000000e+00
; CHECK-NEXT: select i1 %arg, float 5.000000e+00, float 5.400000e+01
; CHECK-NEXT: call void @use_float(float %tmp)
; CHECK: ret float %tmp2
}

; Tests not folding constants if we cannot fold away any of the selects.
define float @test5(i1 zeroext %arg, float %div) {
  %tmp = select i1 %arg, float %div, float 5.000000e+00
  %mul = fmul contract float %tmp, %tmp
  call void @use_float(float %tmp)
  ret float %mul
; CHECK-LABEL: @test5(
; CHECK: [[TMP:%.*]] = select i1 %arg, float %div, float 5.000000e+00
; CHECK-NEXT: [[MUL:%.*]] = fmul contract float [[TMP]], [[TMP]]
; CHECK-NOT: fmul contract float %div, %div
; CHECK: ret float [[MUL]]
}
