; RUN: opt < %s -instcombine -S | FileCheck %s

define float @test1(i1 %A) {
EntryBlock:
  %cf = select i1 %A, float 1.000000e+00, float 0.000000e+00
  %op = fsub float 1.000000e+00, %cf
  ret float %op
; CHECK-LABEL: @test1(
; CHECK: select i1 %A, float 0.000000e+00, float 1.000000e+00
}

define float @test2(i1 %A, float %B) {
EntryBlock:
  %cf = select i1 %A, float 1.000000e+00, float %B
  %op = fadd float 2.000000e+00, %cf
  ret float %op
; CHECK-LABEL: @test2(
; CHECK: [[OP:%.*]] = fadd float %B, 2.000000e+00
; CHECK: select i1 %A, float 3.000000e+00, float [[OP]]
}

define float @test3(i1 %A, float %B) {
EntryBlock:
  %cf = select i1 %A, float 1.000000e+00, float %B
  %op = fsub float 2.000000e+00, %cf
  ret float %op
; CHECK-LABEL: @test3(
; CHECK: [[OP:%.*]] = fsub float 2.000000e+00, %B
; CHECK: select i1 %A, float 1.000000e+00, float [[OP]]
}

define float @test4(i1 %A, float %B) {
EntryBlock:
  %cf = select i1 %A, float 1.000000e+00, float %B
  %op = fmul float 2.000000e+00, %cf
  ret float %op
; CHECK-LABEL: @test4(
; CHECK: [[OP:%.*]] = fmul float %B, 2.000000e+00
; CHECK: select i1 %A, float 2.000000e+00, float [[OP]]
}

define float @test5(i1 %A, float %B) {
EntryBlock:
  %cf = select i1 %A, float 1.000000e+00, float %B
  %op = fdiv float 2.000000e+00, %cf
  ret float %op
; CHECK-LABEL: @test5(
; CHECK: [[OP:%.*]] = fdiv float 2.000000e+00, %B
; CHECK: select i1 %A, float 2.000000e+00, float [[OP]]
}

define float @test6(i1 %A, float %B) {
EntryBlock:
  %cf = select i1 %A, float 1.000000e+00, float %B
  %op = fdiv float %cf, 2.000000e+00
  ret float %op
; CHECK-LABEL: @test6(
; CHECK: [[OP:%.*]] = fmul float %B, 5.000000e-01
; CHECK: select i1 %A, float 5.000000e-01, float [[OP]]
}

define float @test7(i1 %A, float %B) {
EntryBlock:
  %cf = select i1 %A, float 1.000000e+00, float %B
  %op = fdiv float %cf, 3.000000e+00
  ret float %op
; CHECK-LABEL: @test7(
; CHECK: [[OP:%.*]] = fdiv float %B, 3.000000e+00
; CHECK: select i1 %A, float 0x3FD5555560000000, float [[OP]]
}

