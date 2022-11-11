; Verify that small structures and float arguments are passed in the
; least significant part of a stack slot doubleword.

; RUN: llc -verify-machineinstrs < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

%struct.large_arg = type { [8 x i64] }
%struct.small_arg = type { i16, i8 }

@gl = common global %struct.large_arg zeroinitializer, align 8
@gs = common global %struct.small_arg zeroinitializer, align 2
@gf = common global float 0.000000e+00, align 4

define void @callee1(ptr noalias nocapture sret(%struct.small_arg) %agg.result, ptr byval(%struct.large_arg) nocapture readnone %pad, ptr byval(%struct.small_arg) nocapture readonly %x) {
entry:
  %0 = load i32, ptr %x, align 2
  store i32 %0, ptr %agg.result, align 2
  ret void
}
; CHECK: @callee1
; CHECK: lwz {{[0-9]+}}, 104(1)
; CHECK: blr

define void @caller1() {
entry:
  %tmp = alloca %struct.small_arg, align 2
  call void @test1(ptr sret(%struct.small_arg) %tmp, ptr byval(%struct.large_arg) @gl, ptr byval(%struct.small_arg) @gs)
  ret void
}
; CHECK: @caller1
; CHECK: stw {{[0-9]+}}, 104(1)
; CHECK: bl test1

declare void @test1(ptr sret(%struct.small_arg), ptr byval(%struct.large_arg), ptr byval(%struct.small_arg))

define float @callee2(float %pad1, float %pad2, float %pad3, float %pad4, float %pad5, float %pad6, float %pad7, float %pad8, float %pad9, float %pad10, float %pad11, float %pad12, float %pad13, float %x) {
entry:
  ret float %x
}
; CHECK: @callee2
; CHECK: lfs {{[0-9]+}}, 136(1)
; CHECK: blr

define void @caller2() {
entry:
  %0 = load float, ptr @gf, align 4
  %call = tail call float @test2(float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float %0)
  ret void
}
; CHECK: @caller2
; CHECK: stw {{[0-9]+}}, 136({{[0-9]+}})
; CHECK: bl test2

declare float @test2(float, float, float, float, float, float, float, float, float, float, float, float, float, float)

