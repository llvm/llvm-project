; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec -mattr=-vsx < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

@g = common global ppc_fp128 0xM00000000000000000000000000000000, align 16

define void @callee(ppc_fp128 %x) {
entry:
  %x.addr = alloca ppc_fp128, align 16
  store ppc_fp128 %x, ppc_fp128* %x.addr, align 16
  %0 = load ppc_fp128, ppc_fp128* %x.addr, align 16
  store ppc_fp128 %0, ppc_fp128* @g, align 16
  ret void
}
; CHECK: @callee
; CHECK: ld [[REG:[0-9]+]], .LC
; CHECK: stfd 2, 8([[REG]])
; CHECK: stfd 1, 0([[REG]])
; CHECK: blr

define void @caller() {
entry:
  %0 = load ppc_fp128, ppc_fp128* @g, align 16
  call void @test(ppc_fp128 %0)
  ret void
}
; CHECK: @caller
; CHECK: ld [[REG:[0-9]+]], .LC
; CHECK: lfd 2, 8([[REG]])
; CHECK: lfd 1, 0([[REG]])
; CHECK: bl test

declare void @test(ppc_fp128)

define void @caller_const() {
entry:
  call void @test(ppc_fp128 0xM3FF00000000000000000000000000000)
  ret void
}
; CHECK: .LCPI[[LC:[0-9]+]]_0:
; CHECK: .long   1065353216
; CHECK: .LCPI[[LC]]_1:
; CHECK: .long   0
; CHECK: @caller_const
; CHECK: addi [[REG0:[0-9]+]], {{[0-9]+}}, .LCPI[[LC]]_0
; CHECK: addi [[REG1:[0-9]+]], {{[0-9]+}}, .LCPI[[LC]]_1
; CHECK: lfs 1, 0([[REG0]])
; CHECK: lfs 2, 0([[REG1]])
; CHECK: bl test

define ppc_fp128 @result() {
entry:
  %0 = load ppc_fp128, ppc_fp128* @g, align 16
  ret ppc_fp128 %0
}
; CHECK: @result
; CHECK: ld [[REG:[0-9]+]], .LC
; CHECK: lfd 1, 0([[REG]])
; CHECK: lfd 2, 8([[REG]])
; CHECK: blr

define void @use_result() {
entry:
  %call = tail call ppc_fp128 @test_result() #3
  store ppc_fp128 %call, ppc_fp128* @g, align 16
  ret void
}
; CHECK: @use_result
; CHECK: bl test_result
; CHECK: ld [[REG:[0-9]+]], .LC
; CHECK: stfd 2, 8([[REG]])
; CHECK: stfd 1, 0([[REG]])
; CHECK: blr

declare ppc_fp128 @test_result()

define void @caller_result() {
entry:
  %call = tail call ppc_fp128 @test_result()
  tail call void @test(ppc_fp128 %call)
  ret void
}
; CHECK: @caller_result
; CHECK: bl test_result
; CHECK-NEXT: nop
; CHECK-NEXT: bl test
; CHECK-NEXT: nop

define i128 @convert_from(ppc_fp128 %x) {
entry:
  %0 = bitcast ppc_fp128 %x to i128
  ret i128 %0
}
; CHECK: @convert_from
; CHECK: stfd 1, [[OFF1:.*]](1)
; CHECK: stfd 2, [[OFF2:.*]](1)
; CHECK: ld 3, [[OFF1]](1)
; CHECK: ld 4, [[OFF2]](1)
; CHECK: blr

define ppc_fp128 @convert_to(i128 %x) {
entry:
  %0 = bitcast i128 %x to ppc_fp128
  ret ppc_fp128 %0
}
; CHECK: convert_to:
; CHECK: std 3, [[OFF1:.*]](1)
; CHECK: std 4, [[OFF2:.*]](1)
; CHECK: ori 2, 2, 0
; CHECK: lfd 1, [[OFF1]](1)
; CHECK: lfd 2, [[OFF2]](1)
; CHECK: blr

define ppc_fp128 @convert_to2(i128 %x) {
entry:
  %shl = shl i128 %x, 1
  %0 = bitcast i128 %shl to ppc_fp128
  ret ppc_fp128 %0
}

; CHECK: convert_to2:
; CHECK: std 3, [[OFF1:.*]](1)
; CHECK: std 5, [[OFF2:.*]](1)
; CHECK: ori 2, 2, 0
; CHECK: lfd 1, [[OFF1]](1)
; CHECK: lfd 2, [[OFF2]](1)
; CHECK: blr

define double @convert_vector(<4 x i32> %x) {
entry:
  %cast = bitcast <4 x i32> %x to ppc_fp128
  %conv = fptrunc ppc_fp128 %cast to double
  ret double %conv
}
; CHECK: @convert_vector
; CHECK: addi [[REG:[0-9]+]], 1, [[OFF:.*]]
; CHECK: stvx 2, 0, [[REG]]
; CHECK: lfd 1, [[OFF]](1)
; CHECK: blr

declare void @llvm.va_start(i8*)

define double @vararg(i32 %a, ...) {
entry:
  %va = alloca i8*, align 8
  %va1 = bitcast i8** %va to i8*
  call void @llvm.va_start(i8* %va1)
  %arg = va_arg i8** %va, ppc_fp128
  %conv = fptrunc ppc_fp128 %arg to double
  ret double %conv
}
; CHECK: @vararg
; CHECK: lfd 1, 0({{[0-9]+}})
; CHECK: blr

