; RUN: llc -verify-machineinstrs -O2 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; CHECK-LABEL: test1
define i8 @test1(i32 %a) {
entry:
; CHECK-NOT: rlwinm {{{[0-9]+}}}, {{[0-9]+}}, 0, 24, 27
; CHECK: andi. [[REG:[0-9]+]], {{[0-9]+}}, 240
; CHECK-NOT: cmplwi [[REG]], 0
; CHECK: beq 0
  %0 = and i32 %a, 240
  %1 = icmp eq i32 %0, 0
  br i1 %1, label %eq0, label %neq0
eq0:
  ret i8 102
neq0:
  ret i8 116
}

; CHECK-LABEL: test2
define i8 @test2(i32 %a) {
entry:
; CHECK: rlwinm [[REG:[0-9]+]], {{[0-9]+}}, 0, 28, 23
; CHECK: cmplwi [[REG]], 0
; CHECK: beq 0
  %0 = and i32 %a, -241
  %1 = icmp eq i32 %0, 0
  br i1 %1, label %eq0, label %neq0
eq0:
  ret i8 102
neq0:
  ret i8 116
}

declare {i32, i1} @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)

; CHECK-LABEL: test3
define i8 @test3(i32 %a, i32 %b) {
entry:
; CHECK-NOT: rlwnm {{{[0-9]+}}}, {{[0-9]+}}, {{{[0-9]+}}}, 28, 31
; CHECK: rlwnm. [[REG:[0-9]+]], {{[0-9]+}}, 4, 28, 31
; CHECK-NOT: cmplwi [[REG]], 0
; CHECK: beq 0
  %left = shl i32 %a, %b
  %res = call {i32, i1} @llvm.ssub.with.overflow.i32(i32 32, i32 %b)
  %right_amount = extractvalue {i32, i1} %res, 0
  %right = lshr i32 %a, %right_amount
  %0 = or i32 %left, %right
  %1 = and i32 %0, 15
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %eq0, label %neq0
eq0:
  ret i8 102
neq0:
  ret i8 116
}
