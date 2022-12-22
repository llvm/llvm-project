; RUN: llc < %s -mtriple=armv7a-eabi   | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-AT2
; RUN: llc < %s -mtriple=thumbv7m-eabi | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-AT2
; RUN: llc < %s -mtriple=thumbv6m-eabi | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-T1

; This test checks that various kinds of getelementptr are all optimised to a
; simple multiply plus add, with the add being done by a register offset if the
; result is used in a load.

; CHECK-LABEL: calc_1d:
; CHECK: mov{{s?}} [[REG1:r[0-9]+]], #84
; CHECK-AT2: mla r0, r1, [[REG1]], r0
; CHECK-T1: muls [[REG2:r[0-9]+]], r1, [[REG1]]
; CHECK-T1: adds r0, r0, [[REG2]]
define ptr @calc_1d(ptr %p, i32 %n) {
entry:
  %mul = mul nsw i32 %n, 21
  %add.ptr = getelementptr inbounds i32, ptr %p, i32 %mul
  ret ptr %add.ptr
}

; CHECK-LABEL: load_1d:
; CHECK: mov{{s?}} [[REG1:r[0-9]+]], #84
; CHECK: mul{{s?}} [[REG2:r[0-9]+]],{{( r1,)?}} [[REG1]]{{(, r1)?}}
; CHECK: ldr r0, [r0, [[REG2]]]
define i32 @load_1d(ptr %p, i32 %n) #1 {
entry:
  %mul = mul nsw i32 %n, 21
  %arrayidx = getelementptr inbounds i32, ptr %p, i32 %mul
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

; CHECK-LABEL: calc_2d_a:
; CHECK: mov{{s?}} [[REG1:r[0-9]+]], #84
; CHECK-AT2: mla r0, r1, [[REG1]], r0
; CHECK-T1: muls [[REG2:r[0-9]+]], r1, [[REG1]]
; CHECK-T1: adds r0, r0, [[REG2]]
define ptr @calc_2d_a(ptr %p, i32 %n) {
entry:
  %mul = mul nsw i32 %n, 21
  %arrayidx1 = getelementptr inbounds [100 x i32], ptr %p, i32 0, i32 %mul
  ret ptr %arrayidx1
}

; CHECK-LABEL: load_2d_a:
; CHECK: mov{{s?}} [[REG1:r[0-9]+]], #84
; CHECK: mul{{s?}} [[REG2:r[0-9]+]],{{( r1,)?}} [[REG1]]{{(, r1)?}}
; CHECK: ldr r0, [r0, [[REG2]]]
define i32 @load_2d_a(ptr %p, i32 %n) #1 {
entry:
  %mul = mul nsw i32 %n, 21
  %arrayidx1 = getelementptr inbounds [100 x i32], ptr %p, i32 0, i32 %mul
  %0 = load i32, ptr %arrayidx1, align 4
  ret i32 %0
}

; CHECK-LABEL: calc_2d_b:
; CHECK: mov{{s?}} [[REG1:r[0-9]+]], #84
; CHECK-AT2: mla r0, r1, [[REG1]], r0
; CHECK-T1: muls [[REG2:r[0-9]+]], r1, [[REG1]]
; CHECK-T1: adds r0, r0, [[REG2]]
define ptr @calc_2d_b(ptr %p, i32 %n) {
entry:
  %arrayidx1 = getelementptr inbounds [21 x i32], ptr %p, i32 %n, i32 0
  ret ptr %arrayidx1
}

; CHECK-LABEL: load_2d_b:
; CHECK: mov{{s?}} [[REG1:r[0-9]+]], #84
; CHECK: mul{{s?}} [[REG2:r[0-9]+]],{{( r1,)?}} [[REG1]]{{(, r1)?}}
; CHECK: ldr r0, [r0, [[REG2]]]
define i32 @load_2d_b(ptr %p, i32 %n) {
entry:
  %arrayidx1 = getelementptr inbounds [21 x i32], ptr %p, i32 %n, i32 0
  %0 = load i32, ptr %arrayidx1, align 4
  ret i32 %0
}
