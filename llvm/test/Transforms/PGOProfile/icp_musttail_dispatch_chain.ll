; Indirect call promotion of a threaded interpreter dispatch site must not lead
; to the promoted handler being inlined into its caller, which would chain
; handler after handler into one function.

; RUN: opt < %s -passes='pgo-icall-prom,inline' -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@table = external global [256 x ptr]

define i64 @handler_add(ptr %pc, ptr %regs) {
; CHECK-LABEL: define i64 @handler_add(
; CHECK: musttail call i64 @handler_xor(
; CHECK-NOT: musttail call i64 @handler_add(
entry:
  %op = load i8, ptr %pc, align 1
  %opz = zext i8 %op to i64
  %v = load i64, ptr %regs, align 8
  %nv = add i64 %v, 1
  store i64 %nv, ptr %regs, align 8
  %slot = getelementptr inbounds [256 x ptr], ptr @table, i64 0, i64 %opz
  %next = load ptr, ptr %slot, align 8
  %pc1 = getelementptr inbounds i8, ptr %pc, i64 4
  %ret = musttail call i64 %next(ptr %pc1, ptr %regs), !prof !0
  ret i64 %ret
}

define i64 @handler_xor(ptr %pc, ptr %regs) {
; CHECK-LABEL: define i64 @handler_xor(
; CHECK: musttail call i64 @handler_add(
; CHECK-NOT: musttail call i64 @handler_xor(
entry:
  %op = load i8, ptr %pc, align 1
  %opz = zext i8 %op to i64
  %v = load i64, ptr %regs, align 8
  %nv = xor i64 %v, 3
  store i64 %nv, ptr %regs, align 8
  %slot = getelementptr inbounds [256 x ptr], ptr @table, i64 0, i64 %opz
  %next = load ptr, ptr %slot, align 8
  %pc1 = getelementptr inbounds i8, ptr %pc, i64 4
  %ret = musttail call i64 %next(ptr %pc1, ptr %regs), !prof !1
  ret i64 %ret
}

!0 = !{!"VP", i32 0, i64 10000, i64 13351034563885857626, i64 9000}
!1 = !{!"VP", i32 0, i64 10000, i64 8050033870388950925, i64 9000}
