; Phase-1 / Mode-A: a masked store whose address is also accessed on a
; guaranteed path is rewritten as load-blend-store when
; -enable-masked-memory-optimization is on.
;
; RUN: opt < %s -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu \
; RUN:     -mcpu=znver3 -enable-masked-memory-optimization -S \
; RUN: | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @phase1_safe
; The masked store is replaced by load + select + plain store. We don't
; care about the exact instruction order, just that the named values
; introduced by the rewrite appear and no @llvm.masked.store is emitted.
; CHECK: blend.load = load
; CHECK: blend.sel = select
; CHECK-NOT: @llvm.masked.store
define void @phase1_safe(ptr noalias %A, ptr noalias %cond) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep_cond = getelementptr inbounds i32, ptr %cond, i64 %iv
  %cv = load i32, ptr %gep_cond, align 4
  %gep_A = getelementptr inbounds i32, ptr %A, i64 %iv
  ; Guaranteed read of A[iv] makes the same-SCEV predicated store safe.
  %old = load i32, ptr %gep_A, align 4
  %cmp = icmp ne i32 %cv, 0
  br i1 %cmp, label %if.then, label %loop.latch

if.then:
  %new = add i32 %old, 1
  store i32 %new, ptr %gep_A, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}
