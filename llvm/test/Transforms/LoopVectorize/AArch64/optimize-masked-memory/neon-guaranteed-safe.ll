; OptimizeMaskedMemory on AArch64 NEON: a predicated store whose address is
; also accessed on a guaranteed path is rewritten as load-blend-store instead
; of being scalarized into the per-lane predicated ladder.
;
; NEON has no @llvm.masked.store, so without this rewrite the vectorizer emits
; a per-lane extractelement -> br i1 -> scalar store ladder. With the rewrite
; the store becomes an unconditional vector load + select (blend) + plain
; vector store.
;
; RUN: opt < %s -passes=loop-vectorize -mtriple=aarch64-unknown-linux-gnu \
; RUN:     -mattr=+neon,-sve -enable-masked-memory-optimization -S \
; RUN: | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: @neon_safe
; The predicated store is replaced by load + select + plain store. No masked
; store (NEON has none) and no per-lane scalarized store ladder.
; CHECK: blend.load = load
; CHECK: blend.sel = select
; CHECK-NOT: @llvm.masked.store
; CHECK-NOT: extractelement
define void @neon_safe(ptr noalias %A, ptr noalias %cond) {
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
  %mul = mul i32 %old, 3
  %new = add i32 %mul, 7
  store i32 %new, ptr %gep_A, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1024
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}
