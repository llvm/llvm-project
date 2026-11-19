; NOTE: Do not autogenerate: these checks isolate load reuse and walk limits.
; RUN: opt -passes=loop-unroll -memssa-check-limit=1000 -verify-each -S %s | FileCheck %s --check-prefixes=CHECK,COMPLETE
; RUN: opt -passes=loop-unroll -memssa-check-limit=6 -verify-each -S %s | FileCheck %s --check-prefix=LIMIT

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; Stores between matching loads force loadCSE to query loop-scoped MemorySSA.
; The inner preheader belongs to the outer loop but not to the inner graph.

define void @cse_scoped_nested(ptr %src, ptr noalias %dst, i64 %outer.n) {
; CHECK-LABEL: define void @cse_scoped_nested(
; CHECK-NOT: load i32, ptr %src
; CHECK: [[FIRST:%.*]] = load i32, ptr %src
; CHECK-NOT: load i32, ptr %src
; CHECK: store i32 [[FIRST]], ptr %dst
; CHECK-NOT: load i32, ptr %src
; CHECK: store i32 [[FIRST]], ptr %dst
; CHECK-NOT: load i32, ptr %src
; CHECK: store i32 [[FIRST]], ptr %dst
; CHECK-NOT: load i32, ptr %src
; CHECK: store i32 [[FIRST]], ptr %dst
; CHECK-NOT: load i32, ptr %src
; CHECK: br i1 {{.*}}, label %inner.exit, label %inner
; CHECK-NOT: load i32, ptr %src
; CHECK: ret void
entry:
  br label %outer
outer:
  %oi = phi i64 [0, %entry], [%outer.next, %inner.exit]
  br label %inner.preheader
inner.preheader:
  br label %inner
inner:
  %ii = phi i32 [0, %inner.preheader], [%inner.next, %inner]
  %first = load i32, ptr %src
  store i32 %first, ptr %dst
  %second = load i32, ptr %src
  store i32 %second, ptr %dst
  %inner.next = add nuw nsw i32 %ii, 1
  %inner.done = icmp eq i32 %inner.next, 4
  br i1 %inner.done, label %inner.exit, label %inner, !llvm.loop !0
inner.exit:
  %outer.next = add nuw nsw i64 %oi, 1
  %outer.done = icmp eq i64 %outer.next, %outer.n
  br i1 %outer.done, label %exit, label %outer, !llvm.loop !2
exit:
  ret void
}

; loadCSE matches the load addresses by SCEV and uses MemorySSA to check reuse.
; A walk limit of six MemoryDefs keeps %after.1; a larger limit removes it.
; MemoryPhi operand order follows DT child order and affects the bounded walk.

define i32 @cse_phi_walk_limit(ptr noalias %a, ptr noalias %b, ptr noalias %q, i64 noundef %seed) {
; LIMIT-LABEL: define i32 @cse_phi_walk_limit(
; LIMIT-NOT: load i32
; LIMIT: %before = load i32, ptr %early.ptr,
; LIMIT-NOT: load i32
; LIMIT: %before.1 = load i32, ptr %early.ptr.1,
; LIMIT-NOT: load i32
; LIMIT: %after.1 = load i32, ptr %ptr.1,
; LIMIT-NEXT: %pair.1 = add i32 %before.1, %after.1
; LIMIT-NOT: load i32
; LIMIT: ret i32
;
; COMPLETE-LABEL: define i32 @cse_phi_walk_limit(
; COMPLETE-NOT: load i32
; COMPLETE: %before = load i32, ptr %early.ptr,
; COMPLETE-NOT: load i32
; COMPLETE: %before.1 = load i32, ptr %early.ptr.1,
; COMPLETE-NOT: load i32
; COMPLETE: %pair.1 = add i32 %before.1, %before.1
; COMPLETE-NOT: load i32
; COMPLETE: ret i32
entry:
  %b4 = getelementptr i32, ptr %b, i64 1
  br label %h
h:
  %i = phi i64 [0, %entry], [%next, %j]
  %acc = phi i32 [0, %entry], [%sum, %j]
  %cond = icmp ugt ptr %a, %b
  %early.ptr = select i1 %cond, ptr %a, ptr %b
  store i32 72, ptr %a
  store i32 31, ptr %b4
  %before = load i32, ptr %early.ptr
  store i32 27, ptr %q
  br i1 %cond, label %left, label %right
left:
  store i32 101, ptr %q
  br label %j
right:
  %zero = sub i64 %seed, %seed
  %wide.ptr = getelementptr i32, ptr %b, i64 %zero
  store i32 102, ptr %q
  br label %j
j:
  %ptr = phi ptr [%a, %left], [%wide.ptr, %right]
  %after = load i32, ptr %ptr
  %pair = add i32 %before, %after
  %sum = add i32 %acc, %pair
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 4
  br i1 %done, label %exit, label %h, !llvm.loop !4
exit:
  ret i32 %sum
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.count", i32 2}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.unroll.disable"}
!4 = distinct !{!4, !1}
