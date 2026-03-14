; RUN: opt -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=2 \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue -S -o - < %s | FileCheck %s
; RUN: opt -mattr=+sve -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=2 \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue -S -o - < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; This test is defending against a bug that appeared when we have a target
; configuration where masked loads/stores are legal -- e.g. AArch64 with SVE.
; Predication would not be applied during interleaving, enabling the
; possibility of superfluous loads/stores which could result in miscompiles.
; This test checks that, when we disable vectorisation and force interleaving,
; stores are predicated properly.
;
; This is _not_ an SVE-specific test. The same bug could manifest on any
; architecture with masked loads/stores, but we use SVE for testing purposes
; here.

define void @foo(ptr %data1, ptr %data2) {
; CHECK-LABEL: @foo(
; CHECK:       vector.body:
; CHECK:         br i1 {{%.*}}, label %pred.store.if, label %pred.store.continue
; CHECK:       pred.store.if:
; CHECK-NEXT:    store i32 {{%.*}}, ptr {{%.*}}
; CHECK-NEXT:    br label %pred.store.continue
; CHECK:       pred.store.continue:
; CHECK-NEXT:    br i1 {{%.*}}, label %pred.store.if1, label %pred.store.continue2
; CHECK:       pred.store.if1:
; CHECK-NEXT:    store i32 {{%.*}}, ptr {{%.*}}
; CHECK-NEXT:    br label %pred.store.continue2
; CHECK:       pred.store.continue2:

entry:
  br label %while.body

while.body:
  %i = phi i64 [ 1023, %entry ], [ %i.next, %if.end ]
  %arrayidx = getelementptr inbounds i32, ptr %data1, i64 %i
  %ld = load i32, ptr %arrayidx, align 4
  %cmp = icmp sgt i32 %ld, %ld
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %ld, ptr %arrayidx, align 4
  br label %if.end

if.end:
  %i.next = add nsw i64 %i, -1
  %tobool.not = icmp eq i64 %i, 0
  br i1 %tobool.not, label %while.end, label %while.body

while.end:
  ret void
}
