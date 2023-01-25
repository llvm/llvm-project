; RUN: opt -passes=instcombine -S < %s | FileCheck %s

declare void @bar()
declare void @baz()

; Check that nonnull metadata is from non-dominating loads is not propagated.
; CHECK-LABEL: cont:
; CHECK-NOT: !nonnull
define ptr @test_combine_metadata_dominance(i1 %c, ptr dereferenceable(8) %p1, ptr dereferenceable(8) %p2) {
  br i1 %c, label %t, label %f
t:
  call void @bar()
  %v1 = load ptr, ptr %p1, align 8, !nonnull !0
  br label %cont

f:
  call void @baz()
  %v2 = load ptr, ptr %p2, align 8
  br label %cont

cont:
  %res = phi ptr [ %v1, %t ], [ %v2, %f ]
  ret ptr %res
}

!0 = !{}
