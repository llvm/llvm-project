; RUN: opt -passes=instcombine -S < %s | FileCheck %s

declare void @bar()
declare void @baz()

; Check that dereferenceable_or_null metadata is combined
; CHECK-LABEL: cont:
; CHECK: load ptr, ptr
; CHECK-SAME: !dereferenceable_or_null ![[DEREF:[0-9]+]]
define ptr @test_phi_combine_load_metadata(i1 %c, ptr dereferenceable(8) %p1, ptr dereferenceable(8) %p2) {
  br i1 %c, label %t, label %f
t:
  call void @bar()
  %v1 = load ptr, ptr %p1, align 8, !dereferenceable_or_null !0
  br label %cont

f:
  call void @baz()
  %v2 = load ptr, ptr %p2, align 8, !dereferenceable_or_null !1
  br label %cont

cont:
  %res = phi ptr [ %v1, %t ], [ %v2, %f ]
  ret ptr %res
}

; CHECK: ![[DEREF]] = !{i64 8}

!0 = !{i64 8}
!1 = !{i64 16}
