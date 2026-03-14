; RUN: opt -passes=instcombine -S < %s | FileCheck %s

declare void @bar()
declare void @baz()

; Check that align metadata is combined
; CHECK-LABEL: cont:
; CHECK: load ptr, ptr
; CHECK-SAME: !align ![[ALIGN:[0-9]+]]
define ptr @test_phi_combine_load_metadata(i1 %c, ptr dereferenceable(8) %p1, ptr dereferenceable(8) %p2) {
  br i1 %c, label %t, label %f
t:
  call void @bar()
  %v1 = load ptr, ptr %p1, align 8, !align !0
  br label %cont

f:
  call void @baz()
  %v2 = load ptr, ptr %p2, align 8, !align !1
  br label %cont

cont:
  %res = phi ptr [ %v1, %t ], [ %v2, %f ]
  ret ptr %res
}

; CHECK: ![[ALIGN]] = !{i64 8}

!0 = !{i64 8}
!1 = !{i64 16}
