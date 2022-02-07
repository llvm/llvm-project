; RUN: opt -passes=instcombine -S < %s | FileCheck %s

declare void @bar()
declare void @baz()

; Check that dereferenceable metadata is combined
; CHECK-LABEL: cont:
; CHECK: load i32*, i32**
; CHECK-SAME: !dereferenceable ![[DEREF:[0-9]+]]
define i32* @test_phi_combine_load_metadata(i1 %c, i32** dereferenceable(8) %p1, i32** dereferenceable(8) %p2) {
  br i1 %c, label %t, label %f
t:
  call void @bar()
  %v1 = load i32*, i32** %p1, align 8, !dereferenceable !0
  br label %cont

f:
  call void @baz()
  %v2 = load i32*, i32** %p2, align 8, !dereferenceable !1
  br label %cont

cont:
  %res = phi i32* [ %v1, %t ], [ %v2, %f ]
  ret i32* %res
}

; CHECK: ![[DEREF]] = !{i64 8}

!0 = !{i64 8}
!1 = !{i64 16}
