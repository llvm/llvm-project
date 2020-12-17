; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -hoist-common-insts=true -S | FileCheck %s

declare void @bar(i32*)
declare void @baz(i32*)

; CHECK-LABEL: @test_load_combine_metadata(
; Check that dereferenceable metadata is combined
; CHECK: load i32*, i32** %p
; CHECK-SAME: !dereferenceable ![[DEREF:[0-9]+]]
; CHECK: t:
; CHECK: f:
define void @test_load_combine_metadata(i1 %c, i32** %p) {
  br i1 %c, label %t, label %f

t:
  %v1 = load i32*, i32** %p, !dereferenceable !0
  call void @bar(i32* %v1)
  br label %cont

f:
  %v2 = load i32*, i32** %p, !dereferenceable !1
  call void @baz(i32* %v2)
  br label %cont

cont:
  ret void
}

; CHECK: ![[DEREF]] = !{i64 8}

!0 = !{i64 8}
!1 = !{i64 16}
