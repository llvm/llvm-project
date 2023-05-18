; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -hoist-common-insts=true -S | FileCheck %s

declare void @bar(ptr)
declare void @baz(ptr)

; CHECK-LABEL: @test_load_combine_metadata(
; Check that align metadata is combined
; CHECK: load ptr, ptr %p
; CHECK-SAME: !align ![[ALIGN:[0-9]+]]
; CHECK: t:
; CHECK: f:
define void @test_load_combine_metadata(i1 %c, ptr %p) {
  br i1 %c, label %t, label %f

t:
  %v1 = load ptr, ptr %p, !align !0
  call void @bar(ptr %v1)
  br label %cont

f:
  %v2 = load ptr, ptr %p, !align !1
  call void @baz(ptr %v2)
  br label %cont

cont:
  ret void
}

; CHECK: ![[ALIGN]] = !{i64 8}

!0 = !{i64 8}
!1 = !{i64 16}
