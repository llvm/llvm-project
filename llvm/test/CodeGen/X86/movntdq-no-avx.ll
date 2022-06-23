; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

; Test that we produce a movntdq, not a vmovntdq
; CHECK-NOT: vmovntdq

define void @test(ptr nocapture %a, <2 x i64> %b) nounwind optsize {
entry:
  store <2 x i64> %b, ptr %a, align 32, !nontemporal !0
  ret void
}

!0 = !{i32 1}
