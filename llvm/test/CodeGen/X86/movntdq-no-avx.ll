; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

; Test that we produce a movntdq, not a vmovntdq
; CHECK-NOT: vmovntdq

define void @test(<2 x i64>* nocapture %a, <2 x i64> %b) nounwind optsize {
entry:
  store <2 x i64> %b, <2 x i64>* %a, align 16, !nontemporal !0
  ret void
}

!0 = metadata !{i32 1}
