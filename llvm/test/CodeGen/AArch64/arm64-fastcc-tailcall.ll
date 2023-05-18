; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

define void @caller(ptr nocapture %p, i32 %a, i32 %b) nounwind optsize ssp {
; CHECK-NOT: stp
; CHECK: b       {{_callee|callee}}
; CHECK-NOT: ldp
; CHECK: ret
  %1 = icmp eq i32 %b, 0
  br i1 %1, label %3, label %2

  tail call fastcc void @callee(ptr %p, i32 %a) optsize
  br label %3

  ret void
}

define internal fastcc void @callee(ptr nocapture %p, i32 %a) nounwind optsize noinline ssp {
  store volatile i32 %a, ptr %p, align 4, !tbaa !0
  ret void
}

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
