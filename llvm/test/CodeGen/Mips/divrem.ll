; RUN: llc -march=mips < %s | FileCheck %s

; CHECK: div $zero,
define i32 @sdiv1(i32 %a0, i32 %a1) nounwind readnone {
entry:
  %div = sdiv i32 %a0, %a1
  ret i32 %div
}

; CHECK: div $zero,
define i32 @srem1(i32 %a0, i32 %a1) nounwind readnone {
entry:
  %rem = srem i32 %a0, %a1
  ret i32 %rem
}

; CHECK: divu $zero,
define i32 @udiv1(i32 %a0, i32 %a1) nounwind readnone {
entry:
  %div = udiv i32 %a0, %a1
  ret i32 %div
}

; CHECK: divu $zero,
define i32 @urem1(i32 %a0, i32 %a1) nounwind readnone {
entry:
  %rem = urem i32 %a0, %a1
  ret i32 %rem
}

; CHECK: div $zero,
define i32 @sdivrem1(i32 %a0, i32 %a1, i32* nocapture %r) nounwind {
entry:
  %rem = srem i32 %a0, %a1
  store i32 %rem, i32* %r, align 4, !tbaa !0
  %div = sdiv i32 %a0, %a1
  ret i32 %div
}

; CHECK: divu $zero,
define i32 @udivrem1(i32 %a0, i32 %a1, i32* nocapture %r) nounwind {
entry:
  %rem = urem i32 %a0, %a1
  store i32 %rem, i32* %r, align 4, !tbaa !0
  %div = udiv i32 %a0, %a1
  ret i32 %div
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
