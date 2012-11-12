; RUN: opt -tbaa -sink -S < %s | FileCheck %s

; CHECK: a:
; CHECK:   %f = load float* %p, !tbaa !2
; CHECK:   store float %f, float* %q

define void @foo(float* %p, i1 %c, float* %q, float* %r) {
  %f = load float* %p, !tbaa !0
  store float 0.0, float* %r, !tbaa !1
  br i1 %c, label %a, label %b
a:
  store float %f, float* %q
  br label %b
b:
  ret void
}

!0 = metadata !{metadata !"A", metadata !2}
!1 = metadata !{metadata !"B", metadata !2}
!2 = metadata !{metadata !"test"}
