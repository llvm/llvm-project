; RUN: opt -aa-pipeline=tbaa -passes=sink -S < %s | FileCheck %s

; CHECK: a:
; CHECK:   %f = load float, ptr %p, align 4, !tbaa [[TAGA:!.*]]
; CHECK:   store float %f, ptr %q

define void @foo(ptr %p, i1 %c, ptr %q, ptr %r) {
  %f = load float, ptr %p, !tbaa !0
  store float 0.0, ptr %r, !tbaa !1
  br i1 %c, label %a, label %b
a:
  store float %f, ptr %q
  br label %b
b:
  ret void
}

; CHECK: [[TAGA]] = !{[[TYPEA:!.*]], [[TYPEA]], i64 0}
; CHECK: [[TYPEA]] = !{!"A", !{{.*}}}
!0 = !{!3, !3, i64 0}
!1 = !{!4, !4, i64 0}
!2 = !{!"test"}
!3 = !{!"A", !2}
!4 = !{!"B", !2}
