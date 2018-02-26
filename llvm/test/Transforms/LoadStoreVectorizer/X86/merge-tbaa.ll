; RUN: opt -mtriple=x86_64-unknown-linux-gnu -load-store-vectorizer -S < %s | \
; RUN:     FileCheck %s
;
; The GPU Load & Store Vectorizer may merge differently-typed accesses into a
; single instruction. This test checks that we merge TBAA tags for such
; accesses correctly.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; struct S {
;   float f;
;   int i;
; };
%struct.S = type { float, i32 }

; float foo(S *p) {
;   p->f -= 1;
;   p->i -= 1;
;   return p->f;
; }
define float @foo(%struct.S* %p) {
entry:
; CHECK-LABEL: foo
; CHECK: load <2 x i32>, {{.*}}, !tbaa [[TAG_char:!.*]]
; CHECK: store <2 x i32> {{.*}}, !tbaa [[TAG_char]]
  %f = getelementptr inbounds %struct.S, %struct.S* %p, i64 0, i32 0
  %0 = load float, float* %f, align 4, !tbaa !2
  %sub = fadd float %0, -1.000000e+00
  store float %sub, float* %f, align 4, !tbaa !2
  %i = getelementptr inbounds %struct.S, %struct.S* %p, i64 0, i32 1
  %1 = load i32, i32* %i, align 4, !tbaa !8
  %sub1 = add nsw i32 %1, -1
  store i32 %sub1, i32* %i, align 4, !tbaa !8
  ret float %sub
}

!2 = !{!3, !4, i64 0}
!3 = !{!"_ZTS1S", !4, i64 0, !7, i64 4}
!4 = !{!"float", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"int", !5, i64 0}
!8 = !{!3, !7, i64 4}

; CHECK-DAG: [[TYPE_char:!.*]] = !{!"omnipotent char", {{.*}}, i64 0}
; CHECK-FAG: [[TAG_char]] = !{[[TYPE_char]], [[TYPE_char]], i64 0}
