; RUN: opt -disable-output -passes=sandbox-vectorizer -sbvec-passes="regions-from-metadata<print-region>" %s | FileCheck %s
; REQUIRES: asserts

define void @foo(i8 %v) {
; CHECK: -- Region --
; CHECK-NEXT:  %add0 = add i8 %v, 0, !sandboxvec !0 {{.*}}
; CHECK: -- Region --
; CHECK-NEXT:  %add1 = add i8 %v, 1, !sandboxvec !1 {{.*}}
; CHECK-NEXT:  %add2 = add i8 %v, 2, !sandboxvec !1 {{.*}}
; CHECK-NEXT:  %add3 = add i8 %v, 3, !sandboxvec !1, !sandboxaux !2 {{.*}}
; CHECK-NEXT:  %add4 = add i8 %v, 4, !sandboxvec !1, !sandboxaux !3 {{.*}}
; CHECK: Aux:
; CHECK-NEXT:  %add3 = add i8 %v, 3, !sandboxvec !1, !sandboxaux !2 {{.*}}
; CHECK-NEXT:  %add4 = add i8 %v, 4, !sandboxvec !1, !sandboxaux !3 {{.*}}
  %add0 = add i8 %v, 0, !sandboxvec !0
  %add1 = add i8 %v, 1, !sandboxvec !1
  %add2 = add i8 %v, 2, !sandboxvec !1
  %add3 = add i8 %v, 3, !sandboxvec !1, !sandboxaux !2
  %add4 = add i8 %v, 4, !sandboxvec !1, !sandboxaux !3
  ret void
}

!0 = distinct !{!"sandboxregion"}
!1 = distinct !{!"sandboxregion"}
!2 = !{i32 0}
!3 = !{i32 1}
