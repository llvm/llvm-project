; RUN: opt -passes=strip-dead-cg-profile %s -S -o - | FileCheck %s --check-prefix=NOOP
; RUN: llvm-extract %s -func=a -S -o - | FileCheck %s --check-prefix=EXTRACT-A
; RUN: llvm-extract %s -func=a --func=b -S -o - | FileCheck %s --check-prefix=EXTRACT-AB
; RUN: llvm-extract %s -func=solo -S -o - | FileCheck %s --check-prefix=NOTHING-LEFT

define void @a() {
  call void @b()
  ret void
}

define void @b() {
  call void @c()
  ret void
}

define void @c() {
  call void @d()
  ret void
}

define void @d() {
  ret void
}

define void @solo() {
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 5, !"CG Profile", !1}
!1 = !{!2, !3, !4}
!2 = !{ptr @a, ptr @b, i64 42}
!3 = !{ptr @b, ptr @c, i64 20}
!4 = !{ptr @c, ptr @d, i64 101}

; NOOP:       !0 = !{i32 5, !"CG Profile", !1}
; NOOP-NEXT:  !1 = distinct !{!2, !3, !4}
; NOOP-NEXT:  !2 = !{ptr @a, ptr @b, i64 42}
; NOOP-NEXT:  !3 = !{ptr @b, ptr @c, i64 20}
; NOOP-NEXT:  !4 = !{ptr @c, ptr @d, i64 101}

; EXTRACT-A:      !0 = !{i32 5, !"CG Profile", !1}
; EXTRACT-A-NEXT: !1 = distinct !{!2}
; EXTRACT-A-NEXT: !2 = !{ptr @a, ptr @b, i64 42}

; EXTRACT-AB:      !0 = !{i32 5, !"CG Profile", !1}
; EXTRACT-AB-NEXT: !1 = distinct !{!2, !3}
; EXTRACT-AB-NEXT: !2 = !{ptr @a, ptr @b, i64 42}
; EXTRACT-AB-NEXT: !3 = !{ptr @b, ptr @c, i64 20}

; NOTHING-LEFT:      !0 = !{i32 5, !"CG Profile", !1}
; NOTHING-LEFT-NEXT: !1 = distinct !{}