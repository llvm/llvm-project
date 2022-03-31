; RUN: opt -S -passes=instcombine < %s 2>&1 | FileCheck %s

define i64 @f(i64* %p1, i64* %p2) {
top:
  ; check that the tbaa is preserved
  ; CHECK-LABEL: @f(
  ; CHECK: %v1 = load i64, i64* %p1, align 8, !tbaa !0
  ; CHECK: store i64 %v1, i64* %p2, align 8
  ; CHECK: ret i64 %v1
  %v1 = load i64, i64* %p1, align 8, !tbaa !0
  store i64 %v1, i64* %p2, align 8
  %v2 = load i64, i64* %p2, align 8
  ret i64 %v2
}

!0 = !{!1, !1, i64 0}
!1 = !{!"scalar type", !2}
!2 = !{!"load_tbaa"}
