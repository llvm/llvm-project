; RUN: llc -march=hexagon < %s | FileCheck %s
; Generate reg = cmp.

@g0 = common global i8 0, align 1
@g1 = common global i32 0, align 4
@g2 = common global i8 0, align 1
@g3 = global i8 65, align 1

; CHECK-LABEL: f0:
; CHECK: r{{[0-9]+}} = cmp.eq(r{{[0-9]+}},#65)
define i32 @f0() #0 {
b0:
  %v0 = load i8, ptr @g0, align 1, !tbaa !0
  %v1 = icmp eq i8 %v0, 65
  %v2 = zext i1 %v1 to i32
  %v3 = load i32, ptr @g1, align 4, !tbaa !3
  %v4 = or i32 %v2, %v3
  store i32 %v4, ptr @g1, align 4, !tbaa !3
  store i8 66, ptr @g2, align 1, !tbaa !0
  ret i32 undef
}

; CHECK-LABEL: f1:
; CHECK: r{{[0-9]+}} = cmp.eq(r{{[0-9]+}},r{{[0-9]+}})
define i32 @f1() #0 {
b0:
  %v0 = load i8, ptr @g0, align 1, !tbaa !0
  %v1 = load i8, ptr @g3, align 1, !tbaa !0
  %v2 = icmp eq i8 %v0, %v1
  %v3 = zext i1 %v2 to i32
  %v4 = load i32, ptr @g1, align 4, !tbaa !3
  %v5 = or i32 %v3, %v4
  store i32 %v5, ptr @g1, align 4, !tbaa !3
  store i8 66, ptr @g2, align 1, !tbaa !0
  ret i32 undef
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !1}
