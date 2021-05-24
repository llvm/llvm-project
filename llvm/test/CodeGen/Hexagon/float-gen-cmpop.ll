; RUN: llc -march=hexagon -O3 < %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: f0:
; CHECK: p{{[0-9]+}} = sfcmp.ge(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: p{{[0-9]+}} = sfcmp.gt(r{{[0-9]+}},r{{[0-9]+}})
define i32 @f0(float* nocapture %a0) #0 {
b0:
  %v0 = load float, float* %a0, align 4, !tbaa !0
  %v1 = fcmp olt float %v0, 6.000000e+01
  br i1 %v1, label %b1, label %b2

b1:                                               ; preds = %b0
  %v2 = getelementptr inbounds float, float* %a0, i32 1
  %v3 = load float, float* %v2, align 4, !tbaa !0
  %v4 = fcmp ogt float %v3, 0x3FECCCCCC0000000
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v5 = phi i1 [ false, %b0 ], [ %v4, %b1 ]
  %v6 = zext i1 %v5 to i32
  ret i32 %v6
}

; CHECK-LABEL: f1:
; CHECK: p{{[0-9]+}} = sfcmp.eq(r{{[0-9]+}},r{{[0-9]+}})
define i32 @f1(float* nocapture %a0) #0 {
b0:
  %v0 = load float, float* %a0, align 4, !tbaa !0
  %v1 = fcmp oeq float %v0, 6.000000e+01
  %v2 = zext i1 %v1 to i32
  ret i32 %v2
}

; CHECK-LABEL: f2:
; CHECK: p{{[0-9]+}} = dfcmp.ge(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
; CHECK: p{{[0-9]+}} = dfcmp.gt(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
define i32 @f2(double* nocapture %a0) #0 {
b0:
  %v0 = load double, double* %a0, align 8, !tbaa !4
  %v1 = fcmp olt double %v0, 6.000000e+01
  br i1 %v1, label %b1, label %b2

b1:                                               ; preds = %b0
  %v2 = getelementptr inbounds double, double* %a0, i32 1
  %v3 = load double, double* %v2, align 8, !tbaa !4
  %v4 = fcmp ogt double %v3, 0x3FECCCCCC0000000
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v5 = phi i1 [ false, %b0 ], [ %v4, %b1 ]
  %v6 = zext i1 %v5 to i32
  ret i32 %v6
}

define i32 @f3(double* nocapture %a0) #0 {
b0:
  %v0 = load double, double* %a0, align 8, !tbaa !4
  %v1 = fcmp oeq double %v0, 6.000000e+01
  %v2 = zext i1 %v1 to i32
  ret i32 %v2
}

attributes #0 = { nounwind readonly "target-cpu"="hexagonv55" "no-nans-fp-math"="true" }

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"double", !2}
