; RUN: llc -march=hexagon -mcpu=hexagonv67t < %s
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

%s.0 = type { double, double, double, double, double, double, i32, double, double, double, double, ptr, i8, [9 x i8], double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, [200 x ptr], [32 x ptr], [32 x i8], i32 }

define hidden fastcc void @f0() unnamed_addr #0 {
b0:
  %v0 = getelementptr inbounds %s.0, ptr null, i32 0, i32 33
  %v1 = getelementptr inbounds %s.0, ptr null, i32 0, i32 34
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v2 = phi ptr [ undef, %b0 ], [ %v27, %b1 ]
  %v3 = load i32, ptr %v2, align 1, !tbaa !1
  %v4 = getelementptr inbounds [0 x ptr], ptr null, i32 0, i32 %v3
  %v5 = load ptr, ptr %v4, align 1, !tbaa !5
  %v6 = load double, ptr undef, align 1, !tbaa !7
  %v7 = fdiv double 1.000000e+00, %v6
  %v8 = fmul double %v7, 0.000000e+00
  %v9 = fmul double %v7, 0.000000e+00
  %v10 = fmul double %v8, -4.800000e+01
  %v11 = fmul double %v9, 1.680000e+02
  %v12 = fmul double %v7, 0.000000e+00
  %v13 = load double, ptr null, align 1, !tbaa !7
  %v14 = fmul double %v7, %v13
  %v15 = fmul double %v12, 0.000000e+00
  %v16 = getelementptr inbounds %s.0, ptr %v5, i32 0, i32 30
  %v17 = fsub double 0.000000e+00, %v15
  store double %v17, ptr %v16, align 8, !tbaa !9
  %v18 = fmul double %v14, 0.000000e+00
  %v19 = getelementptr inbounds %s.0, ptr %v5, i32 0, i32 32
  %v20 = load double, ptr %v19, align 8, !tbaa !11
  %v21 = fsub double %v20, %v18
  store double %v21, ptr %v19, align 8, !tbaa !11
  %v22 = fmul double %v10, 0.000000e+00
  %v23 = fadd double 0.000000e+00, %v22
  %v24 = fmul double 0.000000e+00, %v11
  %v25 = fadd double %v23, %v24
  %v26 = fsub double 0.000000e+00, %v25
  store double %v26, ptr %v0, align 8, !tbaa !12
  store double 0.000000e+00, ptr %v1, align 8, !tbaa !13
  %v27 = getelementptr i32, ptr %v2, i32 1
  br label %b1
}

attributes #0 = { "use-soft-float"="false" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"EnableSplitLTOUnit", i32 0}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !3, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"double", !3, i64 0}
!9 = !{!10, !8, i64 232}
!10 = !{!"", !8, i64 0, !8, i64 8, !8, i64 16, !8, i64 24, !8, i64 32, !8, i64 40, !2, i64 48, !8, i64 56, !8, i64 64, !8, i64 72, !8, i64 80, !6, i64 88, !3, i64 92, !3, i64 93, !8, i64 104, !8, i64 112, !8, i64 120, !8, i64 128, !8, i64 136, !8, i64 144, !8, i64 152, !8, i64 160, !8, i64 168, !8, i64 176, !8, i64 184, !8, i64 192, !8, i64 200, !8, i64 208, !8, i64 216, !8, i64 224, !8, i64 232, !8, i64 240, !8, i64 248, !8, i64 256, !8, i64 264, !8, i64 272, !8, i64 280, !8, i64 288, !8, i64 296, !3, i64 304, !3, i64 1104, !3, i64 1232, !2, i64 1264}
!11 = !{!10, !8, i64 248}
!12 = !{!10, !8, i64 256}
!13 = !{!10, !8, i64 264}
