; RUN: llc -march=hexagon -machine-sink-split=0 < %s
; REQUIRES: asserts
; Used to fail with: Assertion `ScavengingFrameIndex >= 0 && "Cannot scavenge register without an emergency spill slot!"' failed.

target triple = "hexagon-unknown-linux-gnu"

%s.0 = type { double, double, double, double, double, double, i32, double, double, double, double, ptr, i8, [9 x i8], double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, [200 x ptr], [32 x ptr], [32 x i8], i32 }

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  %v0 = call ptr @f2()
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  ret void

b2:                                               ; preds = %b0
  br i1 undef, label %b3, label %b4

b3:                                               ; preds = %b2
  unreachable

b4:                                               ; preds = %b2
  br i1 undef, label %b5, label %b6

b5:                                               ; preds = %b4
  unreachable

b6:                                               ; preds = %b4
  %v1 = call i32 @f1() #0
  br i1 undef, label %b7, label %b20

b7:                                               ; preds = %b6
  switch i32 undef, label %b8 [
    i32 6, label %b9
    i32 1, label %b14
    i32 2, label %b13
    i32 3, label %b12
    i32 4, label %b11
    i32 5, label %b10
  ]

b8:                                               ; preds = %b7
  br label %b9

b9:                                               ; preds = %b8, %b7
  unreachable

b10:                                              ; preds = %b7
  unreachable

b11:                                              ; preds = %b7
  unreachable

b12:                                              ; preds = %b7
  unreachable

b13:                                              ; preds = %b7
  unreachable

b14:                                              ; preds = %b7
  %v2 = call ptr @f3(i32 0) #0
  br label %b15

b15:                                              ; preds = %b15, %b14
  %v4 = fadd double undef, undef
  br i1 undef, label %b16, label %b15

b16:                                              ; preds = %b15
  switch i32 undef, label %b18 [
    i32 0, label %b19
    i32 2, label %b17
  ]

b17:                                              ; preds = %b16
  %v7 = or i32 0, 16
  %v8 = getelementptr i8, ptr %v0, i32 %v7
  %v10 = load double, ptr undef, align 8, !tbaa !0
  %v11 = fcmp olt double -1.000000e+11, %v10
  %v12 = select i1 %v11, double %v10, double -1.000000e+11
  %v13 = load double, ptr %v0, align 8, !tbaa !0
  %v14 = fcmp olt double -1.000000e+11, %v13
  %v15 = select i1 %v14, double %v13, double -1.000000e+11
  %v16 = load double, ptr %v8, align 8, !tbaa !0
  %v17 = fcmp olt double -1.000000e+11, %v16
  %v18 = select i1 %v17, double %v16, double -1.000000e+11
  %v19 = fcmp ogt double 1.000000e+11, %v13
  %v20 = select i1 %v19, double %v13, double 1.000000e+11
  %v21 = fcmp ogt double 1.000000e+11, %v16
  %v22 = select i1 %v21, double %v16, double 1.000000e+11
  br label %b18

b18:                                              ; preds = %b17, %b16
  %v23 = phi double [ %v12, %b17 ], [ -1.000000e+11, %b16 ]
  %v24 = phi double [ %v15, %b17 ], [ -1.000000e+11, %b16 ]
  %v25 = phi double [ %v20, %b17 ], [ 1.000000e+11, %b16 ]
  %v26 = phi double [ %v18, %b17 ], [ -1.000000e+11, %b16 ]
  %v27 = phi double [ %v22, %b17 ], [ 1.000000e+11, %b16 ]
  %v28 = load double, ptr undef, align 8, !tbaa !0
  %v29 = select i1 undef, double %v28, double %v23
  %v30 = load double, ptr null, align 8, !tbaa !0
  %v31 = select i1 undef, double %v30, double %v24
  %v32 = load double, ptr undef, align 8, !tbaa !0
  %v33 = select i1 undef, double %v32, double %v26
  %v34 = select i1 undef, double %v30, double %v25
  %v35 = select i1 undef, double %v32, double %v27
  br i1 false, label %b20, label %b19

b19:                                              ; preds = %b19, %b18, %b16
  %v36 = phi double [ %v75, %b19 ], [ -1.000000e+11, %b16 ], [ %v29, %b18 ]
  %v37 = phi double [ %v81, %b19 ], [ 1.000000e+11, %b16 ], [ undef, %b18 ]
  %v38 = phi double [ %v78, %b19 ], [ -1.000000e+11, %b16 ], [ %v31, %b18 ]
  %v39 = phi double [ %v82, %b19 ], [ 1.000000e+11, %b16 ], [ %v34, %b18 ]
  %v40 = phi double [ %v80, %b19 ], [ -1.000000e+11, %b16 ], [ %v33, %b18 ]
  %v41 = phi double [ %v84, %b19 ], [ 1.000000e+11, %b16 ], [ %v35, %b18 ]
  %v44 = load double, ptr null, align 8, !tbaa !0
  %v45 = select i1 undef, double %v44, double %v36
  %v46 = load double, ptr %v0, align 8, !tbaa !0
  %v47 = select i1 undef, double %v46, double %v38
  %v48 = load double, ptr undef, align 8, !tbaa !0
  %v49 = select i1 undef, double %v48, double %v40
  %v50 = select i1 undef, double %v44, double %v37
  %v51 = fcmp ogt double %v39, %v46
  %v52 = select i1 %v51, double %v46, double %v39
  %v53 = select i1 undef, double %v48, double %v41
  %v54 = load double, ptr null, align 8, !tbaa !0
  %v55 = select i1 undef, double %v54, double %v45
  %v56 = load double, ptr undef, align 8, !tbaa !0
  %v57 = select i1 undef, double %v56, double %v47
  %v58 = load double, ptr undef, align 8, !tbaa !0
  %v59 = select i1 undef, double %v58, double %v49
  %v60 = select i1 undef, double %v54, double %v50
  %v61 = select i1 undef, double %v56, double %v52
  %v62 = select i1 false, double %v58, double %v53
  %v63 = load double, ptr undef, align 8, !tbaa !0
  %v64 = select i1 undef, double %v63, double %v55
  %v65 = load double, ptr undef, align 8, !tbaa !0
  %v66 = select i1 undef, double %v65, double %v57
  %v67 = load double, ptr null, align 8, !tbaa !0
  %v68 = select i1 undef, double %v67, double %v59
  %v69 = fcmp ogt double %v60, %v63
  %v70 = select i1 %v69, double %v63, double %v60
  %v71 = select i1 false, double %v65, double %v61
  %v72 = select i1 false, double %v67, double %v62
  %v73 = load double, ptr null, align 8, !tbaa !0
  %v74 = fcmp olt double %v64, %v73
  %v75 = select i1 %v74, double %v73, double %v64
  %v76 = load double, ptr null, align 8, !tbaa !0
  %v77 = fcmp olt double %v66, %v76
  %v78 = select i1 %v77, double %v76, double %v66
  %v79 = fcmp olt double %v68, 0.000000e+00
  %v80 = select i1 %v79, double 0.000000e+00, double %v68
  %v81 = select i1 undef, double %v73, double %v70
  %v82 = select i1 undef, double %v76, double %v71
  %v83 = fcmp ogt double %v72, 0.000000e+00
  %v84 = select i1 %v83, double 0.000000e+00, double %v72
  br i1 false, label %b20, label %b19

b20:                                              ; preds = %b19, %b18, %b6
  unreachable
}

declare i32 @f1(...)

; Function Attrs: nounwind
declare noalias ptr @f2() #0

declare ptr @f3(...)

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"double", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
