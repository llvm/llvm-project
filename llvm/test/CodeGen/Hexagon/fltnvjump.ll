; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s
; We do not want to see a new value compare after the convert
; CHECK: r{{[0-9]+}} = convert_df2w
; CHECK-NOT: if (!cmp.eq(r{{[0-9]+}}.new,r{{[0-9]+}})jump
; r3 = convert_df2w(r1:0):chop
; if (!cmp.eq(r3.new, r2)) jump:nt .LBB0_13

target triple = "hexagon"

%s.0 = type { %s.1, ptr, ptr }
%s.1 = type { i16, i16, i32 }
%s.2 = type { i8, i32, i32, i16, i16, i16, i32, i8, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, ptr }
%s.3 = type { [2 x i16], i16, i16, i16, i16, [13 x i16], i16, i16, [2 x ptr], [25 x i16], [49 x i16], [6 x i16], [49 x i16] }

@g0 = internal constant %s.0 { %s.1 { i16 705, i16 0, i32 16 }, ptr @g1, ptr @g2 }, align 4
@g1 = private unnamed_addr constant [110 x i8] c"Assertion ............................................................................................ failed\00", align 1
@g2 = private unnamed_addr constant [13 x i8] c"............\00", align 1

define signext i16 @f0(ptr %a0) #0 {
b0:
  %v0 = alloca i16, align 2
  %v1 = alloca i16, align 2
  %v2 = getelementptr inbounds %s.2, ptr %a0, i32 0, i32 19
  %v3 = load ptr, ptr %v2, align 4, !tbaa !0
  %v4 = getelementptr inbounds %s.3, ptr %v3, i32 0, i32 12, i32 0
  %v5 = getelementptr inbounds %s.3, ptr %v3, i32 0, i32 2
  %v6 = call signext i16 @f1(ptr %v4, ptr %v5, ptr %a0)
  %v7 = icmp eq i16 %v6, 0
  br i1 %v7, label %b1, label %b13

b1:                                               ; preds = %b0
  %v8 = getelementptr inbounds %s.2, ptr %a0, i32 0, i32 11
  %v9 = load i16, ptr %v8, align 2, !tbaa !4
  %v10 = sext i16 %v9 to i32
  %v11 = load i16, ptr %v5, align 2, !tbaa !4
  %v12 = sext i16 %v11 to i32
  %v13 = call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 %v10, i32 %v12)
  %v14 = trunc i32 %v13 to i16
  %v15 = icmp sgt i16 %v14, 0
  br i1 %v15, label %b13, label %b2

b2:                                               ; preds = %b1
  %v16 = getelementptr inbounds %s.3, ptr %v3, i32 0, i32 8, i32 1
  %v17 = load ptr, ptr %v16, align 4, !tbaa !0
  call void @f2(ptr %v17, ptr %v1, ptr %v4, i16 signext %v11, i16 signext %v9)
  %v18 = getelementptr inbounds %s.3, ptr %v3, i32 0, i32 8, i32 0
  %v19 = load ptr, ptr %v18, align 4, !tbaa !0
  %v20 = load ptr, ptr %v16, align 4, !tbaa !0
  %v21 = load i16, ptr %v1, align 2, !tbaa !4
  call void @f3(ptr %v19, ptr %v0, ptr %v20, i16 signext %v21)
  %v22 = load i16, ptr %v0, align 2, !tbaa !4
  store i16 %v22, ptr %v3, align 2, !tbaa !4
  %v24 = load i16, ptr %v1, align 2, !tbaa !4
  %v25 = getelementptr inbounds %s.3, ptr %v3, i32 0, i32 0, i32 1
  store i16 %v24, ptr %v25, align 2, !tbaa !4
  %v26 = load i16, ptr %v0, align 2, !tbaa !4
  %v27 = sext i16 %v26 to i32
  %v28 = icmp slt i16 %v26, 1
  br i1 %v28, label %b13, label %b3

b3:                                               ; preds = %b2
  %v29 = call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 48, i32 1)
  %v30 = call i32 @llvm.hexagon.A2.sath(i32 %v29)
  %v31 = shl i32 %v30, 16
  %v32 = ashr exact i32 %v31, 16
  %v33 = call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 %v27, i32 %v32)
  %v34 = trunc i32 %v33 to i16
  %v35 = icmp sgt i16 %v34, 0
  br i1 %v35, label %b13, label %b4

b4:                                               ; preds = %b3
  %v36 = load ptr, ptr %v18, align 4, !tbaa !0
  %v37 = load i16, ptr %v36, align 2, !tbaa !4
  %v38 = getelementptr inbounds i16, ptr %v36, i32 %v27
  %v39 = load i16, ptr %v38, align 2, !tbaa !4
  %v40 = sext i16 %v37 to i32
  %v41 = call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 %v40, i32 32)
  %v42 = trunc i32 %v41 to i16
  %v43 = icmp sgt i16 %v42, 0
  br i1 %v43, label %b13, label %b5

b5:                                               ; preds = %b4
  %v44 = sext i16 %v39 to i32
  %v45 = call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 %v40, i32 %v44)
  %v46 = and i32 %v45, 32768
  %v47 = icmp eq i32 %v46, 0
  br i1 %v47, label %b13, label %b6

b6:                                               ; preds = %b5
  %v48 = load i16, ptr %v1, align 2, !tbaa !4
  %v49 = sext i16 %v48 to i32
  %v50 = load ptr, ptr %v16, align 4, !tbaa !0
  %v51 = getelementptr inbounds i16, ptr %v50, i32 %v49
  %v52 = load i16, ptr %v51, align 2, !tbaa !4
  %v53 = getelementptr inbounds %s.2, ptr %a0, i32 0, i32 14
  %v54 = load i16, ptr %v53, align 2, !tbaa !4
  %v55 = icmp eq i16 %v54, 0
  br i1 %v55, label %b7, label %b8

b7:                                               ; preds = %b6
  %v56 = getelementptr inbounds %s.3, ptr %v3, i32 0, i32 1
  store i16 1, ptr %v56, align 2, !tbaa !4
  br label %b11

b8:                                               ; preds = %b6
  %v57 = load i16, ptr %v50, align 2, !tbaa !4
  %v58 = sext i16 %v57 to i32
  %v59 = sext i16 %v52 to i32
  %v60 = call signext i16 @f4(i32 %v58, i32 %v59)
  %v61 = sext i16 %v60 to i32
  %v62 = call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v61, i32 2)
  %v63 = call i32 @llvm.hexagon.A2.sath(i32 %v62)
  %v64 = shl i32 %v63, 16
  %v65 = ashr exact i32 %v64, 16
  %v66 = load i16, ptr %v53, align 2, !tbaa !4
  %v67 = sext i16 %v66 to i32
  %v68 = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32 1024, i32 %v65, i32 %v67)
  %v69 = shl i32 %v68, 16
  %v70 = ashr exact i32 %v69, 16
  %v71 = call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v70, i32 1)
  %v72 = call i32 @llvm.hexagon.A2.sath(i32 %v71)
  %v73 = shl i32 %v72, 16
  %v74 = ashr exact i32 %v73, 16
  %v75 = call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v74, i32 10)
  %v76 = call i32 @llvm.hexagon.A2.sath(i32 %v75)
  %v77 = shl i32 %v76, 16
  %v78 = ashr exact i32 %v77, 16
  %v79 = sitofp i16 %v66 to float
  %v80 = sitofp i16 %v52 to float
  %v81 = sitofp i16 %v57 to float
  %v82 = fdiv float %v80, %v81
  %v83 = call float @f7(float %v82, i32 0)
  %v84 = fmul float %v79, %v83
  %v85 = fdiv float %v84, 0x3FE62E4300000000
  %v86 = fpext float %v85 to double
  %v87 = fadd double %v86, 5.000000e-01
  %v88 = fptosi double %v87 to i32
  %v89 = icmp eq i32 %v78, %v88
  br i1 %v89, label %b10, label %b9

b9:                                               ; preds = %b8
  call void @f5(ptr @g0) #2
  unreachable

b10:                                              ; preds = %b8
  %v90 = trunc i32 %v76 to i16
  %v91 = icmp eq i32 %v78, 0
  %v92 = select i1 %v91, i16 1, i16 %v90
  %v93 = getelementptr inbounds %s.3, ptr %v3, i32 0, i32 1
  store i16 %v92, ptr %v93, align 2, !tbaa !4
  br label %b11

b11:                                              ; preds = %b10, %b7
  %v94 = phi i16 [ %v92, %b10 ], [ 1, %b7 ]
  %v95 = getelementptr inbounds %s.3, ptr %v3, i32 0, i32 7
  store i16 %v94, ptr %v95, align 2, !tbaa !4
  %v96 = sext i16 %v94 to i32
  %v97 = call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 %v96, i32 5)
  %v98 = trunc i32 %v97 to i16
  %v99 = icmp sgt i16 %v98, 0
  br i1 %v99, label %b13, label %b12

b12:                                              ; preds = %b11
  %v100 = getelementptr inbounds %s.3, ptr %v3, i32 0, i32 11, i32 0
  %v101 = load ptr, ptr %v18, align 4, !tbaa !0
  %v102 = load i16, ptr %v0, align 2, !tbaa !4
  call void @f6(ptr %v100, i16 signext %v94, ptr %v101, i16 signext %v102)
  %v103 = getelementptr inbounds %s.3, ptr %v3, i32 0, i32 3
  store i16 %v37, ptr %v103, align 2, !tbaa !4
  %v104 = getelementptr inbounds %s.3, ptr %v3, i32 0, i32 4
  store i16 %v39, ptr %v104, align 2, !tbaa !4
  br label %b13

b13:                                              ; preds = %b12, %b11, %b5, %b4, %b3, %b2, %b1, %b0
  %v105 = phi i16 [ 0, %b12 ], [ -1, %b1 ], [ -1, %b0 ], [ -1, %b3 ], [ -1, %b2 ], [ -1, %b5 ], [ -1, %b4 ], [ -1, %b11 ]
  ret i16 %v105
}

declare signext i16 @f1(ptr, ptr, ptr) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32, i32) #1

declare void @f2(ptr, ptr, ptr, i16 signext, i16 signext) #0

declare void @f3(ptr, ptr, ptr, i16 signext) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.sath(i32) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.S2.asr.r.r.sat(i32, i32) #1

declare signext i16 @f4(i32, i32) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s1(i32, i32, i32) #1

; Function Attrs: noreturn
declare void @f5(ptr) #2

declare void @f6(ptr, i16 signext, ptr, i16 signext) #0

declare float @f7(float, i32) #0

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }
attributes #2 = { noreturn }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"short", !2}
