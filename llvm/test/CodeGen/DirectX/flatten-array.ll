
; RUN: opt -S -dxil-flatten-arrays  %s | FileCheck %s

; CHECK-LABEL: alloca_2d_test
define void @alloca_2d_test ()  {
; CHECK-NEXT:    alloca [9 x i32], align 4
; CHECK-NEXT:    ret void
;
  %1 = alloca [3 x [3 x i32]], align 4
  ret void
}

; CHECK-LABEL: alloca_3d_test
define void @alloca_3d_test ()  {
; CHECK-NEXT:    alloca [8 x i32], align 4
; CHECK-NEXT:    ret void
;
  %1 = alloca [2 x[2 x [2 x i32]]], align 4
  ret void
}

; CHECK-LABEL: alloca_4d_test
define void @alloca_4d_test ()  {
; CHECK-NEXT:    alloca [16 x i32], align 4
; CHECK-NEXT:    ret void
;
  %1 = alloca [2x[2 x[2 x [2 x i32]]]], align 4
  ret void
}

; CHECK-LABEL: gep_2d_test
define void @gep_2d_test ()  {
    ; CHECK: [[a:%.*]] = alloca [9 x i32], align 4
    ; CHECK-COUNT-9: getelementptr inbounds [9 x i32], ptr [[a]], i32 {{[0-8]}}
    ; CHECK-NEXT:    ret void
    %1 = alloca [3 x [3 x i32]], align 4
    %g2d0 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* %1, i32 0, i32 0
    %g1d_1 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d0, i32 0, i32 0
    %g1d_2 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d0, i32 0, i32 1
    %g1d_3 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d0, i32 0, i32 2
    %g2d1 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* %1, i32 0, i32 1
    %g1d1_1 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d1, i32 0, i32 0
    %g1d1_2 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d1, i32 0, i32 1
    %g1d1_3 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d1, i32 0, i32 2
    %g2d2 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* %1, i32 0, i32 2
    %g1d2_1 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d2, i32 0, i32 0
    %g1d2_2 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d2, i32 0, i32 1
    %g1d2_3 = getelementptr inbounds [3 x i32], [3 x i32]* %g2d2, i32 0, i32 2
    
    ret void
}

; CHECK-LABEL: gep_3d_test
define void @gep_3d_test ()  {
    ; CHECK: [[a:%.*]] = alloca [8 x i32], align 4
    ; CHECK-COUNT-8: getelementptr inbounds [8 x i32], ptr [[a]], i32 {{[0-7]}}
    ; CHECK-NEXT:    ret void
    %1 = alloca [2 x[2 x [2 x i32]]], align 4
    %g3d0 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %1, i32 0, i32 0
    %g2d0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0, i32 0, i32 0
    %g1d_1 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0, i32 0, i32 0
    %g1d_2 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0, i32 0, i32 1
    %g2d1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0, i32 0, i32 1
    %g1d1_1 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d1, i32 0, i32 0
    %g1d1_2 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d1, i32 0, i32 1
    %g3d1 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %1, i32 0, i32 1
    %g2d2 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d1, i32 0, i32 0
    %g1d2_1 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d2, i32 0, i32 0
    %g1d2_2 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d2, i32 0, i32 1
    %g2d3 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d1, i32 0, i32 1
    %g1d3_1 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d3, i32 0, i32 0
    %g1d3_2 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d3, i32 0, i32 1
    ret void
}

; CHECK-LABEL: gep_4d_test
define void @gep_4d_test ()  {
    ; CHECK: [[a:%.*]] = alloca [16 x i32], align 4
    ; CHECK-COUNT-16: getelementptr inbounds [16 x i32], ptr [[a]], i32 {{[0-9]|1[0-5]}}
    ; CHECK-NEXT:    ret void
    %1 = alloca [2x[2 x[2 x [2 x i32]]]], align 4
    %g4d0 = getelementptr inbounds [2x[2 x[2 x [2 x i32]]]], [2x[2 x[2 x [2 x i32]]]]* %1, i32 0, i32 0
    %g3d0 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %g4d0, i32 0, i32 0
    %g2d0_0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0, i32 0, i32 0
    %g1d_0 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_0, i32 0, i32 0
    %g1d_1 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_0, i32 0, i32 1
    %g2d0_1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0, i32 0, i32 1
    %g1d_2 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_1, i32 0, i32 0
    %g1d_3 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_1, i32 0, i32 1
    %g3d1 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %g4d0, i32 0, i32 1
    %g2d0_2 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d1, i32 0, i32 0
    %g1d_4 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_2, i32 0, i32 0
    %g1d_5 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_2, i32 0, i32 1
    %g2d1_2 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d1, i32 0, i32 1
    %g1d_6 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d1_2, i32 0, i32 0
    %g1d_7 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d1_2, i32 0, i32 1
    %g4d1 = getelementptr inbounds [2x[2 x[2 x [2 x i32]]]], [2x[2 x[2 x [2 x i32]]]]* %1, i32 0, i32 1
    %g3d0_1 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %g4d1, i32 0, i32 0
    %g2d0_3 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0_1, i32 0, i32 0
    %g1d_8 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_3, i32 0, i32 0
    %g1d_9 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_3, i32 0, i32 1
    %g2d0_4 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d0_1, i32 0, i32 1
    %g1d_10 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_4, i32 0, i32 0
    %g1d_11 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_4, i32 0, i32 1
    %g3d1_1 = getelementptr inbounds [2 x[2 x [2 x i32]]], [2 x[2 x [2 x i32]]]* %g4d1, i32 0, i32 1
    %g2d0_5 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d1_1, i32 0, i32 0
    %g1d_12 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_5, i32 0, i32 0
    %g1d_13 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d0_5, i32 0, i32 1
    %g2d1_3 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g3d1_1, i32 0, i32 1
    %g1d_14 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d1_3, i32 0, i32 0
    %g1d_15 = getelementptr inbounds [2 x i32], [2 x i32]* %g2d1_3, i32 0, i32 1
    ret void
}


@a = internal global [2 x [3 x [4 x i32]]] [[3 x [4 x i32]] [[4 x i32] [i32 0, i32 1, i32 2, i32 3], 
                                                             [4 x i32] [i32 4, i32 5, i32 6, i32 7], 
                                                             [4 x i32] [i32 8, i32 9, i32 10, i32 11]], 
                                            [3 x [4 x i32]] [[4 x i32] [i32 12, i32 13, i32 14, i32 15], 
                                                             [4 x i32] [i32 16, i32 17, i32 18, i32 19], 
                                                             [4 x i32] [i32 20, i32 21, i32 22, i32 23]]], align 4

@b = internal global [2 x [3 x [4 x i32]]] zeroinitializer, align 16

define void @global_gep_load() {
  ; CHECK: [[GEP_PTR:%.*]] = getelementptr inbounds [24 x i32], ptr @a.1dim, i32 6
  ; CHECK: load i32, ptr [[GEP_PTR]], align 4
  ; CHECK-NEXT:    ret void
  %1 = getelementptr inbounds [2 x [3 x [4 x i32]]], [2 x [3 x [4 x i32]]]* @a, i32 0, i32 0
  %2 = getelementptr inbounds [3 x [4 x i32]], [3 x [4 x i32]]* %1, i32 0, i32 1
  %3 = getelementptr inbounds [4 x i32], [4 x i32]* %2, i32 0, i32 2
  %4 = load i32, i32* %3, align 4
  ret void
}

define void @global_gep_load_index(i32 %row, i32 %col, i32 %timeIndex) {
; CHECK-LABEL: define void @global_gep_load_index(
; CHECK-SAME: i32 [[ROW:%.*]], i32 [[COL:%.*]], i32 [[TIMEINDEX:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = mul i32 [[TIMEINDEX]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = add i32 0, [[TMP1]]
; CHECK-NEXT:    [[TMP3:%.*]] = mul i32 [[COL]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = add i32 [[TMP2]], [[TMP3]]
; CHECK-NEXT:    [[TMP5:%.*]] = mul i32 [[ROW]], 12
; CHECK-NEXT:    [[TMP6:%.*]] = add i32 [[TMP4]], [[TMP5]]
; CHECK-NEXT:    [[DOTFLAT:%.*]] = getelementptr inbounds [24 x i32], ptr @a.1dim, i32 [[TMP6]]
; CHECK-NOT: getelementptr inbounds [2 x [3 x [4 x i32]]]{{.*}}
; CHECK-NOT: getelementptr inbounds [3 x [4 x i32]]{{.*}}
; CHECK-NOT: getelementptr inbounds [4 x i32]{{.*}}
; CHECK-NEXT:    [[TMP7:%.*]] = load i32, ptr [[DOTFLAT]], align 4
; CHECK-NEXT:    ret void
;
  %1 = getelementptr inbounds [2 x [3 x [4 x i32]]], [2 x [3 x [4 x i32]]]* @a, i32 0, i32 %row
  %2 = getelementptr inbounds [3 x [4 x i32]], [3 x [4 x i32]]* %1, i32 0, i32 %col
  %3 = getelementptr inbounds [4 x i32], [4 x i32]* %2, i32 0, i32 %timeIndex
  %4 = load i32, i32* %3, align 4
  ret void
}

define void @global_incomplete_gep_chain(i32 %row, i32 %col) {
; CHECK-LABEL: define void @global_incomplete_gep_chain(
; CHECK-SAME: i32 [[ROW:%.*]], i32 [[COL:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = mul i32 [[COL]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = add i32 0, [[TMP1]]
; CHECK-NEXT:    [[TMP3:%.*]] = mul i32 [[ROW]], 3
; CHECK-NEXT:    [[TMP4:%.*]] = add i32 [[TMP2]], [[TMP3]]
; CHECK-NEXT:    [[DOTFLAT:%.*]] = getelementptr inbounds [24 x i32], ptr @a.1dim, i32 [[TMP4]]
; CHECK-NOT: getelementptr inbounds [2 x [3 x [4 x i32]]]{{.*}}
; CHECK-NOT: getelementptr inbounds [3 x [4 x i32]]{{.*}}
; CHECK-NOT: getelementptr inbounds [4 x i32]{{.*}}
; CHECK-NEXT:    [[TMP5:%.*]] = load i32, ptr [[DOTFLAT]], align 4
; CHECK-NEXT:    ret void
;
  %1 = getelementptr inbounds [2 x [3 x [4 x i32]]], [2 x [3 x [4 x i32]]]* @a, i32 0, i32 %row
  %2 = getelementptr inbounds [3 x [4 x i32]], [3 x [4 x i32]]* %1, i32 0, i32 %col
  %4 = load i32, i32* %2, align 4
  ret void
}

define void @global_gep_store() {
  ; CHECK: [[GEP_PTR:%.*]] = getelementptr inbounds [24 x i32], ptr @b.1dim, i32 13
  ; CHECK:  store i32 1, ptr [[GEP_PTR]], align 4
  ; CHECK-NEXT:    ret void
  %1 = getelementptr inbounds [2 x [3 x [4 x i32]]], [2 x [3 x [4 x i32]]]* @b, i32 0, i32 1
  %2 = getelementptr inbounds [3 x [4 x i32]], [3 x [4 x i32]]* %1, i32 0, i32 0
  %3 = getelementptr inbounds [4 x i32], [4 x i32]* %2, i32 0, i32 1
  store i32 1, i32* %3, align 4
  ret void
}

; Make sure we don't try to walk the body of a function declaration.
declare void @opaque_function()
