
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
    ; CHECK-COUNT-9: getelementptr inbounds [9 x i32], ptr [[a]], i32 0, i32 {{[0-8]}}
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
    ; CHECK-COUNT-8: getelementptr inbounds [8 x i32], ptr [[a]], i32 0, i32 {{[0-7]}}
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
    ; CHECK-COUNT-16: getelementptr inbounds [16 x i32], ptr [[a]], i32 0, i32 {{[0-9]|1[0-5]}}
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
  ; CHECK-LABEL: define void @global_gep_load(
  ; CHECK: {{.*}} = load i32, ptr getelementptr inbounds ([24 x i32], ptr @a.1dim, i32 0, i32 6), align 4
  ; CHECK-NEXT:    ret void
  %1 = getelementptr inbounds [2 x [3 x [4 x i32]]], [2 x [3 x [4 x i32]]]* @a, i32 0, i32 0
  %2 = getelementptr inbounds [3 x [4 x i32]], [3 x [4 x i32]]* %1, i32 0, i32 1
  %3 = getelementptr inbounds [4 x i32], [4 x i32]* %2, i32 0, i32 2
  %4 = load i32, i32* %3, align 4
  ret void
}

define void @global_nested_geps() {
  ; CHECK-LABEL: define void @global_nested_geps(
  ; CHECK: {{.*}} = load i32, ptr getelementptr inbounds ([24 x i32], ptr @a.1dim, i32 0, i32 6), align 4
  ; CHECK-NEXT:    ret void
  %1 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* getelementptr inbounds ([3 x [4 x i32]], [3 x [4 x i32]]* getelementptr inbounds ([2 x [3 x [4 x i32]]], [2 x [3 x [4 x i32]]]* @a, i32 0, i32 0), i32 0, i32 1), i32 0, i32 2), align 4
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
; CHECK-NEXT:    [[DOTFLAT:%.*]] = getelementptr inbounds [24 x i32], ptr @a.1dim, i32 0, i32 [[TMP6]]
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
; CHECK-NEXT:    [[TMP1:%.*]] = mul i32 [[COL]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = add i32 0, [[TMP1]]
; CHECK-NEXT:    [[TMP3:%.*]] = mul i32 [[ROW]], 12
; CHECK-NEXT:    [[TMP4:%.*]] = add i32 [[TMP2]], [[TMP3]]
; CHECK-NEXT:    [[DOTFLAT:%.*]] = getelementptr inbounds [24 x i32], ptr @a.1dim, i32 0, i32 [[TMP4]]
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
  ; CHECK: store i32 1, ptr getelementptr inbounds ([24 x i32], ptr @b.1dim, i32 0, i32 13), align 4
  ; CHECK-NEXT:    ret void
  %1 = getelementptr inbounds [2 x [3 x [4 x i32]]], [2 x [3 x [4 x i32]]]* @b, i32 0, i32 1
  %2 = getelementptr inbounds [3 x [4 x i32]], [3 x [4 x i32]]* %1, i32 0, i32 0
  %3 = getelementptr inbounds [4 x i32], [4 x i32]* %2, i32 0, i32 1
  store i32 1, i32* %3, align 4
  ret void
}

@g = local_unnamed_addr addrspace(3) global [2 x [2 x float]] zeroinitializer, align 4
define void @two_index_gep() {
  ; CHECK-LABEL: define void @two_index_gep(
  ; CHECK: [[THREAD_ID:%.*]] =  tail call i32 @llvm.dx.thread.id(i32 0)
  ; CHECK-NEXT: [[MUL:%.*]] = mul i32 [[THREAD_ID]], 2
  ; CHECK-NEXT: [[ADD:%.*]] = add i32 1, [[MUL]]
  ; CHECK-NEXT: [[GEP_PTR:%.*]] = getelementptr inbounds nuw [4 x float], ptr addrspace(3) @g.1dim, i32 0, i32 [[ADD]]
  ; CHECK-NEXT: load float, ptr addrspace(3) [[GEP_PTR]], align 4
  ; CHECK-NEXT: ret void
  %1 = tail call i32 @llvm.dx.thread.id(i32 0)
  %2 = getelementptr inbounds nuw [2 x [2 x float]], ptr addrspace(3) @g, i32 0, i32 %1, i32 1
  %3 = load float, ptr addrspace(3) %2, align 4
  ret void
}

define void @two_index_gep_const() {
  ; CHECK-LABEL: define void @two_index_gep_const(
  ; CHECK-NEXT: load float, ptr addrspace(3) getelementptr inbounds nuw ([4 x float], ptr addrspace(3) @g.1dim, i32 0, i32 3), align 4
  ; CHECK-NEXT: ret void
  %1 = getelementptr inbounds nuw [2 x [2 x float]], ptr addrspace(3) @g, i32 0, i32 1, i32 1
  %3 = load float, ptr addrspace(3) %1, align 4
  ret void
}

define void @zero_index_global() {
  ; CHECK-LABEL: define void @zero_index_global(
  ; CHECK-NEXT: [[GEP:%.*]] = getelementptr inbounds nuw [4 x float], ptr addrspace(3) @g.1dim, i32 0, i32 0
  ; CHECK-NEXT: load float, ptr addrspace(3) [[GEP]], align 4
  ; CHECK-NEXT: ret void
  %1 = getelementptr inbounds nuw [2 x [2 x float]], ptr addrspace(3) @g, i32 0, i32 0, i32 0
  %2 = load float, ptr addrspace(3) %1, align 4
  ret void
}

; Note: A ConstantExpr GEP with all 0 indices is equivalent to the pointer
; operand of the GEP. Therefore the visitLoadInst will not see the pointer operand
; as a ConstantExpr GEP and will not create a GEP instruction to be visited.
; The later dxil-legalize pass will insert a GEP in this instance.
define void @zero_index_global_const() {
  ; CHECK-LABEL: define void @zero_index_global_const(
  ; CHECK-NEXT: load float, ptr addrspace(3) @g.1dim, align 4
  ; CHECK-NEXT: ret void
  %1 = load float, ptr addrspace(3) getelementptr inbounds nuw ([2 x [2 x float]], ptr addrspace(3) @g, i32 0, i32 0, i32 0), align 4
  ret void
}

define void @gep_4d_index_test()  {
    ; CHECK-LABEL: gep_4d_index_test
    ; CHECK: [[a:%.*]] = alloca [16 x i32], align 4
    ; CHECK-NEXT: getelementptr inbounds [16 x i32], ptr %.1dim, i32 0, i32 1
    ; CHECK-NEXT: getelementptr inbounds [16 x i32], ptr %.1dim, i32 0, i32 3
    ; CHECK-NEXT: getelementptr inbounds [16 x i32], ptr %.1dim, i32 0, i32 7
    ; CHECK-NEXT: getelementptr inbounds [16 x i32], ptr %.1dim, i32 0, i32 15
    ; CHECK-NEXT:    ret void
    %1 = alloca [2x[2 x[2 x [2 x i32]]]], align 4
    %2 = getelementptr inbounds [2 x [2 x[2 x [2 x i32]]]], [2 x [2 x [2 x [2 x i32]]]]* %1, i32 0, i32 0, i32 0, i32 0, i32 1
    %3 = getelementptr inbounds [2 x [2 x[2 x [2 x i32]]]], [2 x [2 x [2 x [2 x i32]]]]* %1, i32 0, i32 0, i32 0, i32 1, i32 1
    %4 = getelementptr inbounds [2 x [2 x[2 x [2 x i32]]]], [2 x [2 x [2 x [2 x i32]]]]* %1, i32 0, i32 0, i32 1, i32 1, i32 1
    %5 = getelementptr inbounds [2 x [2 x[2 x [2 x i32]]]], [2 x [2 x [2 x [2 x i32]]]]* %1, i32 0, i32 1, i32 1, i32 1, i32 1
    ret void
}

define void @gep_4d_index_and_gep_chain_mixed() {
  ; CHECK-LABEL: gep_4d_index_and_gep_chain_mixed
  ; CHECK-NEXT: [[ALLOCA:%.*]] = alloca [16 x i32], align 4
  ; CHECK-COUNT-16: getelementptr inbounds [16 x i32], ptr [[ALLOCA]], i32 0, i32 {{[0-9]|1[0-5]}}
  ; CHECK-NEXT: ret void
  %1 = alloca [2x[2 x[2 x [2 x i32]]]], align 4
  %a4d0_0 = getelementptr inbounds [2 x [2 x [2 x [2 x i32]]]], [2 x [2 x[2 x [2 x i32]]]]* %1, i32 0, i32 0, i32 0
  %a2d0_0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %a4d0_0, i32 0, i32 0, i32 0
  %a2d0_1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %a4d0_0, i32 0, i32 0, i32 1
  %a2d1_0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %a4d0_0, i32 0, i32 1, i32 0
  %a2d1_1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %a4d0_0, i32 0, i32 1, i32 1
  %b4d0_1 = getelementptr inbounds [2 x [2 x [2 x [2 x i32]]]], [2 x [2 x [2 x [2 x i32]]]]* %1, i32 0, i32 0, i32 1
  %b2d0_0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %b4d0_1, i32 0, i32 0, i32 0
  %b2d0_1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %b4d0_1, i32 0, i32 0, i32 1
  %b2d1_0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %b4d0_1, i32 0, i32 1, i32 0
  %b2d1_1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %b4d0_1, i32 0, i32 1, i32 1
  %c4d1_0 = getelementptr inbounds [2 x [2 x [2 x [2 x i32]]]], [2 x [2 x [2 x [2 x i32]]]]* %1, i32 0, i32 1, i32 0
  %c2d0_0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %c4d1_0, i32 0, i32 0, i32 0
  %c2d0_1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %c4d1_0, i32 0, i32 0, i32 1
  %c2d1_0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %c4d1_0, i32 0, i32 1, i32 0
  %c2d1_1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %c4d1_0, i32 0, i32 1, i32 1
  %g4d1_1 = getelementptr inbounds [2 x [2 x [2 x [2 x i32]]]], [2 x [2 x [2 x [2 x i32]]]]* %1, i32 0, i32 1, i32 1
  %g2d0_0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g4d1_1, i32 0, i32 0, i32 0
  %g2d0_1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g4d1_1, i32 0, i32 0, i32 1
  %g2d1_0 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g4d1_1, i32 0, i32 1, i32 0
  %g2d1_1 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %g4d1_1, i32 0, i32 1, i32 1
  ret void
}

; This test demonstrates that the collapsing of GEP chains occurs regardless of
; the source element type given to the GEP. As long as the root pointer being
; indexed to is an aggregate data structure, the GEP will be flattened.
define void @gep_scalar_flatten() {
  ; CHECK-LABEL: gep_scalar_flatten
  ; CHECK-NEXT: [[ALLOCA:%.*]] = alloca [24 x i32]
  ; CHECK-NEXT: getelementptr inbounds nuw [24 x i32], ptr [[ALLOCA]], i32 0, i32 17
  ; CHECK-NEXT: getelementptr inbounds nuw [24 x i32], ptr [[ALLOCA]], i32 0, i32 17
  ; CHECK-NEXT: getelementptr inbounds nuw [24 x i32], ptr [[ALLOCA]], i32 0, i32 23
  ; CHECK-NEXT: ret void
  %a = alloca [2 x [3 x [4 x i32]]], align 4
  %i8root = getelementptr inbounds nuw i8, [2 x [3 x [4 x i32]]]* %a, i32 68 ; %a[1][1][1]
  %i32root = getelementptr inbounds nuw i32, [2 x [3 x [4 x i32]]]* %a, i32 17 ; %a[1][1][1]
  %c0 = getelementptr inbounds nuw [2 x [3 x [4 x i32]]], [2 x [3 x [4 x i32]]]* %a, i32 0, i32 1 ; %a[1]
  %c1 = getelementptr inbounds nuw i32, [3 x [4 x i32]]* %c0, i32 8 ; %a[1][2]
  %c2 = getelementptr inbounds nuw i8, [4 x i32]* %c1, i32 12 ; %a[1][2][3]
  ret void
}

define void @gep_scalar_flatten_dynamic(i32 %index) {
  ; CHECK-LABEL: gep_scalar_flatten_dynamic
  ; CHECK-SAME: i32 [[INDEX:%.*]]) {
  ; CHECK-NEXT:  [[ALLOCA:%.*]] = alloca [6 x i32], align 4
  ; CHECK-NEXT:  [[I8INDEX:%.*]] = mul i32 [[INDEX]], 12
  ; CHECK-NEXT:  [[MUL:%.*]] = mul i32 [[I8INDEX]], 1
  ; CHECK-NEXT:  [[DIV:%.*]] = lshr i32 [[MUL]], 2
  ; CHECK-NEXT:  [[ADD:%.*]] = add i32 0, [[DIV]]
  ; CHECK-NEXT:  getelementptr inbounds nuw [6 x i32], ptr [[ALLOCA]], i32 0, i32 [[ADD]]
  ; CHECK-NEXT:  [[I32INDEX:%.*]] = mul i32 [[INDEX]], 3
  ; CHECK-NEXT:  [[MUL:%.*]] = mul i32 [[I32INDEX]], 1
  ; CHECK-NEXT:  [[ADD:%.*]] = add i32 0, [[MUL]]
  ; CHECK-NEXT:  getelementptr inbounds nuw [6 x i32], ptr [[ALLOCA]], i32 0, i32 [[ADD]]
  ; CHECK-NEXT:  ret void
  ;
  %a = alloca [2 x [3 x i32]], align 4
  %i8index = mul i32 %index, 12
  %i8root = getelementptr inbounds nuw i8, [2 x [3 x i32]]* %a, i32 %i8index;
  %i32index = mul i32 %index, 3
  %i32root = getelementptr inbounds nuw i32, [2 x [3 x i32]]* %a, i32 %i32index;
  ret void
}

; Make sure we don't try to walk the body of a function declaration.
declare void @opaque_function()
