; RUN: opt -S -passes='dxil-data-scalarization' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=SCHECK,CHECK
; RUN: opt -S -passes='dxil-data-scalarization,dxil-flatten-arrays' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=FCHECK,CHECK

; CHECK-LABEL: alloca_2d__vec_test
define void @alloca_2d__vec_test() {
  ; SCHECK:  alloca [2 x [4 x i32]], align 16
  ; FCHECK:  alloca [8 x i32], align 16
  ; CHECK: ret void
  %1 = alloca [2 x <4 x i32>], align 16
  ret void
}

; CHECK-LABEL: alloca_4d__vec_test
define void @alloca_4d__vec_test() {
  ; SCHECK:  alloca [2 x [2 x [2 x [2 x i32]]]], align 16
  ; FCHECK:  alloca [16 x i32], align 16
  ; CHECK: ret void
  %1 = alloca [2 x [2 x [2 x <2 x i32>]]], align 16
  ret void
}

; CHECK-LABEL: alloca_vec_test
define void @alloca_vec_test() {
  ; CHECK:  alloca [4 x i32], align 16
  ; CHECK: ret void
  %1 = alloca <4 x i32>, align 16
  ret void
}

; CHECK-LABEL: alloca_2d_gep_test
define void @alloca_2d_gep_test() {
  ; SCHECK:  [[alloca_val:%.*]] = alloca [2 x [2 x i32]], align 16
  ; FCHECK:  [[alloca_val:%.*]] = alloca [4 x i32], align 16
  ; CHECK: [[tid:%.*]] = tail call i32 @llvm.dx.thread.id(i32 0)
  ; SCHECK: [[gep:%.*]] = getelementptr inbounds nuw [2 x [2 x i32]], ptr [[alloca_val]], i32 0, i32 [[tid]]
  ; FCHECK: [[flatidx_mul:%.*]] = mul i32 [[tid]], 2
  ; FCHECK: [[flatidx:%.*]] = add i32 0, [[flatidx_mul]]
  ; FCHECK: [[gep:%.*]] = getelementptr inbounds nuw [4 x i32], ptr [[alloca_val]], i32 0, i32 [[flatidx]]
  ; CHECK: ret void
  %1 = alloca [2 x <2 x i32>], align 16
  %2 = tail call i32 @llvm.dx.thread.id(i32 0)
  %3 = getelementptr inbounds nuw [2 x <2 x i32>], ptr %1, i32 0, i32 %2
  ret void
}
