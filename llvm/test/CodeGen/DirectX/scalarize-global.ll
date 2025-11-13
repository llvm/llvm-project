; RUN: opt -S -passes='dxil-data-scalarization' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=SCHECK,CHECK
; RUN: opt -S -passes='dxil-data-scalarization,dxil-flatten-arrays' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=FCHECK,CHECK

@"arrayofVecData" = local_unnamed_addr addrspace(3) global [8 x <4 x i32>] zeroinitializer, align 16
@"vecData" = external addrspace(3) global <4 x i32>, align 4

; SCHECK: [[arrayofVecData:@arrayofVecData.*]] = local_unnamed_addr addrspace(3) global [8 x [4 x i32]] zeroinitializer, align 16
; FCHECK: [[arrayofVecData:@arrayofVecData.*]] = local_unnamed_addr addrspace(3) global [32 x i32] zeroinitializer, align 16
; CHECK: [[vecData:@vecData.*]] = external addrspace(3) global [4 x i32], align 4

; CHECK-LABEL: subtype_array_test
define <4 x i32> @subtype_array_test() {
  ; CHECK: [[tid:%.*]] = tail call i32 @llvm.dx.thread.id(i32 0)
  ; SCHECK: [[gep:%.*]] = getelementptr inbounds nuw [8 x [4 x i32]], ptr addrspace(3) [[arrayofVecData]], i32 0, i32 [[tid]]
  ; FCHECK: [[flatidx_mul:%.*]] = mul i32 [[tid]], 4
  ; FCHECK: [[flatidx:%.*]] = add i32 0, [[flatidx_mul]]
  ; FCHECK: [[gep:%.*]] = getelementptr inbounds nuw [32 x i32], ptr addrspace(3) [[arrayofVecData]], i32 0, i32 [[flatidx]]
  ; CHECK: [[x:%.*]] = load <4 x i32>, ptr addrspace(3) [[gep]], align 4
  ; CHECK: ret <4 x i32> [[x]]
  %i = tail call i32 @llvm.dx.thread.id(i32 0)
  %gep = getelementptr inbounds nuw [4 x i32], ptr addrspace(3) @"arrayofVecData", i32 %i
  %x = load <4 x i32>, ptr addrspace(3) %gep, align 4
  ret <4 x i32> %x
}

; CHECK-LABEL: subtype_vector_test
define <4 x i32> @subtype_vector_test() {
  ; CHECK: [[tid:%.*]] = tail call i32 @llvm.dx.thread.id(i32 0)
  ; SCHECK: [[gep:%.*]] = getelementptr inbounds nuw [8 x [4 x i32]], ptr addrspace(3) [[arrayofVecData]], i32 0, i32 [[tid]]
  ; FCHECK: [[flatidx_mul:%.*]] = mul i32 [[tid]], 4
  ; FCHECK: [[flatidx:%.*]] = add i32 0, [[flatidx_mul]]
  ; FCHECK: [[gep:%.*]] = getelementptr inbounds nuw [32 x i32], ptr addrspace(3) [[arrayofVecData]], i32 0, i32 [[flatidx]]
  ; CHECK: [[x:%.*]] = load <4 x i32>, ptr addrspace(3) [[gep]], align 4
  ; CHECK: ret <4 x i32> [[x]]
  %i = tail call i32 @llvm.dx.thread.id(i32 0)
  %gep = getelementptr inbounds nuw <4 x i32>, ptr addrspace(3) @"arrayofVecData", i32 %i
  %x = load <4 x i32>, ptr addrspace(3) %gep, align 4
  ret <4 x i32> %x
}

; CHECK-LABEL: subtype_scalar_test
define <4 x i32> @subtype_scalar_test() {
  ; CHECK: [[tid:%.*]] = tail call i32 @llvm.dx.thread.id(i32 0)
  ; SCHECK: [[gep:%.*]] = getelementptr inbounds nuw [8 x [4 x i32]], ptr addrspace(3) [[arrayofVecData]], i32 0, i32 0, i32 [[tid]]
  ; FCHECK: [[flatidx_mul:%.*]] = mul i32 [[tid]], 1
  ; FCHECK: [[flatidx:%.*]] = add i32 0, [[flatidx_mul]]
  ; FCHECK: [[gep:%.*]] = getelementptr inbounds nuw [32 x i32], ptr addrspace(3) [[arrayofVecData]], i32 0, i32 [[flatidx]]
  ; CHECK: [[x:%.*]] = load <4 x i32>, ptr addrspace(3) [[gep]], align 4
  ; CHECK: ret <4 x i32> [[x]]
  %i = tail call i32 @llvm.dx.thread.id(i32 0)
  %gep = getelementptr inbounds nuw i32, ptr addrspace(3) @"arrayofVecData", i32 %i
  %x = load <4 x i32>, ptr addrspace(3) %gep, align 4
  ret <4 x i32> %x
}

; CHECK-LABEL: subtype_i8_test
define <4 x i32> @subtype_i8_test() {
  ; CHECK: [[tid:%.*]] = tail call i32 @llvm.dx.thread.id(i32 0)
  ; SCHECK: [[gep:%.*]] = getelementptr inbounds nuw i8, ptr addrspace(3) [[arrayofVecData]], i32 [[tid]]
  ; FCHECK: [[flatidx_mul:%.*]] = mul i32 [[tid]], 1
  ; FCHECK: [[flatidx_lshr:%.*]] = lshr i32 [[flatidx_mul]], 2
  ; FCHECK: [[flatidx:%.*]] = add i32 0, [[flatidx_lshr]]
  ; FCHECK: [[gep:%.*]] = getelementptr inbounds nuw [32 x i32], ptr addrspace(3) [[arrayofVecData]], i32 0, i32 [[flatidx]]
  ; CHECK: [[x:%.*]] = load <4 x i32>, ptr addrspace(3) [[gep]], align 4
  ; CHECK: ret <4 x i32> [[x]]
  %i = tail call i32 @llvm.dx.thread.id(i32 0)
  %gep = getelementptr inbounds nuw i8, ptr addrspace(3) @"arrayofVecData", i32 %i
  %x = load <4 x i32>, ptr addrspace(3) %gep, align 4
  ret <4 x i32> %x
}
