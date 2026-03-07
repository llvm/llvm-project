;; Test that AMDGPU PGO instrumentation linearizes 3D block indices for counter
;; slot computation. The linear block index is computed as:
;;   LinearBlockId = blockIdx.x + blockIdx.y * gridDim.x
;;                 + blockIdx.z * gridDim.x * gridDim.y
;; This ensures correct counter slot assignment for kernels launched with 3D grids.

; RUN: opt %s -mtriple=amdgcn-amd-amdhsa -passes=instrprof -S | FileCheck %s

@__hip_cuid_abcdef789 = addrspace(1) global i8 0
@__profn_kernel_3d = private constant [9 x i8] c"kernel_3d"

define amdgpu_kernel void @kernel_3d() {
  call void @llvm.instrprof.increment(ptr @__profn_kernel_3d, i64 12345, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

;; Check that all three workgroup ID intrinsics are called (X, Y, Z)
; CHECK: %BlockIdxX = call i32 @llvm.amdgcn.workgroup.id.x()
; CHECK: %BlockIdxY = call i32 @llvm.amdgcn.workgroup.id.y()
; CHECK: %BlockIdxZ = call i32 @llvm.amdgcn.workgroup.id.z()

;; Check that grid dimensions are loaded from implicit args
; CHECK: %GridDimX = load i32, ptr addrspace(4)
; CHECK: %GridDimY = load i32, ptr addrspace(4)

;; Check linearization: gridDim.x * gridDim.y
; CHECK: %GridDimXY = mul i32 %GridDimX, %GridDimY

;; Check linearization components
; CHECK-DAG: %yTimesGx = mul i32 %BlockIdxY, %GridDimX
; CHECK-DAG: %zTimesGxy = mul i32 %BlockIdxZ, %GridDimXY

;; Check final linear block index
; CHECK: %LinearBlockId = add i32 %BlockIdxX, %yzContrib

;; Check total grid size: gridDim.x * gridDim.y * gridDim.z
; CHECK: %TotalGridSize = mul i32 %GridDimXY,
