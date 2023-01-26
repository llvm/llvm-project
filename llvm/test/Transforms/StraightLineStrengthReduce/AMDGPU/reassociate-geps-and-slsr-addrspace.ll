; RUN: opt -S -mtriple=amdgcn-- -passes=separate-const-offset-from-gep,slsr,gvn < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes="separate-const-offset-from-gep,slsr,gvn" < %s | FileCheck %s

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"


; CHECK-LABEL: @slsr_after_reassociate_global_geps_mubuf_max_offset(
; CHECK: [[b1:%[0-9]+]] = getelementptr float, ptr addrspace(1) %arr, i64 [[bump:%[0-9]+]]
; CHECK: [[b2:%[0-9]+]] = getelementptr float, ptr addrspace(1) [[b1]], i64 [[bump]]
define amdgpu_kernel void @slsr_after_reassociate_global_geps_mubuf_max_offset(ptr addrspace(1) %out, ptr addrspace(1) noalias %arr, i32 %i) {
bb:
  %i2 = shl nsw i32 %i, 1
  %j1 = add nsw i32 %i, 1023
  %tmp = sext i32 %j1 to i64
  %p1 = getelementptr inbounds float, ptr addrspace(1) %arr, i64 %tmp
  %v11 = load i32, ptr addrspace(1) %p1, align 4
  store i32 %v11, ptr addrspace(1) %out, align 4

  %j2 = add nsw i32 %i2, 1023
  %tmp5 = sext i32 %j2 to i64
  %p2 = getelementptr inbounds float, ptr addrspace(1) %arr, i64 %tmp5
  %v22 = load i32, ptr addrspace(1) %p2, align 4
  store i32 %v22, ptr addrspace(1) %out, align 4

  ret void
}

; CHECK-LABEL: @slsr_after_reassociate_global_geps_over_mubuf_max_offset(
; CHECK: %j1 = add nsw i32 %i, 1024
; CHECK: %tmp = sext i32 %j1 to i64
; CHECK: getelementptr inbounds float, ptr addrspace(1) %arr, i64 %tmp
; CHECK: getelementptr inbounds float, ptr addrspace(1) %arr, i64 %tmp5
define amdgpu_kernel void @slsr_after_reassociate_global_geps_over_mubuf_max_offset(ptr addrspace(1) %out, ptr addrspace(1) noalias %arr, i32 %i) {
bb:
  %i2 = shl nsw i32 %i, 1
  %j1 = add nsw i32 %i, 1024
  %tmp = sext i32 %j1 to i64
  %p1 = getelementptr inbounds float, ptr addrspace(1) %arr, i64 %tmp
  %v11 = load i32, ptr addrspace(1) %p1, align 4
  store i32 %v11, ptr addrspace(1) %out, align 4

  %j2 = add nsw i32 %i2, 1024
  %tmp5 = sext i32 %j2 to i64
  %p2 = getelementptr inbounds float, ptr addrspace(1) %arr, i64 %tmp5
  %v22 = load i32, ptr addrspace(1) %p2, align 4
  store i32 %v22, ptr addrspace(1) %out, align 4

  ret void
}

; CHECK-LABEL: @slsr_after_reassociate_lds_geps_ds_max_offset(
; CHECK: [[B1:%[0-9]+]] = getelementptr float, ptr addrspace(3) %arr, i32 %i
; CHECK: getelementptr inbounds float, ptr addrspace(3) [[B1]], i32 16383

; CHECK: [[B2:%[0-9]+]] = getelementptr float, ptr addrspace(3) [[B1]], i32 %i
; CHECK: getelementptr inbounds float, ptr addrspace(3) [[B2]], i32 16383
define amdgpu_kernel void @slsr_after_reassociate_lds_geps_ds_max_offset(ptr addrspace(1) %out, ptr addrspace(3) noalias %arr, i32 %i) {
bb:
  %i2 = shl nsw i32 %i, 1
  %j1 = add nsw i32 %i, 16383
  %p1 = getelementptr inbounds float, ptr addrspace(3) %arr, i32 %j1
  %v11 = load i32, ptr addrspace(3) %p1, align 4
  store i32 %v11, ptr addrspace(1) %out, align 4

  %j2 = add nsw i32 %i2, 16383
  %p2 = getelementptr inbounds float, ptr addrspace(3) %arr, i32 %j2
  %v22 = load i32, ptr addrspace(3) %p2, align 4
  store i32 %v22, ptr addrspace(1) %out, align 4

  ret void
}

; CHECK-LABEL: @slsr_after_reassociate_lds_geps_over_ds_max_offset(
; CHECK: %j1 = add nsw i32 %i, 16384
; CHECK: getelementptr inbounds float, ptr addrspace(3) %arr, i32 %j1
; CHECK: %j2 = add i32 %j1, %i
; CHECK: getelementptr inbounds float, ptr addrspace(3) %arr, i32 %j2
define amdgpu_kernel void @slsr_after_reassociate_lds_geps_over_ds_max_offset(ptr addrspace(1) %out, ptr addrspace(3) noalias %arr, i32 %i) {
bb:
  %i2 = shl nsw i32 %i, 1
  %j1 = add nsw i32 %i, 16384
  %p1 = getelementptr inbounds float, ptr addrspace(3) %arr, i32 %j1
  %v11 = load i32, ptr addrspace(3) %p1, align 4
  store i32 %v11, ptr addrspace(1) %out, align 4

  %j2 = add nsw i32 %i2, 16384
  %p2 = getelementptr inbounds float, ptr addrspace(3) %arr, i32 %j2
  %v22 = load i32, ptr addrspace(3) %p2, align 4
  store i32 %v22, ptr addrspace(1) %out, align 4

  ret void
}
