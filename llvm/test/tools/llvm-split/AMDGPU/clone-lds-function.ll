; RUN: opt -passes=amdgpu-clone-module-lds %s -S | FileCheck %s

; RUN: opt -passes=amdgpu-clone-module-lds %s -S -o %t
; RUN: llvm-split -o %t %t -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=MOD0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=MOD1 %s

target triple = "amdgcn-amd-amdhsa"

; In this examples, CloneModuleLDS pass creates two copies of LDS_GV
; as two kernels call the same device function where LDS_GV is used.

; CHECK: [[LDS_GV_CLONE:@.*\.clone\.0]] = internal unnamed_addr addrspace(3) global [64 x i32] poison, align 16
; CHECK: [[LDS_GV:@.*]] = internal unnamed_addr addrspace(3) global [64 x i32] poison, align 16
@lds_gv = internal unnamed_addr addrspace(3) global [64 x i32] poison, align 16
@lds_gv_alias = alias ptr addrspace(3), ptr addrspace(3) @lds_gv

define protected amdgpu_kernel void @kernel1(i32 %n) #3 {
; CHECK-LABEL: define protected amdgpu_kernel void @kernel1(
; CHECK-SAME: i32 [[N:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @lds_func(i32 [[N]])
; CHECK-NEXT:    [[CALL_CLONE_0:%.*]] = call i32 @lds_func.clone.0(i32 [[N]])
; CHECK-NEXT:    ret void
;
entry:
  %call = call i32 @lds_func(i32 %n)
  ret void
}

define protected amdgpu_kernel void @kernel2(i32 %n) #3 {
; CHECK-LABEL: define protected amdgpu_kernel void @kernel2(
; CHECK-SAME: i32 [[N:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @lds_func(i32 [[N]])
; CHECK-NEXT:    [[CALL_CLONE_0:%.*]] = call i32 @lds_func.clone.0(i32 [[N]])
; CHECK-NEXT:    ret void
;
entry:
  %call = call i32 @lds_func(i32 %n)
  ret void
}


define i32 @lds_func(i32 %x) {
; CHECK-LABEL: define i32 @lds_func(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[P:%.*]] = getelementptr inbounds [64 x i32], ptr addrspacecast (ptr addrspace(3) [[LDS_GV]] to ptr), i64 0, i64 0
; CHECK-NEXT:    store i32 [[X]], ptr [[P]], align 4
; CHECK-NEXT:    ret i32 [[X]]
;
entry:
  %p = getelementptr inbounds [64 x i32], ptr addrspacecast (ptr addrspace(3) @lds_gv to ptr), i64 0, i64 0
  store i32 %x, ptr %p
  ret i32 %x
}

; CHECK-LABEL: define i32 @lds_func.clone.0(i32 %x) {
; CHECK-NEXT: entry:
; CHECK-NEXT:    [[P:%.*]] = getelementptr inbounds [64 x i32], ptr addrspacecast (ptr addrspace(3) [[LDS_GV_CLONE]] to ptr), i64 0, i64 0
; CHECK-NEXT:   store i32 %x, ptr %p, align 4
; CHECK-NEXT:   ret i32 %x

; MOD0: @lds_gv.clone.0 = {{.*}} addrspace(3) global [64 x i32], align 16
; MOD0: @lds_gv = {{.*}} addrspace(3) global [64 x i32] poison, align 16

; MOD1: @lds_gv.clone.0 = {{.*}} addrspace(3) global [64 x i32] poison, align 16
; MOD1: @lds_gv = {{.*}} addrspace(3) global [64 x i32], align 16
; MOD1: define protected amdgpu_kernel void @kernel1(i32 %n)
; MOD1: define protected amdgpu_kernel void @kernel2(i32 %n)
; MOD1: define i32 @lds_func(i32 %x)
; MOD1: define i32 @lds_func.clone.0(i32 %x)
