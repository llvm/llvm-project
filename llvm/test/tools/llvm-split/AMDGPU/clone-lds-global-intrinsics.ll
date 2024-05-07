; RUN: opt -passes=amdgpu-clone-module-lds %s -S | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

%struct.RT = type { i8, [10 x [20 x i32]], i8 }
%struct.GV = type { i32, double, %struct.RT }

; CHECK: [[LDS_GV_CLONE:@.*\.clone\.0]] = internal addrspace(3) global %struct.GV poison, align 8
; CHECK: [[LDS_GV:@.*]] = internal addrspace(3) global %struct.GV poison, align 8
; CHECK: @llvm.used = appending global [2 x ptr addrspace(3)] [ptr addrspace(3) @lds_gv, ptr addrspace(3) @lds_gv], section "llvm.metadata"

@lds_gv = internal addrspace(3) global %struct.GV poison, align 8
@llvm.used = appending global [2 x ptr addrspace(3)] [ptr addrspace(3) @lds_gv, ptr addrspace(3) @lds_gv], section "llvm.metadata"

define protected amdgpu_kernel void @kernel1(i32 %n) #3 {
; CHECK-LABEL: define protected amdgpu_kernel void @kernel1(
; CHECK-SAME: i32 [[N:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @lds_func()
; CHECK-NEXT:    [[CALL_CLONE_0:%.*]] = call i32 @lds_func.clone.0()
; CHECK-NEXT:    ret void
;
entry:
  %call = call i32 @lds_func()
  ret void
}

define protected amdgpu_kernel void @kernel2(i32 %n) #3 {
; CHECK-LABEL: define protected amdgpu_kernel void @kernel2(
; CHECK-SAME: i32 [[N:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @lds_func()
; CHECK-NEXT:    [[CALL_CLONE_0:%.*]] = call i32 @lds_func.clone.0()
; CHECK-NEXT:    ret void
;
entry:
  %call = call i32 @lds_func()
  ret void
}

define ptr @lds_func() {
; CHECK-LABEL: define ptr @lds_func() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[P:%.*]] = getelementptr inbounds [[STRUCT_GV:%.*]], ptr addrspace(3) [[LDS_GV]], i64 1, i32 2, i32 1, i64 5, i64 13
; CHECK-NEXT:    [[RET_PTR:%.*]] = addrspacecast ptr addrspace(3) [[P]] to ptr
; CHECK-NEXT:    ret ptr [[RET_PTR]]
;
entry:
  %p = getelementptr inbounds %struct.GV, ptr addrspace(3) @lds_gv, i64 1, i32 2, i32 1, i64 5, i64 13
  %ret_ptr = addrspacecast ptr addrspace(3) %p to ptr
  ret ptr %ret_ptr
}

; CHECK-LABEL: define ptr @lds_func.clone.0() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[P:%.*]] = getelementptr inbounds %struct.GV, ptr addrspace(3) [[LDS_GV_CLONE]], i64 1, i32 2, i32 1, i64 5, i64 13
; CHECK-NEXT:   [[RET_PTR:%.*]] = addrspacecast ptr addrspace(3) [[P]] to ptr
; CHECK-NEXT:   ret ptr [[RET_PTR]]
; CHECK-NEXT: }
