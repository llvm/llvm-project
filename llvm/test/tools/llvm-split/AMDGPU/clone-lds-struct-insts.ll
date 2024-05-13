; RUN: opt -passes=amdgpu-clone-module-lds %s -S | FileCheck %s

; RUN: opt -passes=amdgpu-clone-module-lds %s -S -o %t
; RUN: llvm-split -o %t %t -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=MOD0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=MOD1 %s

target triple = "amdgcn-amd-amdhsa"

%struct.RT = type { i8, [10 x [20 x i32]], i8 }
%struct.GV = type { i32, double, %struct.RT }

; CHECK: [[GV_CLONE_0:@.*]] = internal addrspace(3) global %struct.GV poison, align 8
; CHECK: [[GV:@.*]] = internal addrspace(3) global %struct.GV poison, align 8
@lds_gv = internal addrspace(3) global %struct.GV poison, align 8

define protected amdgpu_kernel void @kernel1(i32 %n) #3 {
; CHECK-LABEL: define protected amdgpu_kernel void @kernel1(
; CHECK-SAME: i32 [[N:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @lds_func(i32 [[N]], i1 false)
; CHECK-NEXT:    [[CALL_CLONE_0:%.*]] = call i32 @lds_func.clone.0(i32 [[N]], i1 false)
; CHECK-NEXT:    ret void
;
entry:
  %call = call i32 @lds_func(i32 %n, i1 false)
  ret void
}

define protected amdgpu_kernel void @kernel2(i32 %n) #3 {
; CHECK-LABEL: define protected amdgpu_kernel void @kernel2(
; CHECK-SAME: i32 [[N:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @lds_func(i32 [[N]], i1 true)
; CHECK-NEXT:    [[CALL_CLONE_0:%.*]] = call i32 @lds_func.clone.0(i32 [[N]], i1 true)
; CHECK-NEXT:    ret void
;
entry:
  %call = call i32 @lds_func(i32 %n, i1 1)
  ret void
}

define i32 @lds_func(i32 %x, i1 %cond) {
; CHECK-LABEL: define i32 @lds_func(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[TMP_0:%.*]] = alloca %struct.GV, align 8, addrspace(3)
; CHECK-NEXT:   %p = getelementptr inbounds [[STRUCT_GV:%.*]], ptr addrspace(3) [[GV]], i64 1, i32 2, i32 1, i64 5, i64 13
; CHECK-NEXT:   store i32 %x, ptr addrspace(3) %p, align 4
; CHECK-NEXT:   store i32 %x, ptr addrspace(3) getelementptr inbounds (%struct.GV, ptr addrspace(3) [[GV]], i64 1, i32 2, i32 1, i64 5, i64 12), align 4
; CHECK-NEXT:   store ptr addrspace(3) getelementptr inbounds (%struct.GV, ptr addrspace(3) [[GV]], i64 1, i32 2, i32 1, i64 5, i64 11), ptr addrspace(3) getelementptr inbounds (%struct.GV, ptr addrspace(3) [[GV]], i64 1, i32 2, i32 1, i64 5, i64 1), align 4
; CHECK-NEXT:   %gep.ascast = load i8, ptr getelementptr inbounds (%struct.GV, ptr addrspacecast (ptr addrspace(3) [[GV]] to ptr), i64 6), align 1
; CHECK-NEXT:   br i1 %cond, label %bb.1, label %bb.2
; CHECK:      bb.1:                                             ; preds = %entry
; CHECK-NEXT:   br label %sink
; CHECK:      bb.2:                                             ; preds = %entry
; CHECK-NEXT:   br label %sink
; CHECK:      sink:                                             ; preds = %bb.2, %bb.1
; CHECK-NEXT:   %val = phi ptr addrspace(3) [ [[TMP_0]], %bb.1 ], [ [[GV]], %bb.2 ]
; CHECK-NEXT:   %p.0 = getelementptr inbounds %struct.GV, ptr addrspace(3) [[GV]], i64 1, i32 2, i32 1, i64 5, i64 1
; CHECK-NEXT:   %retval = load i32, ptr addrspace(3) %p.0, align 4
; CHECK-NEXT:   ret i32 %retval
;
entry:
  %tmp.GV = alloca %struct.GV, addrspace(3)
  %p = getelementptr inbounds %struct.GV, ptr addrspace(3) @lds_gv, i64 1, i32 2, i32 1, i64 5, i64 13
  store i32 %x, ptr addrspace(3) %p
  store i32 %x, ptr addrspace(3) getelementptr inbounds (%struct.GV, ptr addrspace(3) @lds_gv, i64 1, i32 2, i32 1, i64 5, i64 12)
  store ptr addrspace(3) getelementptr inbounds (%struct.GV, ptr addrspace(3) @lds_gv, i64 1, i32 2, i32 1, i64 5, i64 11), ptr addrspace(3) getelementptr inbounds (%struct.GV, ptr addrspace(3) @lds_gv, i64 1, i32 2, i32 1, i64 5, i64 1)
  %gep.ascast = load i8, ptr getelementptr inbounds (%struct.GV, ptr addrspacecast (ptr addrspace(3) @lds_gv to ptr), i64 6), align 1
  br i1 %cond, label %bb.1, label %bb.2

bb.1:
  br label %sink

bb.2:
  br label %sink

sink:
  %val = phi ptr addrspace(3) [%tmp.GV, %bb.1], [@lds_gv, %bb.2]
  %p.0 = getelementptr inbounds %struct.GV, ptr addrspace(3) @lds_gv, i64 1, i32 2, i32 1, i64 5, i64 1
  %retval = load i32, ptr addrspace(3) %p.0
  ret i32 %retval
}

; CHECK-LABEL: define i32 @lds_func.clone.0(
; CHECK-SAME: i32 [[X:%.*]]) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[TMP_0]] = alloca %struct.GV, align 8, addrspace(3)
; CHECK-NEXT:   %p = getelementptr inbounds [[STRUCT_GV:%.*]], ptr addrspace(3) [[GV_CLONE_0]], i64 1, i32 2, i32 1, i64 5, i64 13
; CHECK-NEXT:   store i32 %x, ptr addrspace(3) %p, align 4
; CHECK-NEXT:   store i32 %x, ptr addrspace(3) getelementptr inbounds (%struct.GV, ptr addrspace(3) [[GV_CLONE_0]], i64 1, i32 2, i32 1, i64 5, i64 12), align 4
; CHECK-NEXT:   store ptr addrspace(3) getelementptr inbounds (%struct.GV, ptr addrspace(3) [[GV_CLONE_0]], i64 1, i32 2, i32 1, i64 5, i64 11), ptr addrspace(3) getelementptr inbounds (%struct.GV, ptr addrspace(3) [[GV_CLONE_0]], i64 1, i32 2, i32 1, i64 5, i64 1), align 4
; CHECK-NEXT:   %gep.ascast = load i8, ptr getelementptr inbounds (%struct.GV, ptr addrspacecast (ptr addrspace(3) [[GV_CLONE_0]] to ptr), i64 6), align 1
; CHECK-NEXT:   br i1 %cond, label %bb.1, label %bb.2
; CHECK:      bb.1:                                             ; preds = %entry
; CHECK-NEXT:   br label %sink
; CHECK:      bb.2:                                             ; preds = %entry
; CHECK-NEXT:   br label %sink
; CHECK:      sink:                                             ; preds = %bb.2, %bb.1
; CHECK-NEXT:   %val = phi ptr addrspace(3) [ [[TMP_0]], %bb.1 ], [ [[GV_CLONE_0]], %bb.2 ]
; CHECK-NEXT:   %p.0 = getelementptr inbounds %struct.GV, ptr addrspace(3) [[GV_CLONE_0]], i64 1, i32 2, i32 1, i64 5, i64 1
; CHECK-NEXT:   %retval = load i32, ptr addrspace(3) %p.0, align 4
; CHECK-NEXT:   ret i32 %retval

; MOD0: {{.*}} addrspace(3) global %struct.GV, align 8
; MOD0: {{.*}} addrspace(3) global %struct.GV poison, align 8

; MOD1: {{.*}} addrspace(3) global %struct.GV poison, align 8
; MOD1: {{.*}} addrspace(3) global %struct.GV, align 8
; MOD1: define protected amdgpu_kernel void @kernel1(i32 %n)
; MOD1: define protected amdgpu_kernel void @kernel2(i32 %n)
; MOD1: define i32 @lds_func(i32 %x, i1 %cond)
; MOD1: define i32 @lds_func.clone.0(i32 %x, i1 %cond)
