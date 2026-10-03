; Verify entry generation initializes state and calls the selected function.
; RUN: opt < %s -passes=inputgen-gpu -inputgen-gpu-entry-function=vvv_foo -S | FileCheck %s

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; CHECK: @inputgen_buffer = external addrspace(1) global ptr
; CHECK: @inputgen_buffer_size = external addrspace(1) global i64
; CHECK: @inputgen_buffer_offset = external addrspace(1) global i64
; CHECK: @inputgen_mode = external addrspace(1) global i32

define hidden i32 @vvv_foo(ptr noundef %a) {
; CHECK-LABEL: define hidden i32 @vvv_foo(
; CHECK: [[LOAD:%.*]] = load i32, ptr {{%.*}}, align 4
; CHECK: [[EXT:%.*]] = zext i32 [[LOAD]] to i64
; CHECK: [[CALL:%.*]] = call i64 @__ig_post_load(i64 [[EXT]], i64 4, i32 12, i32 -1)
; CHECK: [[TRUNC:%.*]] = trunc i64 [[CALL]] to i32
; CHECK: ret i32 [[TRUNC]]
entry:
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

; CHECK-LABEL: define amdgpu_kernel void @__ig_entry_vvv_foo(
; CHECK-SAME: i32 [[MODE:%.*]], ptr [[BUFFER:%.*]], i64 [[SIZE:%.*]], ptr [[RESULT:%.*]]) {
; CHECK: store i32 [[MODE]], ptr addrspace(1) @inputgen_mode, align 4
; CHECK: store ptr [[BUFFER]], ptr addrspace(1) @inputgen_buffer, align 8
; CHECK: store i64 [[SIZE]], ptr addrspace(1) @inputgen_buffer_size, align 8
; CHECK: store i64 0, ptr addrspace(1) @inputgen_buffer_offset, align 8
; CHECK: [[ARG:%.*]] = alloca i64, align 8, addrspace(5)
; CHECK: [[ARG_CAST:%.*]] = addrspacecast ptr addrspace(5) [[ARG]] to ptr
; CHECK: [[RESULT_VAL:%.*]] = call i32 @vvv_foo(ptr [[ARG_CAST]])
; CHECK: store i32 [[RESULT_VAL]], ptr [[RESULT]], align 4
; CHECK: ret void
