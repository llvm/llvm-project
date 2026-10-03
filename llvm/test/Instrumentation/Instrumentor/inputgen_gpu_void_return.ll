; Verify generated entries call functions with void return type.
; RUN: opt < %s -passes=inputgen-gpu -inputgen-gpu-entry-function=vvv_foo -S | FileCheck %s

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define hidden void @vvv_foo(ptr noundef %a) {
entry:
  store i32 1, ptr %a, align 4
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @__ig_entry_vvv_foo(
; CHECK: call void @vvv_foo(
; CHECK-NEXT: ret void
