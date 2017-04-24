; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5"
target triple = "amdgcn-amd-amdhsa-amdgizcl"

; CHECK_LABEL: atomic_fence
; CHECK: BB#0:
; CHECK: ATOMIC_FENCE 4, 1
; CHECK: s_endpgm

define amdgpu_kernel void @atomic_fence() {
  fence acquire
  ret void
}

