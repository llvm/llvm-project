; RUN: llc -mtriple=amdgcn-amd-amdhsa-amdgizcl -mcpu=kaveri < %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5"

; CHECK-LABEL: atomic_fence
; CHECK:       %bb.0:
; CHECK-NOT:   ATOMIC_FENCE
; CHECK-NEXT:  s_waitcnt vmcnt(0)
; CHECK-NEXT:  buffer_wbinvl1_vol
; CHECK-NEXT:  s_endpgm
define amdgpu_kernel void @atomic_fence() {
  fence acquire
  ret void
}

