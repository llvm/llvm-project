; The test checks if the backend won't crash

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs --spirv-ext=+SPV_INTEL_memory_access_aliasing %s -o - | FileCheck %s

; CHECK: OpControlBarrier
; CHECK-NOT: MemoryAccessAliasingINTEL

define spir_kernel void @barrier_simple()
{
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 272), !noalias !1
  ret void
}

declare dso_local spir_func void @_Z22__spirv_ControlBarrierjjj(i32, i32, i32)

!1 = !{!2}
!2 = distinct !{!2, !3}
!3 = distinct !{!3}
