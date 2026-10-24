; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-codegenprepare -debug-pass=Executions %s -o /dev/null 2>&1 | FileCheck %s
;
; A constant shuffle index needs no divergence analysis. Do not schedule
; the legacy whole-function analysis merely to find that there is no work.
; CHECK-NOT: Executing Pass 'Uniformity Analysis'
; CHECK: Executing Pass 'AGX CodeGen Prepare'
; CHECK-NOT: Executing Pass 'Uniformity Analysis'

declare float @llvm.agx.simd.shuffle.f32(float, i32)

define agx_kernel void @constant_index(ptr addrspace(1) %output, float %x) {
  %r = call float @llvm.agx.simd.shuffle.f32(float %x, i32 1)
  store float %r, ptr addrspace(1) %output
  ret void
}
