; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64v1.5-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.5-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check that nofpclass attributes produce FPFastMathMode decorations on
; OpExtInst in Kernel environments even for SPIR-V versions before 1.6,
; because the Kernel capability makes FPFastMathMode available.

; CHECK: OpDecorate %[[#]] FPFastMathMode NotNaN|NotInf

declare spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf), float noundef nofpclass(nan inf))

define spir_kernel void @test(ptr addrspace(1) %data, ptr addrspace(1) %a) {
entry:
  %0 = load float, ptr addrspace(1) %a, align 4
  %fmax = call spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf) %0, float noundef nofpclass(nan inf) %0)
  store float %fmax, ptr addrspace(1) %data, align 4
  ret void
}
