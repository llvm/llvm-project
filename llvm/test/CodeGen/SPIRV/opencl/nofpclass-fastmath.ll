; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64v1.5-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.5-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %s -o - -filetype=obj | spirv-val %}

; Check that nofpclass attributes on OpenCL builtin calls are translated to
; FPFastMathMode decorations on the corresponding OpExtInst instructions.

; CHECK-DAG: OpDecorate %[[#FMAX_RES:]] FPFastMathMode NotNaN|NotInf
; CHECK-DAG: OpDecorate %[[#FMIN_RES:]] FPFastMathMode NotInf
; CHECK-DAG: OpDecorate %[[#LDEXP_RES:]] FPFastMathMode NotNaN|NotInf

; CHECK: %[[#FMAX_RES]] = OpExtInst %[[#]] %[[#]] fmax
; CHECK: %[[#FMIN_RES]] = OpExtInst %[[#]] %[[#]] fmin
; CHECK: %[[#LDEXP_RES]] = OpExtInst %[[#]] %[[#]] ldexp

; nofpclass(nan inf) on return and all float params -> NotNaN|NotInf
declare spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf), float noundef nofpclass(nan inf))

; nofpclass(inf) on first param only, nofpclass(nan inf) on second -> NotInf
declare spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fminff(float noundef nofpclass(inf), float noundef nofpclass(nan inf))

; nofpclass(nan inf) on return and float param, i32 param skipped -> NotNaN|NotInf
declare spir_func noundef nofpclass(nan inf) float @_Z17__spirv_ocl_ldexpfi(float noundef nofpclass(nan inf), i32 noundef)

define spir_kernel void @test(ptr addrspace(1) %data, ptr addrspace(1) %a, ptr addrspace(1) %b) {
entry:
  %0 = load float, ptr addrspace(1) %a, align 4
  %1 = load float, ptr addrspace(1) %b, align 4
  %fmax = call spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf) %0, float noundef nofpclass(nan inf) %1)
  store float %fmax, ptr addrspace(1) %data, align 4
  %fmin = call spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fminff(float noundef nofpclass(inf) %0, float noundef nofpclass(nan inf) %1)
  store float %fmin, ptr addrspace(1) %data, align 4
  %2 = load i32, ptr addrspace(1) %b, align 4
  %ldexp = call spir_func noundef nofpclass(nan inf) float @_Z17__spirv_ocl_ldexpfi(float noundef nofpclass(nan inf) %0, i32 noundef %2)
  store float %ldexp, ptr addrspace(1) %data, align 4
  ret void
}
