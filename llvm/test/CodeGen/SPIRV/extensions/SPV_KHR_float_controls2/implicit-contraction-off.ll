; Implicit ContractionOff under SPV_KHR_float_controls2 is FPFastMathDefault
; with zero flags per float type; it needs FloatControls2.

; RUN: split-file %s %t
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %t/fadd.ll -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls2 %t/fadd.ll -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %t/fadd.ll -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %t/fadd.ll -o - -filetype=obj | spirv-val %}

; =off forces all kernels, but not declarations.
; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa --spirv-fp-contract=off %t/fadd.ll -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa --spirv-fp-contract=off %t/fadd.ll -o - -filetype=obj | spirv-val %}

; =off overrides FP_CONTRACT metadata.
; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa --spirv-fp-contract=off %t/fp-contract-md.ll -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa --spirv-fp-contract=off %t/fp-contract-md.ll -o - -filetype=obj | spirv-val %}

; No float type: nothing to emit, nothing required.
; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %t/no-float.ll -o - | FileCheck %s --check-prefix=NOFLOAT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %t/no-float.ll -o - -filetype=obj | spirv-val %}

; No implicit mode: explicit spirv.ExecutionMode, or a shader.
; RUN: sed 's/FNPTR/ptr/' %t/exec-mode-md.ll | llc -O0 -mtriple=spirv64-unknown-unknown - -o - | FileCheck %s --check-prefix=NOMODE
; RUN: %if spirv-tools %{ sed 's/FNPTR/ptr/' %t/exec-mode-md.ll | llc -O0 -mtriple=spirv64-unknown-unknown - -o - -filetype=obj | spirv-val %}
; RUN: sed 's/FNPTR/ptr addrspace(4)/' %t/exec-mode-md.ll | llc -O0 -mtriple=spirv64-amd-amdhsa - -o - | FileCheck %s --check-prefix=NOMODE
; RUN: %if spirv-tools %{ sed 's/FNPTR/ptr addrspace(4)/' %t/exec-mode-md.ll | llc -O0 -mtriple=spirv64-amd-amdhsa - -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %t/shader.ll -o - | FileCheck %s --check-prefix=NOMODE
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %t/shader.ll -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_float_controls2 %t/shader.ll -o - | FileCheck %s --check-prefix=NOMODE
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_float_controls2 %t/shader.ll -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: OpCapability FloatControls2
; CHECK: OpExtension "SPV_KHR_float_controls2"
; CHECK-COUNT-3: OpExecutionModeId %[[#K:]] FPFastMathDefault %[[#]] %[[#ZERO:]]
; CHECK-NOT: OpExecutionMode
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ZERO]] = OpConstantNull %[[#INT]]
; CHECK-DAG: OpTypeFloat 16
; CHECK-DAG: OpTypeFloat 32
; CHECK-DAG: OpTypeFloat 64
; CHECK: %[[#K]] = OpFunction

; NOFLOAT-NOT: FloatControls2
; NOFLOAT-NOT: OpExecutionModeId

; NOMODE-NOT: FloatControls2
; NOMODE-NOT: OpExecutionMode
; NOMODE: OpExecutionMode %[[#]] LocalSize 1 1 1
; NOMODE-NOT: OpExecutionMode

;--- fadd.ll
define spir_kernel void @k(ptr addrspace(1) %h, ptr addrspace(1) %f, ptr addrspace(1) %d) {
  %hv = load half, ptr addrspace(1) %h
  %ha = fadd half %hv, %hv
  store half %ha, ptr addrspace(1) %h
  %fv = load float, ptr addrspace(1) %f
  %fa = fadd float %fv, %fv
  store float %fa, ptr addrspace(1) %f
  %dv = load double, ptr addrspace(1) %d
  %da = fadd double %dv, %dv
  store double %da, ptr addrspace(1) %d
  ret void
}

declare spir_kernel void @decl()

;--- fp-contract-md.ll
define spir_kernel void @k(ptr addrspace(1) %h, ptr addrspace(1) %f, ptr addrspace(1) %d) {
  %hv = load half, ptr addrspace(1) %h
  store half %hv, ptr addrspace(1) %h
  %fv = load float, ptr addrspace(1) %f
  store float %fv, ptr addrspace(1) %f
  %dv = load double, ptr addrspace(1) %d
  store double %dv, ptr addrspace(1) %d
  ret void
}

!opencl.enable.FP_CONTRACT = !{}

;--- no-float.ll
define spir_kernel void @k(ptr addrspace(1) %p) {
  %v = load i32, ptr addrspace(1) %p
  store i32 %v, ptr addrspace(1) %p
  ret void
}

;--- exec-mode-md.ll
define spir_kernel void @k(ptr addrspace(1) %p) {
  %v = load float, ptr addrspace(1) %p
  %a = fadd float %v, %v
  store float %a, ptr addrspace(1) %p
  ret void
}

!spirv.ExecutionMode = !{!0}
!0 = !{FNPTR @k, i32 17, i32 1, i32 1, i32 1}

;--- shader.ll
@g = internal addrspace(10) global float 0.0

define void @main() #0 {
  %v = load float, ptr addrspace(10) @g
  %a = fadd float %v, %v
  store float %a, ptr addrspace(10) @g
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="1,1,1" }
