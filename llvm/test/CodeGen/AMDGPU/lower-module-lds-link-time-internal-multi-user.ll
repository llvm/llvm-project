; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-lower-module-lds -amdgpu-enable-object-linking < %s | FileCheck %s

source_filename = "source_a.hip"

; Internal-linkage LDS variables used by multiple kernels must be packed into a
; per-module struct with a module-unique name, rather than promoted individually
; to external linkage, to avoid cross-TU name collisions.

@a = internal addrspace(3) global [32 x i32] poison, align 16
@b = internal addrspace(3) global [16 x float] poison, align 4

; Per-module struct containing both internal multi-user variables.
; CHECK: @[[INTERN:__amdgpu_lds.__internal\.[a-f0-9]+]] = external {{(dso_local )?}}addrspace(3) global %[[INTERN]].t, align 16

; Original internal-linkage variables should be removed.
; CHECK-NOT: @a =
; CHECK-NOT: @b =

; Both kernels reference the struct.
; CHECK-LABEL: define amdgpu_kernel void @kernel1()
; CHECK: @[[INTERN]]
; CHECK: @[[INTERN]]

; CHECK-LABEL: define amdgpu_kernel void @kernel2()
; CHECK: @[[INTERN]]
; CHECK: @[[INTERN]]

; Metadata: struct entries for both kernels.
; CHECK: !amdgpu.lds.uses = !{{{![0-9]+, ![0-9]+}}}
; CHECK-DAG: !{ptr @kernel1, ptr addrspace(3) @[[INTERN]]}
; CHECK-DAG: !{ptr @kernel2, ptr addrspace(3) @[[INTERN]]}

define amdgpu_kernel void @kernel1() {
  %gep_a = getelementptr [32 x i32], ptr addrspace(3) @a, i32 0, i32 0
  store i32 1, ptr addrspace(3) %gep_a
  %gep_b = getelementptr [16 x float], ptr addrspace(3) @b, i32 0, i32 0
  store float 2.0, ptr addrspace(3) %gep_b
  ret void
}

define amdgpu_kernel void @kernel2() {
  %gep_a = getelementptr [32 x i32], ptr addrspace(3) @a, i32 0, i32 0
  store i32 3, ptr addrspace(3) %gep_a
  %gep_b = getelementptr [16 x float], ptr addrspace(3) @b, i32 0, i32 0
  store float 4.0, ptr addrspace(3) %gep_b
  ret void
}
