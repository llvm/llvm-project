; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-lower-module-lds -amdgpu-enable-object-linking < %s | FileCheck %s

; Test the three-way classification of LDS variables:
;   1. Global-scope (external linkage): standalone external declaration
;   2. Kernel-scope (internal linkage, single kernel user): wrapped in per-kernel struct
;   3. Callee-scope (internal linkage, single callee user): wrapped in per-callee struct
;
; Also tests that a global-scope variable used by multiple functions produces
; one metadata entry per (function, variable) pair.

; Global-scope: external linkage, used by both func and my_kernel.
@lds_global = addrspace(3) global [64 x i32] poison, align 16

; Callee-scope: internal linkage, used only by func.
@lds_func_priv = internal addrspace(3) global [32 x float] poison, align 4

; Kernel-scope: internal linkage, used only by my_kernel.
@lds_kernel_priv = internal addrspace(3) global [16 x i64] poison, align 8

declare void @extern_func()

; Global-scope: remains as external declaration.
; CHECK-DAG: @lds_global = external addrspace(3) global [64 x i32]

; Callee-scope: wrapped into per-function struct.
; CHECK-DAG: @__amdgpu_lds.func = external {{(dso_local )?}}addrspace(3) global %__amdgpu_lds.func.t

; Kernel-scope: wrapped into per-kernel struct.
; CHECK-DAG: @__amdgpu_lds.my_kernel = external {{(dso_local )?}}addrspace(3) global %__amdgpu_lds.my_kernel.t

; Original internal-linkage variables should be removed.
; CHECK-NOT: @lds_func_priv
; CHECK-NOT: @lds_kernel_priv

; func: uses lds_global directly, uses lds_func_priv via struct GEP.
; CHECK-LABEL: define void @func()
; CHECK: getelementptr {{.*}} ptr addrspace(3) @lds_global
; CHECK: getelementptr {{.*}} ptr addrspace(3) @__amdgpu_lds.func

; my_kernel: uses lds_global directly, uses lds_kernel_priv via struct GEP.
; CHECK-LABEL: define amdgpu_kernel void @my_kernel()
; CHECK: getelementptr {{.*}} ptr addrspace(3) @lds_global
; CHECK: getelementptr {{.*}} ptr addrspace(3) @__amdgpu_lds.my_kernel

; Metadata:
; CHECK: !amdgpu.lds.uses = !{{{![0-9]+, ![0-9]+, ![0-9]+, ![0-9]+}}}
;   Function-scope entries (one per struct).
; CHECK-DAG: !{ptr @my_kernel, ptr addrspace(3) @__amdgpu_lds.my_kernel}
; CHECK-DAG: !{ptr @func, ptr addrspace(3) @__amdgpu_lds.func}
;   Global-scope entries (one per using function).
; CHECK-DAG: !{ptr @my_kernel, ptr addrspace(3) @lds_global}
; CHECK-DAG: !{ptr @func, ptr addrspace(3) @lds_global}

; Module should be marked with the link-time LDS module flag.
; CHECK: !{i32 1, !"amdgpu-link-time-lds", i32 1}

define void @func() {
  %gep1 = getelementptr [64 x i32], ptr addrspace(3) @lds_global, i32 0, i32 0
  store i32 1, ptr addrspace(3) %gep1
  %gep2 = getelementptr [32 x float], ptr addrspace(3) @lds_func_priv, i32 0, i32 0
  store float 2.0, ptr addrspace(3) %gep2
  call void @extern_func()
  ret void
}

define amdgpu_kernel void @my_kernel() {
  %gep1 = getelementptr [64 x i32], ptr addrspace(3) @lds_global, i32 0, i32 0
  store i32 3, ptr addrspace(3) %gep1
  %gep2 = getelementptr [16 x i64], ptr addrspace(3) @lds_kernel_priv, i32 0, i32 0
  store i64 4, ptr addrspace(3) %gep2
  call void @func()
  ret void
}
