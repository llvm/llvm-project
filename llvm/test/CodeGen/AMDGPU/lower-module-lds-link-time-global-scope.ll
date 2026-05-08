; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-lower-module-lds -amdgpu-enable-object-linking < %s | FileCheck %s

; Comprehensive test for global-scope (external linkage) LDS in link-time mode.
; External LDS variables remain as standalone external declarations -- they are
; NOT wrapped into per-function structs.
;
; Scenarios covered:
;   - Device function and kernel each using distinct global-scope LDS
;   - Multiple kernels sharing a device function that uses LDS
;   - A single function using multiple LDS variables
;   - Transitive call chain where only a leaf uses LDS
;   - Kernel directly using LDS (no device-function LDS user)

; -- Global variables --
@lds_shared   = addrspace(3) global [64 x i32] poison, align 16
@lds_kernel_a = addrspace(3) global [32 x float] poison, align 4
@lds_kernel_b = addrspace(3) global [16 x i64] poison, align 8
@lds_leaf     = addrspace(3) global [8 x i32] poison, align 4
@lds_direct   = addrspace(3) global [32 x float] poison, align 4

declare void @extern_func()

; All external-linkage LDS become external declarations.
; CHECK-DAG: @lds_shared = external addrspace(3) global [64 x i32]
; CHECK-DAG: @lds_kernel_a = external addrspace(3) global [32 x float]
; CHECK-DAG: @lds_kernel_b = external addrspace(3) global [16 x i64]
; CHECK-DAG: @lds_leaf = external addrspace(3) global [8 x i32]
; CHECK-DAG: @lds_direct = external addrspace(3) global [32 x float]

; No per-function structs should be created for any function.
; CHECK-NOT: @__amdgpu_lds.shared_func
; CHECK-NOT: @__amdgpu_lds.kernel_a
; CHECK-NOT: @__amdgpu_lds.kernel_b
; CHECK-NOT: @__amdgpu_lds.leaf_func
; CHECK-NOT: @__amdgpu_lds.mid_func
; CHECK-NOT: @__amdgpu_lds.direct_kernel

; --- shared_func: uses lds_shared, called by both kernel_a and kernel_b ---
; CHECK-LABEL: define void @shared_func()
; CHECK: getelementptr [64 x i32], ptr addrspace(3) @lds_shared
; CHECK: call void @extern_func()

; --- kernel_a: uses its own LDS + calls shared_func ---
; CHECK-LABEL: define amdgpu_kernel void @kernel_a()
; CHECK: getelementptr [32 x float], ptr addrspace(3) @lds_kernel_a
; CHECK: call void @shared_func()

; --- kernel_b: uses its own LDS + calls shared_func ---
; CHECK-LABEL: define amdgpu_kernel void @kernel_b()
; CHECK: getelementptr [16 x i64], ptr addrspace(3) @lds_kernel_b
; CHECK: call void @shared_func()

; --- leaf_func: uses lds_leaf (transitive -- called via mid_func) ---
; CHECK-LABEL: define void @leaf_func()
; CHECK: getelementptr [8 x i32], ptr addrspace(3) @lds_leaf

; --- mid_func: no LDS, just calls leaf_func + extern ---
; CHECK-LABEL: define void @mid_func()
; CHECK-NOT: @__amdgpu_lds
; CHECK: call void @leaf_func()
; CHECK: call void @extern_func()

; --- transitive_kernel: calls mid_func (transitive LDS user) ---
; CHECK-LABEL: define amdgpu_kernel void @transitive_kernel()
; CHECK: call void @mid_func()

; --- direct_kernel: kernel directly uses LDS, no device function uses LDS ---
; CHECK-LABEL: define amdgpu_kernel void @direct_kernel()
; CHECK: getelementptr [32 x float], ptr addrspace(3) @lds_direct

; Metadata: one entry per (function, variable) pair for direct users only.
; CHECK: !amdgpu.lds.uses = !{{{![0-9]+, ![0-9]+, ![0-9]+, ![0-9]+, ![0-9]+}}}
; CHECK-DAG: !{ptr @shared_func, ptr addrspace(3) @lds_shared}
; CHECK-DAG: !{ptr @kernel_a, ptr addrspace(3) @lds_kernel_a}
; CHECK-DAG: !{ptr @kernel_b, ptr addrspace(3) @lds_kernel_b}
; CHECK-DAG: !{ptr @leaf_func, ptr addrspace(3) @lds_leaf}
; CHECK-DAG: !{ptr @direct_kernel, ptr addrspace(3) @lds_direct}

define void @shared_func() {
  %gep = getelementptr [64 x i32], ptr addrspace(3) @lds_shared, i32 0, i32 0
  store i32 1, ptr addrspace(3) %gep
  call void @extern_func()
  ret void
}

define amdgpu_kernel void @kernel_a() {
  %gep = getelementptr [32 x float], ptr addrspace(3) @lds_kernel_a, i32 0, i32 0
  store float 1.0, ptr addrspace(3) %gep
  call void @shared_func()
  ret void
}

define amdgpu_kernel void @kernel_b() {
  %gep = getelementptr [16 x i64], ptr addrspace(3) @lds_kernel_b, i32 0, i32 0
  store i64 1, ptr addrspace(3) %gep
  call void @shared_func()
  ret void
}

define void @leaf_func() {
  %gep = getelementptr [8 x i32], ptr addrspace(3) @lds_leaf, i32 0, i32 0
  store i32 42, ptr addrspace(3) %gep
  ret void
}

define void @mid_func() {
  call void @leaf_func()
  call void @extern_func()
  ret void
}

define amdgpu_kernel void @transitive_kernel() {
  call void @mid_func()
  ret void
}

define amdgpu_kernel void @direct_kernel() {
  %gep = getelementptr [32 x float], ptr addrspace(3) @lds_direct, i32 0, i32 0
  store float 1.0, ptr addrspace(3) %gep
  call void @extern_func()
  ret void
}
