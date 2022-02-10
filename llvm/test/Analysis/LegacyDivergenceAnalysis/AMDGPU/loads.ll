; RUN: opt -mtriple=amdgcn-- -amdgpu-use-legacy-divergence-analysis -passes='print<divergence>' 2>&1 -disable-output %s | FileCheck %s

; Test that we consider loads from flat and private addrspaces to be divergent.

; CHECK: DIVERGENT: %val = load i32, i32* %flat, align 4
define amdgpu_kernel void @flat_load(i32* %flat) {
  %val = load i32, i32* %flat, align 4
  ret void
}

; CHECK: DIVERGENT: %val = load i32, i32 addrspace(5)* %priv, align 4
define amdgpu_kernel void @private_load(i32 addrspace(5)* %priv) {
  %val = load i32, i32 addrspace(5)* %priv, align 4
  ret void
}
