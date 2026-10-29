; RUN: opt -passes='pgo-instr-gen,instrprof,verify' -offload-pgo-sampling=0 -S %s | FileCheck %s

; Wave collection introduces no additional convergent operation.
target triple = "amdgcn-amd-amdhsa"

; CHECK-LABEL: define amdgpu_kernel void @controlled_kernel
; CHECK: call void @__llvm_profile_instrument_gpu(
; CHECK: call token @llvm.experimental.convergence.entry()
; CHECK: ret void
define amdgpu_kernel void @controlled_kernel() convergent {
entry:
  %token = call token @llvm.experimental.convergence.entry()
  ret void
}
