; RUN: opt -passes='pgo-instr-gen,instrprof,mem2reg,verify' -S %s | FileCheck %s
; RUN: not opt -passes='pgo-instr-gen,instrprof' -pgo-block-coverage -disable-output %s 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not opt -passes='pgo-instr-gen,instrprof' -pgo-temporal-instrumentation -disable-output %s 2>&1 | FileCheck %s --check-prefix=ERROR
; ERROR: wave counts require ordinary counter increments

target triple = "amdgcn-amd-amdhsa"

; Wave counts use the existing instrumentation sites and intrinsics.
; CHECK-LABEL: define void @diamond
; CHECK: call void @__llvm_profile_instrument_gpu(
; CHECK: call void @__llvm_profile_instrument_gpu(
define void @diamond(i1 %cond, ptr %p) {
entry:
  br i1 %cond, label %a, label %b
a:
  store volatile i32 1, ptr %p
  br label %exit
b:
  store volatile i32 2, ptr %p
  br label %exit
exit:
  ret void
}

; Sampling must leave static allocas in the entry so mem2reg can promote them.
; CHECK-LABEL: define amdgpu_kernel void @alloca_kernel
; CHECK-NOT: = alloca
; CHECK: store i32 %value, ptr addrspace(1) %out
; CHECK-NOT: = alloca
; CHECK: ret void
define amdgpu_kernel void @alloca_kernel(ptr addrspace(1) %out, i32 %value) {
entry:
  %slot = alloca i32, align 4, addrspace(5)
  store i32 %value, ptr addrspace(5) %slot
  %loaded = load i32, ptr addrspace(5) %slot
  store i32 %loaded, ptr addrspace(1) %out
  ret void
}
