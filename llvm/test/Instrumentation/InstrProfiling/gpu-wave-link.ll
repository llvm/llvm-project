; RUN: split-file %s %t
; RUN: opt -passes='pgo-instr-gen,instrprof,always-inline,globaldce,verify' -offload-pgo-sampling=0 %t/a.ll -o %t/a.bc
; RUN: opt -passes='pgo-instr-gen,instrprof,always-inline,globaldce,verify' -offload-pgo-sampling=0 %t/b.ll -o %t/b.bc
; RUN: llvm-link -S %t/a.bc %t/b.bc | FileCheck %s --check-prefixes=CHECK,AB
; RUN: llvm-link -S %t/b.bc %t/a.bc | FileCheck %s --check-prefixes=CHECK,BA

; Model an inline device function shared by two translation units. Both callers
; must update the same lane/wave allocation after COMDAT selection, in either
; link order. No collection option is needed to obtain the wave slots.
; CHECK: @[[SHARED:__profc_shared[^ ]*]] = {{.*}}[2 x i64] zeroinitializer

; AB-LABEL: define amdgpu_kernel void @caller_a(
; AB: call void @__llvm_profile_instrument_gpu({{.*}}i64 1, ptr {{.*}}@[[SHARED]]{{.*}}i32 1
; AB-LABEL: define amdgpu_kernel void @caller_b(
; AB: call void @__llvm_profile_instrument_gpu({{.*}}i64 1, ptr {{.*}}@[[SHARED]]{{.*}}i32 1
; BA-LABEL: define amdgpu_kernel void @caller_b(
; BA: call void @__llvm_profile_instrument_gpu({{.*}}i64 1, ptr {{.*}}@[[SHARED]]{{.*}}i32 1
; BA-LABEL: define amdgpu_kernel void @caller_a(
; BA: call void @__llvm_profile_instrument_gpu({{.*}}i64 1, ptr {{.*}}@[[SHARED]]{{.*}}i32 1

;--- a.ll
source_filename = "a.cpp"
target triple = "amdgcn-amd-amdhsa"
$shared = comdat any

define amdgpu_kernel void @caller_a(ptr addrspace(1) %out, i32 %x) {
  %r = call i32 @shared(i32 %x)
  store i32 %r, ptr addrspace(1) %out
  ret void
}

define linkonce_odr i32 @shared(i32 %x) alwaysinline comdat {
  %r = add i32 %x, 1
  ret i32 %r
}

;--- b.ll
source_filename = "b.cpp"
target triple = "amdgcn-amd-amdhsa"
$shared = comdat any

define amdgpu_kernel void @caller_b(ptr addrspace(1) %out, i32 %x) {
  %r = call i32 @shared(i32 %x)
  store i32 %r, ptr addrspace(1) %out
  ret void
}

define linkonce_odr i32 @shared(i32 %x) alwaysinline comdat {
  %r = add i32 %x, 1
  ret i32 %r
}
