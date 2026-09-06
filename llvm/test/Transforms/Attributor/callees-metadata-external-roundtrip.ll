; RUN: opt -passes=attributor -S %s | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

; The host may read @h and later pass or store that pointer for @k. Semantic
; !callees is the authority for this external round trip; the absence of an
; in-module use path from @f to %p cannot erase the call and its volatile side
; effect.
@h = protected constant ptr @f
@sink = protected global i32 0
@llvm.used = appending global [1 x ptr] [ptr @h], section "llvm.metadata"

define hidden void @f() noinline {
  store volatile i32 1, ptr @sink
  ret void
}

define amdgpu_kernel void @k(ptr %p) {
; CHECK-LABEL: define amdgpu_kernel void @k(
; CHECK: call void @f()
  call void %p(), !callees !0
  ret void
}

!0 = !{ptr @f}
