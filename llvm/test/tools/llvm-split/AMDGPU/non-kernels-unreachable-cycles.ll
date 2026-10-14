; REQUIRES: asserts
; RUN: not --crash llvm-split -o %t -j 3 -mtriple=amdgpu-amd-amdhsa < %s 2>&1 | FileCheck --check-prefix=CRASH %s

; No entry point reaches @self_rec or the cycle of @mutual_a and @mutual_b, and
; each of these functions has an incoming direct call. FIXME: These cycles get
; no entry point, so the graph verifier fails. Without assertions, no partition
; defines these functions.

; CRASH: not all nodes are reachable through the graph's entry points!

define internal void @helper(ptr %p) {
  store i32 0, ptr %p
  ret void
}

define void @self_rec(ptr %p) {
  call void @helper(ptr %p)
  call void @self_rec(ptr %p)
  ret void
}

define void @mutual_a(ptr %p) {
  call void @mutual_b(ptr %p)
  ret void
}

define void @mutual_b(ptr %p) {
  call void @mutual_a(ptr %p)
  ret void
}

define amdgpu_kernel void @kernel(ptr %p) {
  store i32 1, ptr %p
  ret void
}
