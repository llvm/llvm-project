; RUN: llvm-split -o %t -j 3 -mtriple=amdgpu-amd-amdhsa < %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 --implicit-check-not=define %s

; No entry point reaches @self_rec or the cycle of @mutual_a and @mutual_b, and
; each of these functions has an incoming direct call. Check that each cycle is
; still assigned to a partition. Only @self_rec calls @helper, and @helper is
; defined first. Check that @helper does not become an entry point of its own,
; so it is defined only once.

; CHECK0: define amdgpu_kernel void @kernel

; CHECK1: define void @mutual_a
; CHECK1: define void @mutual_b

; CHECK2: define internal void @helper
; CHECK2: define void @self_rec

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
