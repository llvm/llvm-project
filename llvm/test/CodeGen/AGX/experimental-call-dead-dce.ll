; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -verify-machineinstrs < %s | FileCheck %s
;
; The unused wrapper initially keeps dead_leaf alive while the AGX call
; selector runs. Once the wrapper disappears, GlobalDCE must remove the leaf
; before its large body reaches the per-function backend.
; CHECK-LABEL: kernel:
; CHECK-NOT: dead_leaf:

define agx_kernel void @kernel() {
entry:
  store volatile i32 7, ptr addrspace(1) inttoptr (i64 4096 to ptr addrspace(1))
  ret void
}

define internal i32 @dead_leaf(i32 %x) {
entry:
  %y = add i32 %x, 1
  ret i32 %y
}

define internal i32 @unused_wrapper(i32 %x) {
entry:
  %y = call i32 @dead_leaf(i32 %x)
  ret i32 %y
}
