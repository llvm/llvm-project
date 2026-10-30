; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls -agx-measured-call-frames < %s | FileCheck %s
; Measured call frames (a module of one kernel): the kernel's region is its
; frame, each call depth's is its largest function frame (here depth 1's is
; 6416 bytes, over the fixed 4096), recorded as the functions are compiled in
; increasing depth; the kernel reserves all 32768 addressable bytes:
; 32768 / 4 + 10 = 8202 units, rounded up to 8204.

; CHECK-LABEL: kernel:
; CHECK: stack_adjust 8204
; CHECK: call outer
; CHECK-LABEL: outer:
; CHECK: stack_store 0, i32, x, {{r[0-9]+}}, 6400
; CHECK: call inner
; CHECK-LABEL: inner:
; CHECK: stack_store 0, i32, x, {{r[0-9]+}}, 6416

define agx_kernel void @kernel(ptr addrspace(1) %out, i32 %v) {
  %r = call i32 @outer(i32 %v)
  store i32 %r, ptr addrspace(1) %out, align 4
  ret void
}

; Depth 1, a frame of 6412 bytes (optnone keeps the array whole), after the
; kernel's empty region.
define internal i32 @outer(i32 %v) noinline optnone {
  %big = alloca [1600 x i32], align 4, addrspace(5)
  %p = getelementptr [1600 x i32], ptr addrspace(5) %big, i32 0, i32 1599
  store volatile i32 %v, ptr addrspace(5) %p, align 4
  %r = call i32 @inner(i32 %v)
  %q = load volatile i32, ptr addrspace(5) %p, align 4
  %s = add i32 %r, %q
  ret i32 %s
}

; Depth 2, after depth 1's 6416 bytes.
define internal i32 @inner(i32 %v) noinline {
  %small = alloca [4 x i32], align 4, addrspace(5)
  %p = getelementptr [4 x i32], ptr addrspace(5) %small, i32 0, i32 1
  store volatile i32 %v, ptr addrspace(5) %p, align 4
  %r = load volatile i32, ptr addrspace(5) %p, align 4
  ret i32 %r
}
