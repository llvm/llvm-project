; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls < %s | FileCheck %s
; The only kernel of a module may have a frame over 4096 bytes when it calls:
; its region is its frame (6412 bytes, here at optnone so the array stays
; whole) rounded to 16, and the called function's depth-1 region follows it.
; stack_adjust covers both: (6416 + 4096) / 4 + 10 = 2638 units, rounded up to
; 2640.

; CHECK-LABEL: kernel:
; CHECK: stack_adjust 2640
; CHECK: stack_store 0, i32, x, {{r[0-9]+}}, 6408
; CHECK: call callee
; CHECK-LABEL: callee:
; CHECK: stack_store 0, i32, x, {{r[0-9]+}}, 6416
define agx_kernel void @kernel(ptr addrspace(1) %out, i32 %i, i32 %v) optnone noinline {
  %big = alloca [1600 x i32], align 4, addrspace(5)
  %p = getelementptr [1600 x i32], ptr addrspace(5) %big, i32 0, i32 1599
  store volatile i32 %v, ptr addrspace(5) %p, align 4
  %r = call i32 @callee(i32 %i, i32 %v)
  %q = load volatile i32, ptr addrspace(5) %p, align 4
  %s = add i32 %r, %q
  store i32 %s, ptr addrspace(1) %out, align 4
  ret void
}

define internal i32 @callee(i32 %i, i32 %v) noinline {
  %small = alloca [4 x i32], align 4, addrspace(5)
  %p = getelementptr [4 x i32], ptr addrspace(5) %small, i32 0, i32 1
  store volatile i32 %v, ptr addrspace(5) %p, align 4
  %r = load volatile i32, ptr addrspace(5) %p, align 4
  ret i32 %r
}
