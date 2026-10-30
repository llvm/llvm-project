; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-experimental-leaf-calls < %s | FileCheck %s
; Called functions are emitted after the kernels, whatever the module's order
; (AIR lists functions first): a kernel's program runs from its symbol to the
; end of .text.

; CHECK: kernel:
; CHECK: call callee
; CHECK: callee:

define internal i32 @callee(i32 %x) noinline {
  %y = mul i32 %x, %x
  ret i32 %y
}

define agx_kernel void @kernel(ptr addrspace(1) %out, i32 %x) {
  %y = call i32 @callee(i32 %x)
  store i32 %y, ptr addrspace(1) %out, align 4
  ret void
}
