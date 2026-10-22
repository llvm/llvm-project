; RUN: llc -mtriple=agx -mcpu=agx-g13g -o /dev/null -print-after=agx-always-inline < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=agx -mcpu=agx-g14c -agx-recursion-stack-size=256 < %s | FileCheck %s --check-prefix=SMALL

; Recursive functions become, for each function of the cycle called from
; outside it, a state machine over a stack of frames in private memory, which
; is inlined like any other function.

; CHECK: @__agx_stack_overflow = addrspace(1) global i32 0, align 4
; CHECK-NOT: define {{.*}} @fib(
; CHECK-LABEL: define agx_kernel void @kernel(
; CHECK: call i32 @fib.stack(
; CHECK: call i1 @even.stack(
; CHECK-LABEL: define internal i32 @fib.stack(i32 %0)
; CHECK: %stack = alloca [4096 x i8], align 16, addrspace(5)
; CHECK: %sp = alloca i32, align 4, addrspace(5)
; The first frame: no call site to return to; the argument after the header.
; CHECK: store i32 0, ptr addrspace(5) %sp
; CHECK: store i32 %0, ptr addrspace(5)
; A call: past the end of the stack, set the flag and end; else the frame of
; the callee follows the caller's (32 bytes), with the call site's number and
; the caller's stack pointer.
; CHECK: %n1 = sub i32
; CHECK-NEXT: [[OLD:%.*]] = load i32, ptr addrspace(5) %sp
; CHECK-NEXT: [[NEW:%.*]] = add i32 [[OLD]], 32
; CHECK-NEXT: [[END:%.*]] = add i32 [[NEW]], 32
; CHECK-NEXT: icmp ugt i32 [[END]], 4096
; CHECK: store volatile i32 1, ptr addrspace(1) @__agx_stack_overflow
; CHECK-NEXT: call void @llvm.trap()
; CHECK: store i32 1, ptr addrspace(5)
; CHECK: store i32 [[OLD]], ptr addrspace(5)
; CHECK: store i32 [[NEW]], ptr addrspace(5) %sp
; CHECK: store i32 2, ptr addrspace(5)
; A return switches on the call site: 0 leaves the function.
; CHECK: exit:
; CHECK-NEXT: [[R:%.*]] = load i32, ptr addrspace(5) %result
; CHECK-NEXT: ret i32 [[R]]
; CHECK: fib.return:
; CHECK-NEXT: %site = phi i32
; CHECK-NEXT: switch i32 %site, label %exit [
; CHECK-NEXT: i32 1, label
; CHECK-NEXT: i32 2, label
; Mutual recursion: one function for the cycle of even and odd, called from
; outside through even only.
; CHECK-LABEL: define internal i1 @even.stack(i32 %0)
; CHECK: odd.return:
; CHECK: even.return:
; CHECK-NOT: define {{.*}} @odd

; ASM-LABEL: kernel:
; ASM: stack size: {{[0-9]+}} bytes
; ASM-NOT: call
; ASM: stack_store
; ASM: stop

; The stack of frames is the private memory of the threads that run it.
; SMALL: stack size: {{2[0-9][0-9]|3[0-9][0-9]}} bytes
; SMALL: .globl __agx_stack_overflow

define i32 @fib(i32 %n) {
  %small = icmp ult i32 %n, 2
  br i1 %small, label %base, label %rec
base:
  ret i32 %n
rec:
  %n1 = sub i32 %n, 1
  %a = call i32 @fib(i32 %n1)
  %n2 = sub i32 %n, 2
  %b = call i32 @fib(i32 %n2)
  %r = add i32 %a, %b
  ret i32 %r
}

define i1 @even(i32 %n) {
  %z = icmp eq i32 %n, 0
  br i1 %z, label %yes, label %rec
yes:
  ret i1 true
rec:
  %m = sub i32 %n, 1
  %o = call i1 @odd(i32 %m)
  %r = xor i1 %o, false
  ret i1 %r
}

define i1 @odd(i32 %n) {
  %z = icmp eq i32 %n, 0
  br i1 %z, label %no, label %rec
no:
  ret i1 false
rec:
  %m = sub i32 %n, 1
  %e = call i1 @even(i32 %m)
  ret i1 %e
}

define agx_kernel void @kernel(ptr addrspace(1) %out, i32 %x) {
  %f = call i32 @fib(i32 %x)
  store i32 %f, ptr addrspace(1) %out
  %e = call i1 @even(i32 %x)
  %ez = zext i1 %e to i32
  %p = getelementptr i32, ptr addrspace(1) %out, i32 1
  store i32 %ez, ptr addrspace(1) %p
  ret void
}
