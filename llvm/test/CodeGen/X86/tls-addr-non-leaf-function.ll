; RUN: llc < %s -relocation-model=pic -O2 -frame-pointer=all -o - | FileCheck %s
; RUN: llc < %s -relocation-model=pic -O2 -o - | FileCheck %s

; This test runs twice with different options regarding the frame pointer:
; first the elimination is disabled, then it is enabled. The disabled case is
; the "control group".
; The function 'foo' below is marked with the "frame-pointer"="non-leaf"
; attribute which dictates that the frame pointer should not be eliminated
; unless the function is a leaf (i.e. it doesn't call any other function).
; Now, 'foo' is not a leaf function, because it performs a TLS access which on
; X86 ELF in PIC mode is expanded as a library call.
; This call is represented with a pseudo-instruction which doesn't appear to be
; a call when inspected by the analysis passes (it doesn't have the "isCall"
; flag), and the ISel lowering code creating the pseudo was not informing the 
; MachineFrameInfo that the function contained calls. This affected the decision
; whether to eliminate the frame pointer.
; With the fix, the "hasCalls" flag is set in the MFI for the function whenever
; a TLS access pseudo-instruction is created, so 'foo' appears to be a non-leaf
; function, and the difference in the options does not affect codegen: both
; versions will have a frame pointer.

; Test that there's some frame pointer usage in 'foo'...
; CHECK: foo:
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; ... and the TLS library call is also present.
; CHECK: leaq x@TLSGD(%rip), %rdi
; CHECK: callq __tls_get_addr@PLT

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = thread_local global i32 0
define i32 @foo() "frame-pointer"="non-leaf" {
  %a = load i32, ptr @x, align 4
  ret i32 %a
}
