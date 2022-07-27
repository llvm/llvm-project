; RUN: llc < %s -mcpu=skx -mtriple x86_64-unknown-linux-gnu -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mcpu=skx -mtriple=x86_64-linux-gnux32 -verify-machineinstrs | FileCheck %s --check-prefix=X32

define i32 @A() {
; CHECK: movq %rsp, %rdi
; CHECK-NEXT: call

; X32: movl %esp, %edi
; X32-NEXT: call
  %alloc = alloca i32, align 8
  %call = call i32 @foo(ptr %alloc)
  ret i32 %call
}

declare i32 @foo(ptr)
