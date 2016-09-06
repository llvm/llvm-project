; RUN: llc -verify-machineinstrs -mtriple=x86_64-pc-linux < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=x86_64-pc-linux-gnux32 < %s | FileCheck -check-prefix=X32ABI %s
; RUN: llc -verify-machineinstrs -mtriple=x86_64-pc-nacl < %s | FileCheck -check-prefix=NACL %s

; x32 uses %esp, %ebp as stack and frame pointers

; CHECK-LABEL: foo
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; CHECK: movq %rdi, -8(%rbp)
; CHECK: popq %rbp
; X32ABI-LABEL: foo
; X32ABI: pushq %rbp
; X32ABI: movl %esp, %ebp
; X32ABI: movl %edi, -4(%ebp)
; X32ABI: popq %rbp
; NACL-LABEL: foo
; NACL: pushq %rbp
; NACL: movq %rsp, %rbp
; NACL: movl %edi, -4(%rbp)
; NACL: popq %rbp


define void @foo(i32* %a) #0 {
entry:
  %a.addr = alloca i32*, align 4
  %b = alloca i32*, align 4
  store i32* %a, i32** %a.addr, align 4
  ret void
}

attributes #0 = { nounwind uwtable "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"}


