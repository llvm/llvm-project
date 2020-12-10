; RUN: llc < %s -mcpu=generic -mtriple=i686-linux -verify-machineinstrs | FileCheck %s -check-prefix=X86
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux -verify-machineinstrs | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-gnux32 -verify-machineinstrs | FileCheck %s -check-prefix=X32ABI
; RUN: llc < %s -mcpu=generic -mtriple=i686-linux -filetype=obj
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux -filetype=obj
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-gnux32 -filetype=obj

; Just to prevent the alloca from being optimized away
declare void @dummy_use(i32*, i32)

define i32 @test_basic(i32 %l) #0 {
        %mem = alloca i32, i32 %l
        call void @dummy_use (i32* %mem, i32 %l)
        %terminate = icmp eq i32 %l, 0
        br i1 %terminate, label %true, label %false

true:
        ret i32 0

false:
        %newlen = sub i32 %l, 1
        %retvalue = call i32 @test_basic(i32 %newlen)
        ret i32 %retvalue

; X86-LABEL:      test_basic:

; X86:      cmpl %gs:48, %esp
; X86-NEXT: jbe	.LBB0_1

; X86:      movl %esp, %eax
; X86:      subl %ecx, %eax
; X86-NEXT: cmpl %eax, %gs:48

; X86:      movl %eax, %esp

; X86:      subl $12, %esp
; X86-NEXT: pushl %ecx
; X86-NEXT: calll __morestack_allocate_stack_space
; X86-NEXT: addl $16, %esp

; X86:      pushl $4
; X86-NEXT: pushl $12
; X86-NEXT: calll __morestack
; X86-NEXT: ret

; X64-LABEL:      test_basic:

; X64:      cmpq %fs:112, %rsp
; X64-NEXT: jbe      .LBB0_1

; X64:      movq %rsp, %[[RDI:rdi|rax]]
; X64:      subq %{{.*}}, %[[RDI]]
; X64-NEXT: cmpq %[[RDI]], %fs:112

; X64:      movq %[[RDI]], %rsp

; X64:      movq %{{.*}}, %rdi
; X64-NEXT: callq __morestack_allocate_stack_space
; X64:      movq %rax, %rdi

; X64:      movabsq $24, %r10
; X64-NEXT: movabsq $0, %r11
; X64-NEXT: callq __morestack
; X64-NEXT: ret

; X32ABI-LABEL:      test_basic:

; X32ABI:      cmpl %fs:64, %esp
; X32ABI-NEXT: jbe      .LBB0_1

; X32ABI:      movl %esp, %[[EDI:edi|eax]]
; X32ABI:      subl %{{.*}}, %[[EDI]]
; X32ABI-NEXT: cmpl %[[EDI]], %fs:64

; X32ABI:      movl %[[EDI]], %esp

; X32ABI:      movl %{{.*}}, %edi
; X32ABI-NEXT: callq __morestack_allocate_stack_space
; X32ABI:      movl %eax, %edi

; X32ABI:      movl $24, %r10d
; X32ABI-NEXT: movl $0, %r11d
; X32ABI-NEXT: callq __morestack
; X32ABI-NEXT: ret

}

attributes #0 = { "split-stack" }
