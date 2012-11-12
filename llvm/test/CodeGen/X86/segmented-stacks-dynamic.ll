; RUN: llc < %s -mcpu=generic -mtriple=i686-linux -segmented-stacks -verify-machineinstrs | FileCheck %s -check-prefix=X32
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux  -segmented-stacks -verify-machineinstrs | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mcpu=generic -mtriple=i686-linux -segmented-stacks -filetype=obj
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux -segmented-stacks -filetype=obj

; Just to prevent the alloca from being optimized away
declare void @dummy_use(i32*, i32)

define i32 @test_basic(i32 %l) {
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

; X32:      test_basic:

; X32:      cmpl %gs:48, %esp
; X32-NEXT: ja      .LBB0_2

; X32:      pushl $4
; X32-NEXT: pushl $12
; X32-NEXT: calll __morestack
; X32-NEXT: ret

; X32:      movl %esp, %eax
; X32-NEXT: subl %ecx, %eax
; X32-NEXT: cmpl %eax, %gs:48

; X32:      movl %eax, %esp

; X32:      subl $12, %esp
; X32-NEXT: pushl %ecx
; X32-NEXT: calll __morestack_allocate_stack_space
; X32-NEXT: addl $16, %esp

; X64:      test_basic:

; X64:      cmpq %fs:112, %rsp
; X64-NEXT: ja      .LBB0_2

; X64:      movabsq $24, %r10
; X64-NEXT: movabsq $0, %r11
; X64-NEXT: callq __morestack
; X64-NEXT: ret

; X64:      movq %rsp, %[[RDI:rdi|rax]]
; X64-NEXT: subq %{{.*}}, %[[RDI]]
; X64-NEXT: cmpq %[[RDI]], %fs:112

; X64:      movq %[[RDI]], %rsp

; X64:      movq %{{.*}}, %rdi
; X64-NEXT: callq __morestack_allocate_stack_space
; X64:      movq %rax, %rdi

}
