; RUN: llc < %s -mcpu=generic -mtriple=i686-linux -segmented-stacks -verify-machineinstrs | FileCheck %s -check-prefix=X32-Linux
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux  -segmented-stacks -verify-machineinstrs | FileCheck %s -check-prefix=X64-Linux
; RUN: llc < %s -mcpu=generic -mtriple=i686-darwin -segmented-stacks -verify-machineinstrs | FileCheck %s -check-prefix=X32-Darwin
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-darwin -segmented-stacks -verify-machineinstrs | FileCheck %s -check-prefix=X64-Darwin
; RUN: llc < %s -mcpu=generic -mtriple=i686-mingw32 -segmented-stacks -verify-machineinstrs | FileCheck %s -check-prefix=X32-MinGW
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-freebsd -segmented-stacks -verify-machineinstrs | FileCheck %s -check-prefix=X64-FreeBSD

; We used to crash with filetype=obj
; RUN: llc < %s -mcpu=generic -mtriple=i686-linux -segmented-stacks -filetype=obj
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux -segmented-stacks -filetype=obj
; RUN: llc < %s -mcpu=generic -mtriple=i686-darwin -segmented-stacks -filetype=obj
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-darwin -segmented-stacks -filetype=obj
; RUN: llc < %s -mcpu=generic -mtriple=i686-mingw32 -segmented-stacks -filetype=obj
; RUN: llc < %s -mcpu=generic -mtriple=x86_64-freebsd -segmented-stacks -filetype=obj

; RUN: not llc < %s -mcpu=generic -mtriple=x86_64-solaris -segmented-stacks 2> %t.log
; RUN: FileCheck %s -input-file=%t.log -check-prefix=X64-Solaris
; RUN: not llc < %s -mcpu=generic -mtriple=x86_64-mingw32 -segmented-stacks 2> %t.log
; RUN: FileCheck %s -input-file=%t.log -check-prefix=X64-MinGW
; RUN: not llc < %s -mcpu=generic -mtriple=i686-freebsd -segmented-stacks 2> %t.log
; RUN: FileCheck %s -input-file=%t.log -check-prefix=X32-FreeBSD

; X64-Solaris: Segmented stacks not supported on this platform
; X64-MinGW: Segmented stacks not supported on this platform
; X32-FreeBSD: Segmented stacks not supported on FreeBSD i386

; Just to prevent the alloca from being optimized away
declare void @dummy_use(i32*, i32)

define void @test_basic() {
        %mem = alloca i32, i32 10
        call void @dummy_use (i32* %mem, i32 10)
	ret void

; X32-Linux:       test_basic:

; X32-Linux:       cmpl %gs:48, %esp
; X32-Linux-NEXT:  ja      .LBB0_2

; X32-Linux:       pushl $0
; X32-Linux-NEXT:  pushl $60
; X32-Linux-NEXT:  calll __morestack
; X32-Linux-NEXT:  ret

; X64-Linux:       test_basic:

; X64-Linux:       cmpq %fs:112, %rsp
; X64-Linux-NEXT:  ja      .LBB0_2

; X64-Linux:       movabsq $40, %r10
; X64-Linux-NEXT:  movabsq $0, %r11
; X64-Linux-NEXT:  callq __morestack
; X64-Linux-NEXT:  ret

; X32-Darwin:      test_basic:

; X32-Darwin:      movl $432, %ecx
; X32-Darwin-NEXT: cmpl %gs:(%ecx), %esp
; X32-Darwin-NEXT: ja      LBB0_2

; X32-Darwin:      pushl $0
; X32-Darwin-NEXT: pushl $60
; X32-Darwin-NEXT: calll ___morestack
; X32-Darwin-NEXT: ret

; X64-Darwin:      test_basic:

; X64-Darwin:      cmpq %gs:816, %rsp
; X64-Darwin-NEXT: ja      LBB0_2

; X64-Darwin:      movabsq $40, %r10
; X64-Darwin-NEXT: movabsq $0, %r11
; X64-Darwin-NEXT: callq ___morestack
; X64-Darwin-NEXT: ret

; X32-MinGW:       test_basic:

; X32-MinGW:       cmpl %fs:20, %esp
; X32-MinGW-NEXT:  ja      LBB0_2

; X32-MinGW:       pushl $0
; X32-MinGW-NEXT:  pushl $48
; X32-MinGW-NEXT:  calll ___morestack
; X32-MinGW-NEXT:  ret

; X64-FreeBSD:       test_basic:

; X64-FreeBSD:       cmpq %fs:24, %rsp
; X64-FreeBSD-NEXT:  ja      .LBB0_2

; X64-FreeBSD:       movabsq $40, %r10
; X64-FreeBSD-NEXT:  movabsq $0, %r11
; X64-FreeBSD-NEXT:  callq __morestack
; X64-FreeBSD-NEXT:  ret

}

define i32 @test_nested(i32 * nest %closure, i32 %other) {
       %addend = load i32 * %closure
       %result = add i32 %other, %addend
       ret i32 %result

; X32-Linux:       cmpl %gs:48, %esp
; X32-Linux-NEXT:  ja      .LBB1_2

; X32-Linux:       pushl $4
; X32-Linux-NEXT:  pushl $0
; X32-Linux-NEXT:  calll __morestack
; X32-Linux-NEXT:  ret

; X64-Linux:       cmpq %fs:112, %rsp
; X64-Linux-NEXT:  ja      .LBB1_2

; X64-Linux:       movq %r10, %rax
; X64-Linux-NEXT:  movabsq $0, %r10
; X64-Linux-NEXT:  movabsq $0, %r11
; X64-Linux-NEXT:  callq __morestack
; X64-Linux-NEXT:  ret
; X64-Linux-NEXT:  movq %rax, %r10

; X32-Darwin:      movl $432, %edx
; X32-Darwin-NEXT: cmpl %gs:(%edx), %esp
; X32-Darwin-NEXT: ja      LBB1_2

; X32-Darwin:      pushl $4
; X32-Darwin-NEXT: pushl $0
; X32-Darwin-NEXT: calll ___morestack
; X32-Darwin-NEXT: ret

; X64-Darwin:      cmpq %gs:816, %rsp
; X64-Darwin-NEXT: ja      LBB1_2

; X64-Darwin:      movq %r10, %rax
; X64-Darwin-NEXT: movabsq $0, %r10
; X64-Darwin-NEXT: movabsq $0, %r11
; X64-Darwin-NEXT: callq ___morestack
; X64-Darwin-NEXT: ret
; X64-Darwin-NEXT: movq %rax, %r10

; X32-MinGW:       cmpl %fs:20, %esp
; X32-MinGW-NEXT:  ja      LBB1_2

; X32-MinGW:       pushl $4
; X32-MinGW-NEXT:  pushl $0
; X32-MinGW-NEXT:  calll ___morestack
; X32-MinGW-NEXT:  ret

; X64-FreeBSD:       cmpq %fs:24, %rsp
; X64-FreeBSD-NEXT:  ja      .LBB1_2

; X64-FreeBSD:       movq %r10, %rax
; X64-FreeBSD-NEXT:  movabsq $0, %r10
; X64-FreeBSD-NEXT:  movabsq $0, %r11
; X64-FreeBSD-NEXT:  callq __morestack
; X64-FreeBSD-NEXT:  ret
; X64-FreeBSD-NEXT:  movq %rax, %r10

}

define void @test_large() {
        %mem = alloca i32, i32 10000
        call void @dummy_use (i32* %mem, i32 0)
        ret void

; X32-Linux:       leal -40012(%esp), %ecx
; X32-Linux-NEXT:  cmpl %gs:48, %ecx
; X32-Linux-NEXT:  ja      .LBB2_2

; X32-Linux:       pushl $0
; X32-Linux-NEXT:  pushl $40012
; X32-Linux-NEXT:  calll __morestack
; X32-Linux-NEXT:  ret

; X64-Linux:       leaq -40008(%rsp), %r11
; X64-Linux-NEXT:  cmpq %fs:112, %r11
; X64-Linux-NEXT:  ja      .LBB2_2

; X64-Linux:       movabsq $40008, %r10
; X64-Linux-NEXT:  movabsq $0, %r11
; X64-Linux-NEXT:  callq __morestack
; X64-Linux-NEXT:  ret

; X32-Darwin:      leal -40012(%esp), %ecx
; X32-Darwin-NEXT: movl $432, %eax
; X32-Darwin-NEXT: cmpl %gs:(%eax), %ecx
; X32-Darwin-NEXT: ja      LBB2_2

; X32-Darwin:      pushl $0
; X32-Darwin-NEXT: pushl $40012
; X32-Darwin-NEXT: calll ___morestack
; X32-Darwin-NEXT: ret

; X64-Darwin:      leaq -40008(%rsp), %r11
; X64-Darwin-NEXT: cmpq %gs:816, %r11
; X64-Darwin-NEXT: ja      LBB2_2

; X64-Darwin:      movabsq $40008, %r10
; X64-Darwin-NEXT: movabsq $0, %r11
; X64-Darwin-NEXT: callq ___morestack
; X64-Darwin-NEXT: ret

; X32-MinGW:       leal -40008(%esp), %ecx
; X32-MinGW-NEXT:  cmpl %fs:20, %ecx
; X32-MinGW-NEXT:  ja      LBB2_2

; X32-MinGW:       pushl $0
; X32-MinGW-NEXT:  pushl $40008
; X32-MinGW-NEXT:  calll ___morestack
; X32-MinGW-NEXT:  ret

; X64-FreeBSD:       leaq -40008(%rsp), %r11
; X64-FreeBSD-NEXT:  cmpq %fs:24, %r11
; X64-FreeBSD-NEXT:  ja      .LBB2_2

; X64-FreeBSD:       movabsq $40008, %r10
; X64-FreeBSD-NEXT:  movabsq $0, %r11
; X64-FreeBSD-NEXT:  callq __morestack
; X64-FreeBSD-NEXT:  ret

}

define fastcc void @test_fastcc() {
        %mem = alloca i32, i32 10
        call void @dummy_use (i32* %mem, i32 10)
        ret void

; X32-Linux:       test_fastcc:

; X32-Linux:       cmpl %gs:48, %esp
; X32-Linux-NEXT:  ja      .LBB3_2

; X32-Linux:       pushl $0
; X32-Linux-NEXT:  pushl $60
; X32-Linux-NEXT:  calll __morestack
; X32-Linux-NEXT:  ret

; X64-Linux:       test_fastcc:

; X64-Linux:       cmpq %fs:112, %rsp
; X64-Linux-NEXT:  ja      .LBB3_2

; X64-Linux:       movabsq $40, %r10
; X64-Linux-NEXT:  movabsq $0, %r11
; X64-Linux-NEXT:  callq __morestack
; X64-Linux-NEXT:  ret

; X32-Darwin:      test_fastcc:

; X32-Darwin:      movl $432, %eax
; X32-Darwin-NEXT: cmpl %gs:(%eax), %esp
; X32-Darwin-NEXT: ja      LBB3_2

; X32-Darwin:      pushl $0
; X32-Darwin-NEXT: pushl $60
; X32-Darwin-NEXT: calll ___morestack
; X32-Darwin-NEXT: ret

; X64-Darwin:      test_fastcc:

; X64-Darwin:      cmpq %gs:816, %rsp
; X64-Darwin-NEXT: ja      LBB3_2

; X64-Darwin:      movabsq $40, %r10
; X64-Darwin-NEXT: movabsq $0, %r11
; X64-Darwin-NEXT: callq ___morestack
; X64-Darwin-NEXT: ret

; X32-MinGW:       test_fastcc:

; X32-MinGW:       cmpl %fs:20, %esp
; X32-MinGW-NEXT:  ja      LBB3_2

; X32-MinGW:       pushl $0
; X32-MinGW-NEXT:  pushl $48
; X32-MinGW-NEXT:  calll ___morestack
; X32-MinGW-NEXT:  ret

; X64-FreeBSD:       test_fastcc:

; X64-FreeBSD:       cmpq %fs:24, %rsp
; X64-FreeBSD-NEXT:  ja      .LBB3_2

; X64-FreeBSD:       movabsq $40, %r10
; X64-FreeBSD-NEXT:  movabsq $0, %r11
; X64-FreeBSD-NEXT:  callq __morestack
; X64-FreeBSD-NEXT:  ret

}

define fastcc void @test_fastcc_large() {
        %mem = alloca i32, i32 10000
        call void @dummy_use (i32* %mem, i32 0)
        ret void

; X32-Linux:       test_fastcc_large:

; X32-Linux:       leal -40012(%esp), %eax
; X32-Linux-NEXT:  cmpl %gs:48, %eax
; X32-Linux-NEXT:  ja      .LBB4_2

; X32-Linux:       pushl $0
; X32-Linux-NEXT:  pushl $40012
; X32-Linux-NEXT:  calll __morestack
; X32-Linux-NEXT:  ret

; X64-Linux:       test_fastcc_large:

; X64-Linux:       leaq -40008(%rsp), %r11
; X64-Linux-NEXT:  cmpq %fs:112, %r11
; X64-Linux-NEXT:  ja      .LBB4_2

; X64-Linux:       movabsq $40008, %r10
; X64-Linux-NEXT:  movabsq $0, %r11
; X64-Linux-NEXT:  callq __morestack
; X64-Linux-NEXT:  ret

; X32-Darwin:      test_fastcc_large:

; X32-Darwin:      leal -40012(%esp), %eax
; X32-Darwin-NEXT: movl $432, %ecx
; X32-Darwin-NEXT: cmpl %gs:(%ecx), %eax
; X32-Darwin-NEXT: ja      LBB4_2

; X32-Darwin:      pushl $0
; X32-Darwin-NEXT: pushl $40012
; X32-Darwin-NEXT: calll ___morestack
; X32-Darwin-NEXT: ret

; X64-Darwin:      test_fastcc_large:

; X64-Darwin:      leaq -40008(%rsp), %r11
; X64-Darwin-NEXT: cmpq %gs:816, %r11
; X64-Darwin-NEXT: ja      LBB4_2

; X64-Darwin:      movabsq $40008, %r10
; X64-Darwin-NEXT: movabsq $0, %r11
; X64-Darwin-NEXT: callq ___morestack
; X64-Darwin-NEXT: ret

; X32-MinGW:       test_fastcc_large:

; X32-MinGW:       leal -40008(%esp), %eax
; X32-MinGW-NEXT:  cmpl %fs:20, %eax
; X32-MinGW-NEXT:  ja      LBB4_2

; X32-MinGW:       pushl $0
; X32-MinGW-NEXT:  pushl $40008
; X32-MinGW-NEXT:  calll ___morestack
; X32-MinGW-NEXT:  ret

; X64-FreeBSD:       test_fastcc_large:

; X64-FreeBSD:       leaq -40008(%rsp), %r11
; X64-FreeBSD-NEXT:  cmpq %fs:24, %r11
; X64-FreeBSD-NEXT:  ja      .LBB4_2

; X64-FreeBSD:       movabsq $40008, %r10
; X64-FreeBSD-NEXT:  movabsq $0, %r11
; X64-FreeBSD-NEXT:  callq __morestack
; X64-FreeBSD-NEXT:  ret

}

define fastcc void @test_fastcc_large_with_ecx_arg(i32 %a) {
        %mem = alloca i32, i32 10000
        call void @dummy_use (i32* %mem, i32 %a)
        ret void

; This is testing that the Mac implementation preserves ecx

; X32-Darwin:      test_fastcc_large_with_ecx_arg:

; X32-Darwin:      leal -40012(%esp), %eax
; X32-Darwin-NEXT: pushl %ecx
; X32-Darwin-NEXT: movl $432, %ecx
; X32-Darwin-NEXT: cmpl %gs:(%ecx), %eax
; X32-Darwin-NEXT: popl %ecx
; X32-Darwin-NEXT: ja      LBB5_2

; X32-Darwin:      pushl $0
; X32-Darwin-NEXT: pushl $40012
; X32-Darwin-NEXT: calll ___morestack
; X32-Darwin-NEXT: ret

}
