; RUN: llc -mtriple=x86_64-linux-android < %s | FileCheck -check-prefix=CHECK-X86-64 %s 
; RUN: llc -mtriple=i686-linux-android < %s | FileCheck -check-prefix=CHECK-X86-32 %s 

define i32 @foo() local_unnamed_addr #0 {

  %a = alloca i32, i64 18000, align 16
  %b0 = getelementptr inbounds i32, i32* %a, i64 98
  %b1 = getelementptr inbounds i32, i32* %a, i64 7198
  store volatile i32 1, i32* %b0
  store volatile i32 1, i32* %b1
  %c = load volatile i32, i32* %a
  ret i32 %c
}

attributes #0 =  {"probe-stack"="inline-asm"}

; CHECK-X86-64-LABEL: foo:
; CHECK-X86-64:        # %bb.0:
; CHECK-X86-64-NEXT:	movq	%rsp, %r11
; CHECK-X86-64-NEXT:	subq	$69632, %r11 # imm = 0x11000
; CHECK-X86-64-NEXT:   .LBB0_1:
; CHECK-X86-64-NEXT:	subq	$4096, %rsp # imm = 0x1000
; CHECK-X86-64-NEXT:	movq	$0, (%rsp)
; CHECK-X86-64-NEXT:	cmpq	%r11, %rsp
; CHECK-X86-64-NEXT:	jne	.LBB0_1
; CHECK-X86-64-NEXT:# %bb.2:
; CHECK-X86-64-NEXT:	subq    $2248, %rsp
; CHECK-X86-64-NEXT:	.cfi_def_cfa_offset 71888
; CHECK-X86-64-NEXT:	movl	$1, 264(%rsp)
; CHECK-X86-64-NEXT:	movl	$1, 28664(%rsp)
; CHECK-X86-64-NEXT:	movl	-128(%rsp), %eax
; CHECK-X86-64-NEXT:	addq	$71880, %rsp # imm = 0x118C8
; CHECK-X86-64-NEXT:	.cfi_def_cfa_offset 8
; CHECK-X86-64-NEXT:	retq

; CHECK-X86-32-LABEL: foo:
; CHECK-X86-32:      # %bb.0:
; CHECK-X86-32-NEXT:    movl    %esp, %r11d
; CHECK-X86-32-NEXT:    subl    $69632, %r11d # imm = 0x11000
; CHECK-X86-32-NEXT:    .LBB0_1: # =>This Inner Loop Header: Depth=1
; CHECK-X86-32-NEXT:    subl    $4096, %esp # imm = 0x1000
; CHECK-X86-32-NEXT:    movl    $0, (%esp)
; CHECK-X86-32-NEXT:    cmpl    %r11d, %esp
; CHECK-X86-32-NEXT:    jne  .LBB0_1
; CHECK-X86-32-NEXT:# %bb.2:
; CHECK-X86-32-NEXT:    subl    $2380, %esp
; CHECK-X86-32-NEXT:    .cfi_def_cfa_offset 72016
; CHECK-X86-32-NEXT:    movl    $1, 392(%esp)
; CHECK-X86-32-NEXT:    movl    $1, 28792(%esp)
; CHECK-X86-32-NEXT:    movl    (%esp), %eax
; CHECK-X86-32-NEXT:    addl    $72012, %esp # imm = 0x1194C
; CHECK-X86-32-NEXT:    .cfi_def_cfa_offset 4
; CHECK-X86-32-NEXT:    retl
