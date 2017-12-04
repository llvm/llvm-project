; RUN: llc < %s -mtriple=x86_64-pc-win32-coreclr | FileCheck %s -check-prefix=WIN_X64
; RUN: llc < %s -mtriple=x86_64-pc-linux         | FileCheck %s -check-prefix=LINUX

; By default, windows CoreCLR requires an inline prologue stack expansion check
; if more than 4096 bytes are allocated on the stack.

; Prolog stack allocation >= 4096 bytes will require the probe sequence
define i32 @main4k() nounwind {
entry:
; WIN_X64-LABEL:main4k:
; WIN_X64: # %bb.0:
; WIN_X64:      movl    $4096, %eax
; WIN_X64:      movq    %rcx, 8(%rsp)
; WIN_X64:	movq	%rdx, 16(%rsp)
; WIN_X64:	xorq	%rcx, %rcx
; WIN_X64:	movq	%rsp, %rdx
; WIN_X64:	subq	%rax, %rdx
; WIN_X64:	cmovbq	%rcx, %rdx
; WIN_X64:	movq	%gs:16, %rcx
; WIN_X64:	cmpq	%rcx, %rdx
; WIN_X64:	jae	.LBB0_3
; WIN_X64:# %bb.1:
; WIN_X64:	andq	$-4096, %rdx
; WIN_X64:.LBB0_2:
; WIN_X64:	leaq	-4096(%rcx), %rcx
; WIN_X64:	movb	$0, (%rcx)
; WIN_X64:	cmpq	%rcx, %rdx
; WIN_X64:	jne	.LBB0_2
; WIN_X64:.LBB0_3:
; WIN_X64:	movq	8(%rsp), %rcx
; WIN_X64:	movq	16(%rsp), %rdx
; WIN_X64:	subq	%rax, %rsp
; WIN_X64:	xorl	%eax, %eax
; WIN_X64:	addq	$4096, %rsp
; WIN_X64:	retq
; LINUX-LABEL:main4k:
; LINUX-NOT:    movq    %gs:16, %rcx
; LINUX: 	retq
  %a = alloca [4096 x i8]
  ret i32 0
}

; Prolog stack allocation >= 4096 bytes will require the probe sequence
; Case with frame pointer
define i32 @main4k_frame() nounwind "no-frame-pointer-elim"="true" {
entry:
; WIN_X64-LABEL:main4k_frame:
; WIN_X64:      movq    %rcx,   16(%rsp)
; WIN_X64:      movq    %gs:16, %rcx
; LINUX-LABEL:main4k_frame:
; LINUX-NOT:    movq    %gs:16, %rcx
; LINUX: 	retq
  %a = alloca [4096 x i8]
  ret i32 0
}

; Prolog stack allocation >= 4096 bytes will require the probe sequence
; Case with INT args
define i32 @main4k_intargs(i32 %x, i32 %y) nounwind {
entry:
; WIN_X64:      movq    %rcx,   8(%rsp)
; WIN_X64:      movq    %gs:16, %rcx
; LINUX-NOT:    movq    %gs:16, %rcx
; LINUX: 	retq
  %a = alloca [4096 x i8]
  %t = add i32 %x, %y
  ret i32 %t
}

; Prolog stack allocation >= 4096 bytes will require the probe sequence
; Case with FP regs
define i32 @main4k_fpargs(double %x, double %y) nounwind {
entry:
; WIN_X64:      movq    %rcx,   8(%rsp)
; WIN_X64:      movq    %gs:16, %rcx
; LINUX-NOT:    movq    %gs:16, %rcx
; LINUX: 	retq
  %a = alloca [4096 x i8]
  ret i32 0
}

; Prolog stack allocation >= 4096 bytes will require the probe sequence
; Case with mixed regs
define i32 @main4k_mixargs(double %x, i32 %y) nounwind {
entry:
; WIN_X64:      movq    %gs:16, %rcx
; LINUX-NOT:    movq    %gs:16, %rcx
; LINUX: 	retq
  %a = alloca [4096 x i8]
  ret i32 %y
}

; Make sure we don't emit the probe for a smaller prolog stack allocation.
define i32 @main128() nounwind {
entry:
; WIN_X64-NOT:  movq    %gs:16, %rcx
; WIN_X64:      retq
; LINUX-NOT:    movq    %gs:16, %rcx
; LINUX: 	retq
  %a = alloca [128 x i8]
  ret i32 0
}

; Make sure we don't emit the probe sequence if not on windows even if the
; caller has the Win64 calling convention.
define win64cc i32 @main4k_win64() nounwind {
entry:
; WIN_X64:      movq    %gs:16, %rcx
; LINUX-NOT:    movq    %gs:16, %rcx
; LINUX: 	retq
  %a = alloca [4096 x i8]
  ret i32 0
}

declare i32 @bar(i8*) nounwind

; Within-body inline probe expansion
define win64cc i32 @main4k_alloca(i64 %n) nounwind {
entry:
; WIN_X64: 	callq	bar
; WIN_X64:  	movq	%gs:16, [[R:%r.*]]
; WIN_X64: 	callq	bar
; LINUX: 	callq	bar
; LINUX-NOT:  	movq	%gs:16, [[R:%r.*]]
; LINUX: 	callq	bar
  %a = alloca i8, i64 1024
  %ra = call i32 @bar(i8* %a) nounwind
  %b = alloca i8, i64 %n
  %rb = call i32 @bar(i8* %b) nounwind
  %r = add i32 %ra, %rb
  ret i32 %r
}

; Influence of stack-probe-size attribute
; Note this is not exposed in coreclr
define i32 @test_probe_size() "stack-probe-size"="8192" nounwind {
; WIN_X64-NOT:  movq    %gs:16, %rcx
; WIN_X64: 	retq
; LINUX-NOT:    movq    %gs:16, %rcx
; LINUX: 	retq
  %a = alloca [4096 x i8]
  ret i32 0
}
