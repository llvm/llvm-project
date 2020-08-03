; BB cluster sections test for optimizing basic block branches.
;
; Test1: Basic blocks #0 (entry) and #2 will be placed in the same section.
; There should be a jne from #0 to #1 and a fallthrough to #2.
; BB #1 will be in a unique section. Therefore, it should retain its jmp to #3.
; #2 must also have an explicit jump to #3.
; RUN: echo '!foo' > %t1
; RUN: echo '!!0 2' >> %t1
; RUN: echo '!!1' >> %t1
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t1 | FileCheck %s -check-prefix=LINUX-SECTIONS1
;
; Test2: Basic blocks #1 and #3 will be placed in the same section.
; The rest (#0 and #2) go into the function's section.
; This means #1 must fall through to #3, and #0 must fall through to #2.
; #2 must have an explicit jump to #3.
; RUN: echo '!foo' > %t2
; RUN: echo '!!1 3' >> %t2
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t2 | FileCheck %s -check-prefix=LINUX-SECTIONS2

define void @foo(i1 zeroext) nounwind {
  %2 = alloca i8, align 1
  %3 = zext i1 %0 to i8
  store i8 %3, i8* %2, align 1
  %4 = load i8, i8* %2, align 1
  %5 = trunc i8 %4 to i1
  br i1 %5, label %6, label %8

6:                                                ; preds = %1
  %7 = call i32 @bar()
  br label %10

8:                                                ; preds = %1
  %9 = call i32 @baz()
  br label %10

10:                                               ; preds = %8, %6
  ret void
}

declare i32 @bar() #1

declare i32 @baz() #1

; LINUX-SECTIONS1:	   	.section	.text.foo,"ax",@progbits
; LINUX-SECTIONS1-LABEL:	foo:
; LINUX-SECTIONS1:		jne foo.1
; LINUX-SECTIONS1-NOT:		{{jne|je|jmp}}
; LINUX-SECTIONS1-LABEL:	# %bb.2:
; LINUX-SECTIONS1:		jmp foo.cold
; LINUX-SECTIONS1:		.section        .text.foo,"ax",@progbits,unique,1
; LINUX-SECTIONS1-LABEL:	foo.1:
; LINUX-SECTIONS1:		jmp foo.cold
; LINUX-SECTIONS1:		.section        .text.unlikely.foo,"ax",@progbits
; LINUX-SECTIONS1-LABEL:	foo.cold:

; LINUX-SECTIONS2:		.section        .text.foo,"ax",@progbits
; LINUX-SECTIONS2-LABEL:	foo:
; LINUX-SECTIONS2:		jne foo.0
; LINUX-SECTIONS2-NOT:		{{jne|je|jmp}}
; LINUX-SECTIONS2-LABEL:	# %bb.2:
; LINUX-SECTIONS2:		jmp .LBB0_3
; LINUX-SECTIONS2:		.section        .text.foo,"ax",@progbits,unique,1
; LINUX-SECTIONS2:		foo.0:
; LINUX-SECTIONS2-NOT:		{{jne|je|jmp}}
; LINUX-SECTIONS2:		.LBB0_3:
