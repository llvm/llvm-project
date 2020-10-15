; BB cluster section tests.
;
; Test1: Basic blocks #0 (entry) and #2 will be placed in the same section.
; Basic block 1 will be placed in a unique section.
; The rest will be placed in the cold section.
; RUN: echo '!foo' > %t1
; RUN: echo '!!0 2' >> %t1
; RUN: echo '!!1' >> %t1
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t1 | FileCheck %s -check-prefix=LINUX-SECTIONS1
;
; Test2: Basic blocks #1 and #3 will be placed in the same section.
; All other BBs (including the entry block) go into the function's section.
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
; LINUX-SECTIONS1-NOT:  	.section
; LINUX-SECTIONS1-LABEL:	foo:
; LINUX-SECTIONS1-NOT:  	.section
; LINUX-SECTIONS1-NOT:  	.LBB_END0_{{0-9}}+
; LINUX-SECTIONS1-LABEL:	# %bb.2:
; LINUX-SECTIONS1-NOT:  	.LBB_END0_{{0-9}}+
; LINUX-SECTIONS1:		.section        .text.foo,"ax",@progbits,unique,1
; LINUX-SECTIONS1-LABEL:	foo.1:
; LINUX-SECTIONS1-LABEL:	.LBB_END0_1:
; LINUX-SECTIONS1-NEXT:        .size   foo.1, .LBB_END0_1-foo.1
; LINUX-SECTIONS1-NOT:  	.section
; LINUX-SECTIONS1:		.section        .text.split.foo,"ax",@progbits
; LINUX-SECTIONS1-LABEL:	foo.cold:
; LINUX-SECTIONS1-LABEL:	.LBB_END0_3:
; LINUX-SECTIONS1-NEXT:        .size   foo.cold, .LBB_END0_3-foo.cold
; LINUX-SECTIONS1:	   	.section	.text.foo,"ax",@progbits
; LINUX-SECTIONS1-LABEL:	.Lfunc_end0:
; LINUX-SECTIONS1-NEXT:		.size foo, .Lfunc_end0-foo

; LINUX-SECTIONS2:		.section        .text.foo,"ax",@progbits
; LINUX-SECTIONS2-NOT:   	.section
; LINUX-SECTIONS2-LABEL:	foo:
; LINUX-SECTIONS2-NOT:  	.LBB_END0_{{0-9}}+
; LINUX-SECTIONS2-NOT:   	.section
; LINUX-SECTIONS2-LABEL:	# %bb.2:
; LINUX-SECTIONS2-NOT:  	.LBB_END0_{{0-9}}+
; LINUX-SECTIONS2:		.section        .text.foo,"ax",@progbits,unique,1
; LINUX-SECTIONS2-NEXT:		foo.0:
; LINUX-SECTIONS2-NOT:  	.LBB_END0_{{0-9}}+
; LINUX-SECTIONS2-NOT:  	.section
; LINUX-SECTIONS2-LABEL:	.LBB0_3:
; LINUX-SECTIONS2-LABEL:	.LBB_END0_3:
; LINUX-SECTIONS2-NEXT:        .size   foo.0, .LBB_END0_3-foo.0
; LINUX-SECTIONS2:		.section        .text.foo,"ax",@progbits
; LINUX-SECTIONS2-NOT:  	.LBB_END0_{{0-9}}+
; LINUX-SECTIONS2-LABEL:	.Lfunc_end0:
; LINUX-SECTIONS2-NEXT:		.size foo, .Lfunc_end0-foo
