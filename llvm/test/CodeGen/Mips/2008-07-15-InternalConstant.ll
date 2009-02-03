; RUN: llvm-as < %s | llc -march=mips -f -o %t
; RUN: grep {rodata.str1.4,"aMS",@progbits}  %t | count 1
; RUN: grep {r.data,}  %t | count 1
; RUN: grep {\%hi} %t | count 2
; RUN: grep {\%lo} %t | count 2
; RUN: not grep {gp_rel} %t

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"
@.str = internal constant [10 x i8] c"AAAAAAAAA\00"
@i0 = internal constant [5 x i32] [ i32 0, i32 1, i32 2, i32 3, i32 4 ] 

define i8* @foo() nounwind {
entry:
	ret i8* getelementptr ([10 x i8]* @.str, i32 0, i32 0)
}

define i32* @bar() nounwind  {
entry:
  ret i32* getelementptr ([5 x i32]* @i0, i32 0, i32 0)
}

