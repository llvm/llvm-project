; RUN: llc -mtriple=mips -relocation-model=static  < %s | FileCheck %s

@.str = internal unnamed_addr constant [10 x i8] c"AAAAAAAAA\00"
@i0 = internal unnamed_addr constant [5 x i32] [ i32 0, i32 1, i32 2, i32 3, i32 4 ]

define ptr @foo() nounwind {
entry:
; CHECK: foo
; CHECK: %hi(.str)
; CHECK: %lo(.str)
	ret ptr @.str
}

define ptr @bar() nounwind  {
entry:
; CHECK: bar
; CHECK: %hi(i0)
; CHECK: %lo(i0)
  ret ptr @i0
}

; CHECK: rodata.str1.4,"aMS",@progbits
; CHECK: rodata,"a",@progbits
