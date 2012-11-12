; RUN: llc -march=mips -relocation-model=static  < %s | FileCheck %s

@.str = internal unnamed_addr constant [10 x i8] c"AAAAAAAAA\00"
@i0 = internal unnamed_addr constant [5 x i32] [ i32 0, i32 1, i32 2, i32 3, i32 4 ]

define i8* @foo() nounwind {
entry:
; CHECK: foo
; CHECK: %hi(.str)
; CHECK: %lo(.str)
	ret i8* getelementptr ([10 x i8]* @.str, i32 0, i32 0)
}

define i32* @bar() nounwind  {
entry:
; CHECK: bar
; CHECK: %hi(i0)
; CHECK: %lo(i0)
  ret i32* getelementptr ([5 x i32]* @i0, i32 0, i32 0)
}

; CHECK: rodata.str1.4,"aMS",@progbits
; CHECK: rodata,"a",@progbits
