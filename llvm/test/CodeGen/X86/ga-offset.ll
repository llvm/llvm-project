; RUN: llc < %s -mtriple=x86_64-linux -relocation-model=static | FileCheck %s

@ptr = global i32* null
@dst = global [131072 x i32] zeroinitializer

define void @foo() nounwind {
; This store should fold to a single mov instruction.
; CHECK: movq    $dst+64, ptr(%rip)
  store i32* getelementptr ([131072 x i32], [131072 x i32]* @dst, i32 0, i32 16), i32** @ptr
  ret void
}
