; RUN: llvm-as < %s | llc -march=x86 > %t
; RUN: not grep lea %t
; RUN: not grep add %t
; RUN: grep mov %t | count 1
; RUN: llvm-as < %s | llc -march=x86-64 -relocation-model=static > %t
; RUN: not grep lea %t
; RUN: not grep add %t
; RUN: grep mov %t | count 1

; This store should fold to a single mov instruction.

@ptr = global i32* null
@dst = global [131072 x i32] zeroinitializer

define void @foo() nounwind {
  store i32* getelementptr ([131072 x i32]* @dst, i32 0, i32 16), i32** @ptr
  ret void
}
