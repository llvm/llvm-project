; RUN: llc < %s -march=x86 -o %t
; RUN: grep "movl	.48, %ecx" %t
; RUN: grep "movl	.24, %edx" %t
; RUN: grep "movl	.12, %eax" %t

%0 = type { i32, i32, i32 }

define internal fastcc %0 @ReturnBigStruct() nounwind readnone {
entry:
  %0 = insertvalue %0 zeroinitializer, i32 12, 0
  %1 = insertvalue %0 %0, i32 24, 1
  %2 = insertvalue %0 %1, i32 48, 2
  ret %0 %2
}

