; RUN: llc < %s -mtriple=i386-pc-linux -mcpu=corei7 | FileCheck %s

define i64 @atomicSub(i64* %a, i64 %b) nounwind {
entry:
  %0 = atomicrmw sub i64* %a, i64 %b seq_cst
  ret i64 %0
; CHECK: atomicSub
; movl %eax, %ebx
; subl {{%[a-z]+}}, %ebx
; movl %edx, %ecx
; sbbl {{%[a-z]+}}, %ecx
; CHECK: ret
}
