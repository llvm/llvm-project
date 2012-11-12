; RUN: llc < %s -mtriple=x86_64-linux -mcpu=corei7 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 -mcpu=corei7 | FileCheck %s

; llc should share constant pool entries between this integer vector
; and this floating-point vector since they have the same encoding.

; CHECK:  LCPI0_0(%rip), %xmm0
; CHECK:  movaps        %xmm0, ({{%rdi|%rcx}})
; CHECK:  movaps        %xmm0, ({{%rsi|%rdx}})

define void @foo(<4 x i32>* %p, <4 x float>* %q, i1 %t) nounwind {
entry:
  br label %loop
loop:
  store <4 x i32><i32 1073741824, i32 1073741824, i32 1073741824, i32 1073741824>, <4 x i32>* %p
  store <4 x float><float 2.0, float 2.0, float 2.0, float 2.0>, <4 x float>* %q
  br i1 %t, label %loop, label %ret
ret:
  ret void
}
