; RUN: llc < %s -mtriple=x86_64-linux -mcpu=corei7 | FileCheck %s --check-prefix=COMMON --check-prefix=LINUX
; RUN: llc < %s -mtriple=x86_64-win32 -mcpu=corei7 | FileCheck %s --check-prefix=COMMON --check-prefix=MSVC

; llc should share constant pool entries between this integer vector
; and this floating-point vector since they have the same encoding.

; LINUX:   LCPI0_0(%rip), %xmm0
; MSVC:    __xmm@40000000400000004000000040000000(%rip), %xmm0
; COMMON:  movaps        %xmm0, ({{%rdi|%rcx}})
; COMMON:  movaps        %xmm0, ({{%rsi|%rdx}})

define void @foo(ptr %p, ptr %q, i1 %t) nounwind {
entry:
  br label %loop
loop:
  store <4 x i32><i32 1073741824, i32 1073741824, i32 1073741824, i32 1073741824>, ptr %p
  store <4 x float><float 2.0, float 2.0, float 2.0, float 2.0>, ptr %q
  br i1 %t, label %loop, label %ret
ret:
  ret void
}
