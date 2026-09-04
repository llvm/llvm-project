; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -verify-machineinstrs | FileCheck %s --check-prefix=X64
; RUN: llc < %s -mtriple=i686-unknown-linux-gnu -mattr=+avx2 -verify-machineinstrs | FileCheck %s --check-prefix=X86

; The variable-index extract needs a 32-byte aligned stack temporary, so the
; stack is realigned. The callee load from a local must not be folded into the
; tail call, which executes after the epilogue.

@g2 = global i16 0, align 2

define void @f25() nounwind {
; X64-LABEL: f25:
; X64:         andq $-32, %rsp
; X64:         movq {{[0-9]*}}(%rsp), %[[FP:r[a-z0-9]+]]
; X64-NOT:     jmpq *{{.*}}(%rsp)
; X64:         jmpq *%[[FP]]
;
; X86-LABEL: f25:
; X86:         andl $-32, %esp
; X86:         movl {{[0-9]*}}(%esp), %[[FP:e[a-z]+]]
; X86-NOT:     jmpl *{{.*}}(%esp)
; X86:         jmpl *%[[FP]]
entry:
  %fp5 = alloca ptr, align 8
  %0 = load i16, ptr @g2, align 2
  %vecext = extractelement <8 x i32> <i32 60, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, i16 %0
  %conv = trunc i32 %vecext to i16
  store i16 %conv, ptr @g2, align 2
  %fp = load volatile ptr, ptr %fp5, align 8
  tail call void %fp()
  ret void
}
