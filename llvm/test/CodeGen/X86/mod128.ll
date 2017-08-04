; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s -check-prefix=X86-64
; RUN: llc < %s -mtriple=x86_64-cygwin | FileCheck %s -check-prefix=WIN64
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s -check-prefix=WIN64
; RUN: llc < %s -mtriple=x86_64-mingw32 | FileCheck %s -check-prefix=WIN64

define i64 @mod128(i128 %x) {
  ; X86-64: movl  $3, %edx
  ; X86-64: xorl  %ecx, %ecx
  ; X86-64: callq __modti3
  ; X86-64-NOT: movd %xmm0, %rax

  ; WIN64-NOT: movl $3, %r8d
  ; WIN64-NOT: xorl %r9d, %r9d
  ; WIN64-DAG: movq %rdx, 56(%rsp)
  ; WIN64-DAG: movq %rcx, 48(%rsp)
  ; WIN64-DAG: leaq 48(%rsp), %rcx
  ; WIN64-DAG: leaq 32(%rsp), %rdx
  ; WIN64-DAG: movq $0, 40(%rsp)
  ; WIN64-DAG: movq $3, 32(%rsp)
  ; WIN64: callq   __modti3
  ; WIN64: movq    %xmm0, %rax

  %1 = srem i128 %x, 3
  %2 = trunc i128 %1 to i64
  ret i64 %2
}
