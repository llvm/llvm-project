; RUN: llc < %s -frame-pointer=all -mtriple=i686-- | FileCheck %s --check-prefix=CHECK-X86
; RUN: llc < %s -frame-pointer=all -mtriple=x86_64-- | FileCheck %s --check-prefix=CHECK-X64

define ptr @f() nounwind readnone optsize {
entry:
  %0 = tail call ptr @llvm.addressofreturnaddress()    ; <ptr> [#uses=1]
  ret ptr %0
  ; CHECK-X86-LABEL: f:
  ; CHECK-X86: pushl   %ebp
  ; CHECK-X86: movl    %esp, %ebp
  ; CHECK-X86: leal    4(%ebp), %eax
  
  ; CHECK-X64-LABEL: f:
  ; CHECK-X64: pushq   %rbp
  ; CHECK-X64: movq    %rsp, %rbp
  ; CHECK-X64: leaq    8(%rbp), %rax
}

declare ptr @llvm.addressofreturnaddress() nounwind readnone
