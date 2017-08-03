; RUN: llc < %s -disable-fp-elim -march=x86 | FileCheck %s --check-prefix=CHECK-X86
; RUN: llc < %s -disable-fp-elim -march=x86-64 | FileCheck %s --check-prefix=CHECK-X64

define i8* @f() nounwind readnone optsize {
entry:
  %0 = tail call i8* @llvm.addressofreturnaddress()    ; <i8*> [#uses=1]
  ret i8* %0
  ; CHECK-X86-LABEL: f:
  ; CHECK-X86: pushl   %ebp
  ; CHECK-X86: movl    %esp, %ebp
  ; CHECK-X86: leal    4(%ebp), %eax
  
  ; CHECK-X64-LABEL: f:
  ; CHECK-X64: pushq   %rbp
  ; CHECK-X64: movq    %rsp, %rbp
  ; CHECK-X64: leaq    8(%rbp), %rax
}

declare i8* @llvm.addressofreturnaddress() nounwind readnone
