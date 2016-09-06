; RUN: llc < %s -mtriple=x86_64-linux -mattr=+mwaitx | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 -mattr=+mwaitx | FileCheck %s -check-prefix=WIN64
; RUN: llc < %s -mtriple=x86_64-linux -mcpu=bdver4 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 -mcpu=bdver4 | FileCheck %s -check-prefix=WIN64

; CHECK-LABEL: foo:
; CHECK: leaq    (%rdi), %rax
; CHECK-NEXT: movl    %esi, %ecx
; CHECK-NEXT: monitorx
; WIN64-LABEL: foo:
; WIN64:      leaq    (%rcx), %rax
; WIN64-NEXT: movl    %edx, %ecx
; WIN64-NEXT: movl    %r8d, %edx
; WIN64-NEXT: monitorx
define void @foo(i8* %P, i32 %E, i32 %H) nounwind {
entry:
  tail call void @llvm.x86.monitorx(i8* %P, i32 %E, i32 %H)
  ret void
}

declare void @llvm.x86.monitorx(i8*, i32, i32) nounwind

; CHECK-LABEL: bar:
; CHECK: movl    %edi, %ecx
; CHECK-NEXT: movl    %esi, %eax
; CHECK-NEXT: movl    %edx, %ebx
; CHECK-NEXT: mwaitx
; WIN64-LABEL: bar:
; WIN64:      movl    %edx, %eax
; WIN64:      movl    %r8d, %ebx
; WIN64-NEXT: mwaitx
define void @bar(i32 %E, i32 %H, i32 %C) nounwind {
entry:
  tail call void @llvm.x86.mwaitx(i32 %E, i32 %H, i32 %C)
  ret void
}

declare void @llvm.x86.mwaitx(i32, i32, i32) nounwind
