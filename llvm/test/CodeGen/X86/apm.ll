; RUN: llc < %s -mtriple=x86_64-linux -mattr=+sse3 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 -mattr=+sse3 | FileCheck %s -check-prefix=WIN64
; PR8573

; CHECK-LABEL: foo:
; CHECK: leaq    (%rdi), %rax
; CHECK-NEXT: movl    %esi, %ecx
; CHECK-NEXT: monitor
; WIN64-LABEL: foo:
; WIN64:      leaq    (%rcx), %rax
; WIN64-NEXT: movl    %edx, %ecx
; WIN64-NEXT: movl    %r8d, %edx
; WIN64-NEXT: monitor
define void @foo(i8* %P, i32 %E, i32 %H) nounwind {
entry:
  tail call void @llvm.x86.sse3.monitor(i8* %P, i32 %E, i32 %H)
  ret void
}

declare void @llvm.x86.sse3.monitor(i8*, i32, i32) nounwind

; CHECK-LABEL: bar:
; CHECK: movl    %edi, %ecx
; CHECK-NEXT: movl    %esi, %eax
; CHECK-NEXT: mwait
; WIN64-LABEL: bar:
; WIN64:      movl    %edx, %eax
; WIN64-NEXT: mwait
define void @bar(i32 %E, i32 %H) nounwind {
entry:
  tail call void @llvm.x86.sse3.mwait(i32 %E, i32 %H)
  ret void
}

declare void @llvm.x86.sse3.mwait(i32, i32) nounwind
