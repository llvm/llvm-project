; RUN: llc < %s -mtriple=x86_64-linux-gnux32  | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux-gnux32 -fast-isel | FileCheck %s

; Test for x32 function pointer tail call

@foo1 = external dso_local global void (i8*)*
@foo2 = external dso_local global void (i8*)*

define void @bar(i8* %h) nounwind uwtable {
entry:
  %0 = load void (i8*)*, void (i8*)** @foo1, align 4
; CHECK: movl	foo1(%rip), %e{{[^,]*}}
  tail call void %0(i8* %h) nounwind
; CHECK: callq	*%r{{[^,]*}}
  %1 = load void (i8*)*, void (i8*)** @foo2, align 4
; CHECK: movl	foo2(%rip), %e{{[^,]*}}
  tail call void %1(i8* %h) nounwind
; CHECK: jmpq	*%r{{[^,]*}}
  ret void
}
