; RUN: llc < %s -mtriple=x86_64-linux-gnux32  | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux-gnux32 -fast-isel | FileCheck %s

; Test for x32 function pointer tail call

@foo1 = external dso_local global ptr
@foo2 = external dso_local global ptr

define void @bar(ptr %h) nounwind uwtable {
entry:
  %0 = load ptr, ptr @foo1, align 4
; CHECK: movl	foo1(%rip), %e{{[^,]*}}
  tail call void %0(ptr %h) nounwind
; CHECK: callq	*%r{{[^,]*}}
  %1 = load ptr, ptr @foo2, align 4
; CHECK: movl	foo2(%rip), %e{{[^,]*}}
  tail call void %1(ptr %h) nounwind
; CHECK: jmpq	*%r{{[^,]*}}
  ret void
}
