; RUN: llc < %s -mtriple=x86_64-linux-unknown -verify-machineinstrs | FileCheck %s --check-prefix=CHECK-64
; RUN: llc < %s -mtriple=i386-linux-unknown -verify-machineinstrs | FileCheck %s --check-prefix=CHECK-32

; Test that a large stack offset uses a single add/sub instruction to
; adjust the stack pointer.

define void @foo() nounwind {
; CHECK-64-LABEL: foo:
; CHECK-64:      movabsq $50000000{{..}}, %rax
; CHECK-64-NEXT: subq    %rax, %rsp
; CHECK-64-NOT:  subq    $2147483647, %rsp
; CHECK-64:      movabsq $50000000{{..}}, [[RAX:%r..]]
; CHECK-64-NEXT: addq    [[RAX]], %rsp

; CHECK-32-LABEL: foo:
; CHECK-32:      movl    $50000000{{..}}, %eax
; CHECK-32-NEXT: subl    %eax, %esp
; CHECK-32-NOT:  subl    $2147483647, %esp
; CHECK-32:      movl    $50000000{{..}}, [[EAX:%e..]]
; CHECK-32-NEXT: addl    [[EAX]], %esp
  %1 = alloca [5000000000 x i8], align 16
  call void @bar(ptr %1)
  ret void
}

; Verify that we do not clobber the return value.

define i32 @foo2() nounwind {
; CHECK-64-LABEL: foo2:
; CHECK-64:     movl    $10, %eax
; CHECK-64-NOT: movabsq ${{.*}}, %rax

; CHECK-32-LABEL: foo2:
; CHECK-32:     movl    $10, %eax
; CHECK-32-NOT: movl    ${{.*}}, %eax
  %1 = alloca [5000000000 x i8], align 16
  call void @bar(ptr %1)
  ret i32 10
}

; Verify that we do not clobber EAX when using inreg attribute

define i32 @foo3(i32 inreg %x) nounwind {
; CHECK-64-LABEL: foo3:
; CHECK-64:      movabsq $50000000{{..}}, %rax
; CHECK-64-NEXT: subq    %rax, %rsp

; CHECK-32-LABEL: foo3:
; CHECK-32:      subl $2147483647, %esp
; CHECK-32-NOT:  movl ${{.*}}, %eax
  %1 = alloca [5000000000 x i8], align 16
  call void @bar(ptr %1)
  ret i32 %x
}

declare void @bar(ptr)
