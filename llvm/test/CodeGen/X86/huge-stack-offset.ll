; RUN: llc < %s -mtriple=x86_64-linux-unknown -verify-machineinstrs | FileCheck %s

; Test that a large stack offset uses a single add/sub instruction to
; adjust the stack pointer.

define void @foo() nounwind {
; CHECK--LABEL: foo:
; CHECK:      movabsq $50000000{{..}}, %rax
; CHECK-NEXT: subq    %rax, %rsp
; CHECK-NOT:  subq    $2147483647, %rsp
; CHECK:      movabsq $50000000{{..}}, [[RAX:%r..]]
; CHECK-NEXT: addq    [[RAX]], %rsp
  %1 = alloca [5000000000 x i8], align 16
  call void @bar(ptr %1)
  ret void
}

; Verify that we do not clobber the return value.

define i32 @foo2() nounwind {
; CHECK-LABEL: foo2:
; CHECK:     movl    $10, %eax
; CHECK-NOT: movabsq ${{.*}}, %rax
  %1 = alloca [5000000000 x i8], align 16
  call void @bar(ptr %1)
  ret i32 10
}

; Verify that we do not clobber EAX when using inreg attribute

define i32 @foo3(i32 inreg %x) nounwind {
; CHECK-LABEL: foo3:
; CHECK:      movabsq $50000000{{..}}, %rax
; CHECK-NEXT: subq    %rax, %rsp
  %1 = alloca [5000000000 x i8], align 16
  call void @bar(ptr %1)
  ret i32 %x
}

declare void @bar(ptr)
