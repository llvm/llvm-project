; RUN: llc < %s -mtriple x86_64-apple-darwin | FileCheck %s

@a = thread_local global i32 4, align 4

define i32 @f2(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5) {
entry:
; Parameters are in %edi, %esi, %edx, %ecx, %r8d, there is no need to save
; these parameters except the one in %edi, before making the TLS call.
; %edi is used to pass parameter to the TLS call.
; CHECK-NOT: movl %r8d
; CHECK-NOT: movl %ecx
; CHECK-NOT: movl %edx
; CHECK-NOT: movl %esi
; CHECK: movq {{.*}}TLVP{{.*}}, %rdi
; CHECK-NEXT: callq
; CHECK-NEXT: movl (%rax),
; CHECK-NOT: movl {{.*}}, %esi
; CHECK-NOT: movl {{.*}}, %edx
; CHECK-NOT: movl {{.*}}, %ecx
; CHECK-NOT: movl {{.*}}, %r8d
; CHECK: callq
  %0 = load i32, i32* @a, align 4
  %call = tail call i32 @f3(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5)
  %add = add nsw i32 %call, %0
  ret i32 %add
}

declare i32 @f3(i32, i32, i32, i32, i32)
