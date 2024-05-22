; RUN: llc -mtriple=x86_64-unknown-unknown -no-integrated-as < %s 2>&1 | FileCheck %s

define ptr @foo(ptr %Ptr) {
; CHECK-LABEL: foo:
; asm {mov rax, Pointer; lea rax, Pointer}
; LEA: Computes the effective address of the second operand and stores it in the first operand
  %Ptr.addr = alloca ptr, align 8
  store ptr %Ptr, ptr %Ptr.addr, align 8
; CHECK: movq    %rdi, -8(%rsp)
  %1 = tail call ptr asm "mov $1, $0\0A\09lea $2, $0", "=r,p,*m,~{dirflag},~{fpsr},~{flags}"(ptr %Ptr, ptr elementtype(ptr) %Ptr.addr)
; CHECK-NEXT: #APP
; CHECK-NEXT: mov (%rdi), %rax
; CHECK-NEXT: lea -8(%rsp), %rax
; CHECK-NEXT: #NO_APP
  ret ptr %1
; CHECK-NEXT: retq
}

define void @intptr() {
; Don't assert on a non-ptr operand, existing code & gcc accept these.
entry:
; CHECK-LABEL: intptr:
; CHECK: ud1l 49150(%eax), %eax
  call void asm "ud1l $0(%eax), %eax", "p,~{dirflag},~{fpsr},~{flags}"(i32 49150)
  unreachable
}
