; RUN: llc < %s -mtriple=x86_64-pc-linux -enable-misched=false | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-pc-linux-gnux32 -enable-misched=false | FileCheck %s -check-prefix=X32ABI

declare void @bar(<2 x i64>* %n)

define void @foo(i64 %h) {
  %p = alloca <2 x i64>, i64 %h
  call void @bar(<2 x i64>* %p)
  ret void
; CHECK-LABEL: foo
; CHECK-NOT: andq $-32, %rax
; X32ABI-LABEL: foo
; X32ABI-NOT: andl $-32, %eax
}

define void @foo2(i64 %h) {
  %p = alloca <2 x i64>, i64 %h, align 32
  call void @bar(<2 x i64>* %p)
  ret void
; CHECK-LABEL: foo2
; CHECK: andq $-32, %rsp
; CHECK: andq $-32, %rax
; X32ABI-LABEL: foo2
; X32ABI: andl $-32, %esp
; X32ABI: andl $-32, %eax
}

define void @foo3(i64 %h) {
  %p = alloca <2 x i64>, i64 %h
  ret void
; CHECK-LABEL: foo3
; CHECK: movq %rbp, %rsp
; X32ABI-LABEL: foo3
; X32ABI: movl %ebp, %esp
}
