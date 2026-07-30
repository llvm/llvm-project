; RUN: llc < %s -mtriple=i686-apple-darwin | FileCheck %s

declare void @bar(ptr %n)

define void @foo(i32 %h) {
  %p = alloca <2 x i64>, i32 %h
  call void @bar(ptr %p)
  ret void
; CHECK: foo
; CHECK-NOT: andl $-32, %eax
}

define void @foo2(i32 %h) {
  %p = alloca <2 x i64>, i32 %h, align 32
  call void @bar(ptr %p)
  ret void
; CHECK: foo2
; CHECK: andl $-32, %esp
; CHECK: andl $-32, %e{{..}}
}
