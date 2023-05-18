; RUN: llc < %s -mtriple=x86_64-linux-gnux32 | FileCheck %s

declare void @bar(ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32)

; Function Attrs: nounwind uwtable
define void @foo() {
entry:
  %i1 = alloca i32, align 4
  %i2 = alloca i32, align 4
  %i3 = alloca i32, align 4
  %i4 = alloca i32, align 4
  %i5 = alloca i32, align 4
  %i6 = alloca i64, align 8
  store i32 1, ptr %i1, align 4
; CHECK: movl $1, 28(%esp)
  store i32 2, ptr %i2, align 4
; CHECK-NEXT: movl $2, 24(%esp)
  store i32 3, ptr %i3, align 4
; CHECK-NEXT: movl $3, 20(%esp)
  store i32 4, ptr %i4, align 4
; CHECK-NEXT: movl $4, 16(%esp)
  store i32 5, ptr %i5, align 4
; CHECK-NEXT: movl $5, 12(%esp)
  store i64 6, ptr %i6, align 8
; CHECK-NEXT: movq $6, 32(%esp)
; CHECK-NEXT: subl $8, %esp
; CHECK: leal 36(%rsp), %edi
; CHECK-NEXT: leal 32(%rsp), %esi
; CHECK-NEXT: leal 28(%rsp), %edx
; CHECK-NEXT: leal 24(%rsp), %ecx
; CHECK-NEXT: leal 20(%rsp), %r8d
; CHECK-NEXT: leal 40(%rsp), %r9d
; CHECK: pushq $0
; CHECK: pushq $0
; CHECK: pushq $0
  call void @bar(ptr nonnull %i1, ptr nonnull %i2, ptr nonnull %i3, ptr nonnull %i4, ptr nonnull %i5, ptr nonnull %i6, i32 0, i32 0, i32 0)
  ret void
}
