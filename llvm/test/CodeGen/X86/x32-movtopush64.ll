; RUN: llc < %s -mtriple=x86_64-linux-gnux32 | FileCheck %s

declare void @bar(i32*, i32*, i32*, i32*, i32*, i64*, i32, i32, i32)

; Function Attrs: nounwind uwtable
define void @foo() {
entry:
  %i1 = alloca i32, align 4
  %i2 = alloca i32, align 4
  %i3 = alloca i32, align 4
  %i4 = alloca i32, align 4
  %i5 = alloca i32, align 4
  %i6 = alloca i64, align 8
  %0 = bitcast i32* %i1 to i8*
  store i32 1, i32* %i1, align 4
; CHECK: movl $1, 28(%esp)
  %1 = bitcast i32* %i2 to i8*
  store i32 2, i32* %i2, align 4
; CHECK-NEXT: movl $2, 24(%esp)
  %2 = bitcast i32* %i3 to i8*
  store i32 3, i32* %i3, align 4
; CHECK-NEXT: movl $3, 20(%esp)
  %3 = bitcast i32* %i4 to i8*
  store i32 4, i32* %i4, align 4
; CHECK-NEXT: movl $4, 16(%esp)
  %4 = bitcast i32* %i5 to i8*
  store i32 5, i32* %i5, align 4
; CHECK-NEXT: movl $5, 12(%esp)
  %5 = bitcast i64* %i6 to i8*
  store i64 6, i64* %i6, align 8
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
  call void @bar(i32* nonnull %i1, i32* nonnull %i2, i32* nonnull %i3, i32* nonnull %i4, i32* nonnull %i5, i64* nonnull %i6, i32 0, i32 0, i32 0)
  ret void
}
