; RUN: llc < %s -mtriple=aarch64-pc-win32 | FileCheck %s

define void @pass_va(i32 %count, ...) nounwind {
entry:
; CHECK: sub     sp, sp, #80
; CHECK: add     x8, sp, #24
; CHECK: add     x0, sp, #24
; CHECK: stp     x6, x7, [sp, #64]
; CHECK: stp     x4, x5, [sp, #48]
; CHECK: stp     x2, x3, [sp, #32]
; CHECK: str     x1, [sp, #24]
; CHECK: stp     x30, x8, [sp]
; CHECK: bl      other_func
; CHECK: ldr     x30, [sp], #80
; CHECK: ret
  %ap = alloca i8*, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %ap2 = load i8*, i8** %ap, align 8
  call void @other_func(i8* %ap2)
  ret void
}

declare void @other_func(i8*) local_unnamed_addr

declare void @llvm.va_start(i8*) nounwind
declare void @llvm.va_copy(i8*, i8*) nounwind

; CHECK-LABEL: f9:
; CHECK: sub     sp, sp, #16
; CHECK: add     x8, sp, #24
; CHECK: add     x0, sp, #24
; CHECK: str     x8, [sp, #8]
; CHECK: add     sp, sp, #16
; CHECK: ret
define i8* @f9(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %ap2 = load i8*, i8** %ap, align 8
  ret i8* %ap2
}

; CHECK-LABEL: f8:
; CHECK: sub     sp, sp, #16
; CHECK: add     x8, sp, #16
; CHECK: add     x0, sp, #16
; CHECK: str     x8, [sp, #8]
; CHECK: add     sp, sp, #16
; CHECK: ret
define i8* @f8(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %ap2 = load i8*, i8** %ap, align 8
  ret i8* %ap2
}

; CHECK-LABEL: f7:
; CHECK: sub     sp, sp, #16
; CHECK: add     x8, sp, #8
; CHECK: add     x0, sp, #8
; CHECK: stp     x8, x7, [sp], #16
; CHECK: ret
define i8* @f7(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %ap2 = load i8*, i8** %ap, align 8
  ret i8* %ap2
}

; CHECK-LABEL: copy1:
; CHECK: sub     sp, sp, #80
; CHECK: add     x8, sp, #24
; CHECK: stp     x6, x7, [sp, #64]
; CHECK: stp     x4, x5, [sp, #48]
; CHECK: stp     x2, x3, [sp, #32]
; CHECK: stp     x8, x1, [sp, #16]
; CHECK: str     x8, [sp, #8]
; CHECK: add     sp, sp, #80
; CHECK: ret
define void @copy1(i64 %a0, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %cp = alloca i8*, align 8
  %ap1 = bitcast i8** %ap to i8*
  %cp1 = bitcast i8** %cp to i8*
  call void @llvm.va_start(i8* %ap1)
  call void @llvm.va_copy(i8* %cp1, i8* %ap1)
  ret void
}
