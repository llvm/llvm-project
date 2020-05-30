; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s

define win64cc void @pass_va(i32 %count, ...) nounwind {
entry:
; CHECK: sub     sp, sp, #96
; CHECK: add     x8, sp, #40
; CHECK: add     x0, sp, #40
; CHECK: stp     x30, x18, [sp, #16]
; CHECK: stp     x1, x2, [sp, #40]
; CHECK: stp     x3, x4, [sp, #56]
; CHECK: stp     x5, x6, [sp, #72]
; CHECK: str     x7, [sp, #88]
; CHECK: str     x8, [sp, #8]
; CHECK: bl      other_func
; CHECK: ldp     x30, x18, [sp, #16]
; CHECK: add     sp, sp, #96
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
; CHECK: str     x18, [sp, #-16]!
; CHECK: add     x8, sp, #24
; CHECK: add     x0, sp, #24
; CHECK: str     x8, [sp, #8]
; CHECK: ldr     x18, [sp], #16
; CHECK: ret
define win64cc i8* @f9(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %ap2 = load i8*, i8** %ap, align 8
  ret i8* %ap2
}

; CHECK-LABEL: f8:
; CHECK: str     x18, [sp, #-16]!
; CHECK: add     x8, sp, #16
; CHECK: add     x0, sp, #16
; CHECK: str     x8, [sp, #8]
; CHECK: ldr     x18, [sp], #16
; CHECK: ret
define win64cc i8* @f8(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %ap2 = load i8*, i8** %ap, align 8
  ret i8* %ap2
}

; CHECK-LABEL: f7:
; CHECK: str     x18, [sp, #-32]!
; CHECK: add     x8, sp, #24
; CHECK: str     x7, [sp, #24]
; CHECK: add     x0, sp, #24
; CHECK: str     x8, [sp, #8]
; CHECK: ldr     x18, [sp], #32
; CHECK: ret
define win64cc i8* @f7(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  %ap2 = load i8*, i8** %ap, align 8
  ret i8* %ap2
}
