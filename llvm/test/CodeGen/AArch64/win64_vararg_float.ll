; RUN: llc < %s -mtriple=aarch64-windows -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,DAGISEL
; RUN: llc < %s -mtriple=aarch64-windows -verify-machineinstrs -O0 -fast-isel | FileCheck %s --check-prefixes=CHECK,O0
; RUN: llc < %s -mtriple=aarch64-windows -verify-machineinstrs -O0 -global-isel | FileCheck %s --check-prefixes=CHECK,O0

define void @float_va_fn(float %a, i32 %b, ...) nounwind {
entry:
; CHECK-LABEL: float_va_fn:
; O0: str x7, [sp, #72]
; O0: str x6, [sp, #64]
; O0: str x5, [sp, #56]
; O0: str x4, [sp, #48]
; O0: str x3, [sp, #40]
; O0: str x2, [sp, #32]
; CHECK: fmov s0, w0
; O0: add x8, sp, #32
; O0: str x8, [sp, #8]
; O0: ldr x0, [sp, #8]
; DAGISEL: add x0, sp, #16
; DAGISEL: stp x3, x4, [sp, #24]
; DAGISEL: stp x5, x6, [sp, #40]
; DAGISEL: stp x8, x2, [sp, #8]
; CHECK: bl f_va_list
  %ap = alloca i8*, align 8
  %0 = bitcast i8** %ap to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0)
  call void @llvm.va_start(i8* nonnull %0)
  %1 = load i8*, i8** %ap, align 8
  call void @f_va_list(float %a, i8* %1)
  call void @llvm.va_end(i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.va_start(i8*)
declare void @f_va_list(float, i8*)
declare void @llvm.va_end(i8*)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

define void @double_va_fn(double %a, i32 %b, ...) nounwind {
entry:
; CHECK-LABEL: double_va_fn:
; O0: str x7, [sp, #72]
; O0: str x6, [sp, #64]
; O0: str x5, [sp, #56]
; O0: str x4, [sp, #48]
; O0: str x3, [sp, #40]
; O0: str x2, [sp, #32]
; CHECK: fmov d0, x0
; O0: add x8, sp, #32
; O0: str x8, [sp, #8]
; O0: ldr x0, [sp, #8]
; DAGISEL: add x0, sp, #16
; DAGISEL: stp x3, x4, [sp, #24]
; DAGISEL: stp x5, x6, [sp, #40]
; DAGISEL: stp x8, x2, [sp, #8]
; CHECK: bl d_va_list
  %ap = alloca i8*, align 8
  %0 = bitcast i8** %ap to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0)
  call void @llvm.va_start(i8* nonnull %0)
  %1 = load i8*, i8** %ap, align 8
  call void @d_va_list(double %a, i8* %1)
  call void @llvm.va_end(i8* nonnull %0)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0)
  ret void
}

declare void @d_va_list(double, i8*)

define void @call_f_va() nounwind {
entry:
; CHECK-LABEL: call_f_va:
; DAGISEL: mov w0, #1065353216
; FASTISEL: mov w0, #1065353216
; GISEL: fmov s0, #1.00000000
; GISEL: fmov w0, s0
; CHECK: mov w1, #2
; DAGISEL: mov x2, #4613937818241073152
; FASTISEL: mov x2, #4613937818241073152
; GISEL: fmov d0, #3.00000000
; GISEL: fmov x2, d0
; CHECK: mov w3, #4
; CHECK: b other_f_va_fn
  tail call void (float, i32, ...) @other_f_va_fn(float 1.000000e+00, i32 2, double 3.000000e+00, i32 4)
  ret void
}

declare void @other_f_va_fn(float, i32, ...)

define void @call_d_va() nounwind {
entry:
; CHECK-LABEL: call_d_va:
; DAGISEL: mov x0, #4607182418800017408
; FASTISEL: mov x0, #4607182418800017408
; GISEL: fmov d0, #1.00000000
; GISEL: fmov x0, d0
; CHECK: mov w1, #2
; DAGISEL: mov x2, #4613937818241073152
; FASTISEL: mov x2, #4613937818241073152
; GISEL: fmov d0, #3.00000000
; CHECK: mov w3, #4
; CHECK: b other_d_va_fn
  tail call void (double, i32, ...) @other_d_va_fn(double 1.000000e+00, i32 2, double 3.000000e+00, i32 4)
  ret void
}

declare void @other_d_va_fn(double, i32, ...)

define void @call_d_non_va() nounwind {
entry:
; CHECK-LABEL: call_d_non_va:
; CHECK-DAG: fmov d0, #1.00000000
; CHECK-DAG: fmov d1, #3.00000000
; CHECK-DAG: mov w0, #2
; CHECK-DAG: mov w1, #4
; CHECK: b other_d_non_va_fn
  tail call void (double, i32, double, i32) @other_d_non_va_fn(double 1.000000e+00, i32 2, double 3.000000e+00, i32 4)
  ret void
}

declare void @other_d_non_va_fn(double, i32, double, i32)
