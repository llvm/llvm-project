; RUN: llc -verify-machineinstrs -mcpu=corei7-avx %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=NO-FLAGS
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.__va_list_tag = type { i32, i32, ptr, ptr }

; Check that vastart gets the right thing.
define i32 @sum(i32 %count, ...) nounwind optsize ssp uwtable {
; CHECK:      testb   %al, %al
; CHECK-NEXT: je
; CHECK-NEXT: ## %bb.{{[0-9]+}}:
; CHECK-NEXT: vmovaps %xmm0, 48(%rsp)
; CHECK-NEXT: vmovaps %xmm1, 64(%rsp)
; CHECK-NEXT: vmovaps %xmm2, 80(%rsp)
; CHECK-NEXT: vmovaps %xmm3, 96(%rsp)
; CHECK-NEXT: vmovaps %xmm4, 112(%rsp)
; CHECK-NEXT: vmovaps %xmm5, 128(%rsp)
; CHECK-NEXT: vmovaps %xmm6, 144(%rsp)
; CHECK-NEXT: vmovaps %xmm7, 160(%rsp)

; Check that [EFLAGS] hasn't been pulled in.
; NO-FLAGS-NOT: %flags

  %ap = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.va_start(ptr %ap)
  %1 = icmp sgt i32 %count, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %2 = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %ap, i64 0, i64 0, i32 2
  %.pre = load i32, ptr %ap, align 16
  br label %3

; <label>:5                                       ; preds = %.lr.ph, %11
  %4 = phi i32 [ %.pre, %.lr.ph ], [ %12, %11 ]
  %.01 = phi i32 [ %count, %.lr.ph ], [ %13, %11 ]
  %5 = icmp ult i32 %4, 41
  br i1 %5, label %6, label %8

; <label>:8                                       ; preds = %3
  %7 = add i32 %4, 8
  store i32 %7, ptr %ap, align 16
  br label %11

; <label>:10                                      ; preds = %3
  %9 = load ptr, ptr %2, align 8
  %10 = getelementptr i8, ptr %9, i64 8
  store ptr %10, ptr %2, align 8
  br label %11

; <label>:13                                      ; preds = %8, %6
  %12 = phi i32 [ %4, %8 ], [ %7, %6 ]
  %13 = add nsw i32 %.01, 1
  %14 = icmp sgt i32 %13, 0
  br i1 %14, label %3, label %._crit_edge

._crit_edge:                                      ; preds = %11, %0
  %.0.lcssa = phi i32 [ %count, %0 ], [ %13, %11 ]
  call void @llvm.va_end(ptr %ap)
  ret i32 %.0.lcssa
}

declare void @llvm.va_start(ptr) nounwind

declare void @llvm.va_end(ptr) nounwind
