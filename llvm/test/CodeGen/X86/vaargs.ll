; RUN: llc -verify-machineinstrs -mcpu=corei7-avx %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=NO-FLAGS
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.__va_list_tag = type { i32, i32, i8*, i8* }

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
  %1 = bitcast [1 x %struct.__va_list_tag]* %ap to i8*
  call void @llvm.va_start(i8* %1)
  %2 = icmp sgt i32 %count, 0
  br i1 %2, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0
  %3 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i64 0, i64 0, i32 0
  %4 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i64 0, i64 0, i32 2
  %.pre = load i32, i32* %3, align 16
  br label %5

; <label>:5                                       ; preds = %.lr.ph, %13
  %6 = phi i32 [ %.pre, %.lr.ph ], [ %14, %13 ]
  %.01 = phi i32 [ %count, %.lr.ph ], [ %15, %13 ]
  %7 = icmp ult i32 %6, 41
  br i1 %7, label %8, label %10

; <label>:8                                       ; preds = %5
  %9 = add i32 %6, 8
  store i32 %9, i32* %3, align 16
  br label %13

; <label>:10                                      ; preds = %5
  %11 = load i8*, i8** %4, align 8
  %12 = getelementptr i8, i8* %11, i64 8
  store i8* %12, i8** %4, align 8
  br label %13

; <label>:13                                      ; preds = %10, %8
  %14 = phi i32 [ %6, %10 ], [ %9, %8 ]
  %15 = add nsw i32 %.01, 1
  %16 = icmp sgt i32 %15, 0
  br i1 %16, label %5, label %._crit_edge

._crit_edge:                                      ; preds = %13, %0
  %.0.lcssa = phi i32 [ %count, %0 ], [ %15, %13 ]
  call void @llvm.va_end(i8* %1)
  ret i32 %.0.lcssa
}

declare void @llvm.va_start(i8*) nounwind

declare void @llvm.va_end(i8*) nounwind
