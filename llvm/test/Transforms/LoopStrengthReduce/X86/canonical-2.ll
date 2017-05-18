; REQUIRES: asserts
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -loop-reduce -S < %s
; PR33077. Check the LSR Use formula to be inserted is already canonicalized and
; will not trigger assertion.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: uwtable
define void @foo() { 
cHeapLvb.exit:
  br label %not_zero48.us

not_zero48.us:                                    ; preds = %not_zero48.us, %cHeapLvb.exit
  %indvars.iv.us = phi i64 [ %indvars.iv.next.us.7, %not_zero48.us ], [ undef, %cHeapLvb.exit ]
  %0 = phi i32 [ %13, %not_zero48.us ], [ undef, %cHeapLvb.exit ]
  %indvars.iv.next.us = add nuw nsw i64 %indvars.iv.us, 1
  %1 = add i32 %0, 2
  %2 = getelementptr inbounds i32, i32 addrspace(1)* undef, i64 %indvars.iv.next.us
  %3 = load i32, i32 addrspace(1)* %2, align 4
  %4 = add i32 %0, 3
  %5 = load i32, i32 addrspace(1)* undef, align 4
  %6 = sub i32 undef, %5
  %factor.us.2 = shl i32 %6, 1
  %7 = add i32 %factor.us.2, %1
  %8 = load i32, i32 addrspace(1)* undef, align 4
  %9 = sub i32 %7, %8
  %factor.us.3 = shl i32 %9, 1
  %10 = add i32 %factor.us.3, %4
  %11 = load i32, i32 addrspace(1)* undef, align 4
  %12 = sub i32 %10, %11
  %factor.us.4 = shl i32 %12, 1
  %13 = add i32 %0, 8
  %indvars.iv.next.us.7 = add nsw i64 %indvars.iv.us, 8
  br label %not_zero48.us
}

