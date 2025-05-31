; Test that SystemZISelLowering::supportedAddressingMode() does not crash on
; a comparison of constant wider than 64 bits.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

%0 = type <{ i64, [11 x i8] }>

@g_101 = external dso_local global <{ <{ i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }>, <{ i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }>, <{ i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }>, <{ i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }>, <{ i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }>, <{ i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }>, <{ i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }>, <{ i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }>, <{ i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }>, <{ i64, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }> }>, align 2

; Function Attrs: nounwind
define void @main() local_unnamed_addr #0 {
  br label %1

; <label>:1:                                      ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %7, %1 ]
  %3 = getelementptr inbounds [10 x %0], ptr @g_101, i64 0, i64 %2, i32 1
  %4 = bitcast ptr %3 to ptr
  %5 = load i88, ptr %4, align 1
  %6 = icmp ult i88 %5, 2361183241434822606848
  %7 = add nuw nsw i64 %2, 1
  br label %1
}
