; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z10 -verify-machineinstrs -systemz-subreg-liveness < %s | FileCheck %s

; Check for successful compilation.
; CHECK: lgfrl %r1, g_65

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

@g_65 = external dso_local global i32, align 4

; Function Attrs: nounwind
define void @main(i1 %x) #0 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  %tmp = load i32, ptr @g_65, align 4
  %tmp2 = sext i32 %tmp to i64
  %tmp3 = shl i32 %tmp, 16
  %tmp4 = ashr exact i32 %tmp3, 16
  %tmp5 = shl i32 %tmp4, 0
  %tmp6 = zext i32 %tmp5 to i64
  %tmp7 = shl i64 %tmp6, 48
  %tmp8 = ashr exact i64 %tmp7, 48
  br i1 undef, label %bb12, label %bb9

bb9:                                              ; preds = %bb1
  %tmp10 = select i1 %x, i64 0, i64 %tmp2
  %tmp11 = add nsw i64 %tmp10, %tmp8
  br label %bb12

bb12:                                             ; preds = %bb9, %bb1
  %tmp13 = phi i64 [ %tmp11, %bb9 ], [ %tmp8, %bb1 ]
  %tmp14 = trunc i64 %tmp13 to i32
  %tmp15 = and i32 %tmp14, 255
  %tmp16 = shl i32 %tmp15, 0
  %tmp17 = trunc i32 %tmp16 to i8
  %tmp18 = icmp eq i8 %tmp17, 0
  br i1 %tmp18, label %bb20, label %bb19

bb19:                                             ; preds = %bb12
  unreachable

bb20:                                             ; preds = %bb12
  unreachable
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="z10" "unsafe-fp-math"="false" "use-soft-float"="false" }
