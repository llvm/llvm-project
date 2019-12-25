; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -disable-early-taildup -disable-cgp -systemz-subreg-liveness < %s | FileCheck %s

; Check for successful compilation.
; CHECK: lhi %r0, -5

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

; Function Attrs: nounwind
define void @main() #0 {
bb:
  %tmp = xor i8 0, -5
  %tmp1 = sext i8 %tmp to i32
  %tmp2 = icmp sgt i8 0, -1
  br label %bb3

bb3:                                              ; preds = %bb15, %bb
  %tmp4 = phi i64 [ %tmp16, %bb15 ], [ -1, %bb ]
  br i1 undef, label %bb14, label %bb5

bb5:                                              ; preds = %bb3
  %tmp6 = or i1 %tmp2, false
  %tmp7 = select i1 %tmp6, i32 0, i32 100
  %tmp8 = ashr i32 %tmp1, %tmp7
  %tmp9 = zext i32 %tmp8 to i64
  %tmp10 = shl i64 %tmp9, 48
  %tmp11 = ashr exact i64 %tmp10, 48
  %tmp12 = and i64 %tmp11, %tmp4
  %tmp13 = trunc i64 %tmp12 to i32
  store i32 %tmp13, i32* undef, align 4
  br label %bb15

bb14:                                             ; preds = %bb3
  br label %bb15

bb15:                                             ; preds = %bb14, %bb5
  %tmp16 = phi i64 [ %tmp4, %bb14 ], [ %tmp12, %bb5 ]
  br label %bb3
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="z13" "target-features"="+transactional-execution,+vector" "unsafe-fp-math"="false" "use-soft-float"="false" }
