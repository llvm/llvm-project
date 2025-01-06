; RUN: opt < %s -passes=adce --preserve-ll-uselistorder -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; CHECK: uselistorder label %bb16, { 1, 0 }
; Function Attrs: noinline nounwind ssp uwtable
define void @ham(i1 %arg) local_unnamed_addr #0 {
bb:
  br i1 false, label %bb1, label %bb22

bb1:                                              ; preds = %bb
  br i1 %arg, label %bb2, label %bb20

bb2:                                              ; preds = %bb1
  br label %bb5

bb5:                                              ; preds = %bb16, %bb2
  br i1 %arg, label %bb6, label %bb17

bb6:                                              ; preds = %bb5
  br i1 %arg, label %bb7, label %bb16

bb7:                                              ; preds = %bb6
  br i1 %arg, label %bb9, label %bb8

bb8:                                              ; preds = %bb7
  br i1 %arg, label %bb9, label %bb10

bb9:                                              ; preds = %bb8, %bb7
  br label %bb13

bb10:                                             ; preds = %bb8
  br label %bb12

bb12:                                             ; preds = %bb10
  br label %bb13

bb13:                                             ; preds = %bb12, %bb9
  br label %bb14

bb14:                                             ; preds = %bb13
  br label %bb15

bb15:                                             ; preds = %bb14
  br label %bb16

bb16:                                             ; preds = %bb15, %bb6
  br label %bb5

bb17:                                             ; preds = %bb5
  br label %bb19

bb19:                                             ; preds = %bb17
  br label %bb21

bb20:                                             ; preds = %bb1
  br label %bb21

bb21:                                             ; preds = %bb20, %bb19
  br label %bb22

bb22:                                             ; preds = %bb21, %bb
  ret void
}

attributes #0 = { noinline nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"PIC Level", i32 2}
