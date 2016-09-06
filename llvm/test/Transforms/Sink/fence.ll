; RUN: opt -S -sink < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1(i32* ()*) {
entry:
  %1 = call i32* %0() #0
  fence singlethread seq_cst
  %2 = load i32, i32* %1, align 4
  fence singlethread seq_cst
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %fail, label %pass

fail:                                             ; preds = %top
  br label %pass

pass:                                             ; preds = %fail, %top
  ret void
}

; CHECK-LABEL: @test1(
; CHECK:  %[[call:.*]] = call i32* %0()
; CHECK:  fence singlethread seq_cst
; CHECK:  load i32, i32* %[[call]], align 4
; CHECK:  fence singlethread seq_cst


attributes #0 = { nounwind readnone }
