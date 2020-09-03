;; Test that we skip unconditional PredBB when threading jumps through two
;; successive basic blocks.
; RUN: opt -S -passes='function(jump-threading)' < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f(i32* %0) {
; CHECK-LABEL: @f
; CHECK: br i1 %good, label %pred.bb, label %pred.pred.bb
entry:
  %size = call i64 @get_size(i32* %0)
  %good = icmp ugt i64 %size, 3
  br i1 %good, label %pred.bb, label %pred.pred.bb

; CHECK:      pred.pred.bb:
; CHECK:       br label %pred.bb
; CHECK:      pred.bb:
; CHECK:       br label %bb
; CHECK:      bb:
pred.pred.bb:                                        ; preds = %entry
  call void @effect()
  br label %pred.bb
pred.bb:                                             ; preds = %pred.pred.bb, %entry
  %v = load i32, i32* %0
  br label %bb

bb:                                                  ; preds = %pred.bb
  call void @effect1(i8* blockaddress(@f, %bb))
  br i1 %good, label %cont2, label %cont1

cont1:                                               ; preds = %bb
  br i1 %good, label %exit, label %cont2
cont2:                                               ; preds = %bb
  br label %exit
exit:                                                ; preds = %cont1, %cont2
  ret i32 %v
}

declare i64 @get_size(i32*)
declare void @effect()
declare void @effect1(i8*)
