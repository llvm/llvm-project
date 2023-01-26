; RUN: opt -verify-memoryssa -passes='loop-mssa(loop-rotate)' -S %s | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

; Check verification passes after loop rotate, when adding phis in blocks
; receiving incoming edges and adding phis in IDF blocks.
; CHECK-LABEL: @f
define void @f() align 32 {
entry:
  br label %while.cond.outer

while.cond80.while.cond.loopexit_crit_edge:       ; preds = %if.else99
  br label %while.cond.outer

while.cond.outer:                                 ; preds = %while.cond80.while.cond.loopexit_crit_edge, %entry
  br i1 undef, label %while.cond.outer.return.loopexit2_crit_edge, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %while.cond.outer
  br label %while.body

while.body:                                       ; preds = %while.body.lr.ph
  br i1 undef, label %if.then42, label %if.end61

if.then42:                                        ; preds = %while.body
  br label %return.loopexit2

if.end61:                                         ; preds = %while.body
  br label %while.body82

while.body82:                                     ; preds = %if.end61
  br i1 undef, label %return.loopexit, label %if.else99

if.else99:                                        ; preds = %while.body82
  store i32 0, ptr inttoptr (i64 44 to ptr), align 4
  br label %while.cond80.while.cond.loopexit_crit_edge

return.loopexit:                                  ; preds = %while.body82
  br label %return

while.cond.outer.return.loopexit2_crit_edge:      ; preds = %while.cond.outer
  br label %return.loopexit2

return.loopexit2:                                 ; preds = %while.cond.outer.return.loopexit2_crit_edge, %if.then42
  br label %return

return:                                           ; preds = %return.loopexit2, %return.loopexit
  ret void
}

