; RUN: opt -S -simplifycfg < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32* @fn1(i32* returned)

define i32 @test1(i1 %B) {
entry:
  br label %for.cond.us

for.cond.us:                                      ; preds = %for.cond.us, %entry
  br i1 %B, label %for.cond4.preheader, label %for.cond.us

for.cond4.preheader:                              ; preds = %for.cond.us
  br i1 %B, label %for.cond4.preheader.split.us, label %for.cond4

for.cond4.preheader.split.us:                     ; preds = %for.cond4.preheader
  unreachable

for.cond4:                                        ; preds = %for.end, %for.cond4.preheader
  %phi = phi i32* [ %call, %for.end ], [ undef, %for.cond4.preheader ]
  %call = call i32* @fn1(i32* %phi)
  br label %for.cond5

for.cond5:                                        ; preds = %for.cond5, %for.cond4
  br i1 %B, label %for.cond5, label %for.end

for.end:                                          ; preds = %for.cond5
  %load = load i32, i32* %call, align 4
  br label %for.cond4
}

; CHECK-LABEL: define i32 @test1(
; CHECK: br label %[[LABEL:.*]]
; CHECK: [[LABEL]]:
; CHECK: br label %[[LABEL]]
