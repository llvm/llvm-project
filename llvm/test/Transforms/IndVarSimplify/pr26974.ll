; RUN: opt -passes=indvars  -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; indvars will try to replace %b.0.lcssa with %t.1. If it does this,
; it will break LCSSA.

@c = external global i32, align 4

; CHECK-LABEL: @fn1
define void @fn1(i1 %arg) {
entry:
  br label %for.body

for.cond1.preheader:                              ; preds = %for.body
  %0 = load i32, ptr @c, align 4
  br i1 %arg, label %for.cond1.us.preheader, label %for.cond1

for.cond1.us.preheader:                           ; preds = %for.cond1.preheader
  br label %for.cond1.us

for.cond1.us:                                     ; preds = %for.cond1.us, %for.cond1.us.preheader
  br label %for.cond1.us

for.body:                                         ; preds = %for.body, %entry
  br i1 %arg, label %for.body, label %for.cond1.preheader

for.cond1:                                        ; preds = %for.cond1.preheader
  br i1 true, label %for.body9.lr.ph, label %for.cond13.preheader

for.body9.lr.ph:                                  ; preds = %for.cond1
  br i1 %arg, label %for.body9.us.preheader, label %for.body9

for.body9.us.preheader:                           ; preds = %for.body9.lr.ph
  br label %for.body9.us

for.body9.us:                                     ; preds = %for.body9.us, %for.body9.us.preheader
  br label %for.body9.us

for.cond13.preheader:                             ; preds = %for.body9, %for.cond1
  %b.0.lcssa = phi i32 [ %0, %for.body9 ], [ 0, %for.cond1 ]
  br label %for.cond13

for.body9:                                        ; preds = %for.body9.lr.ph
  br label %for.cond13.preheader

for.cond13:                                       ; preds = %for.cond13, %for.cond13.preheader
  %d.1 = phi i32 [ %t.1, %for.cond13 ], [ %0, %for.cond13.preheader ]
  %t.1 = phi i32 [ %b.0.lcssa, %for.cond13 ], [ %0, %for.cond13.preheader ]
  br i1 %arg, label %for.cond18.preheader, label %for.cond13

for.cond18.preheader:                             ; preds = %for.cond13
  br label %for.cond18

for.cond18:                                       ; preds = %for.cond18, %for.cond18.preheader
  %b.1 = phi i32 [ %xor, %for.cond18 ], [ %b.0.lcssa, %for.cond18.preheader ]
  %add = add nsw i32 %b.1, %d.1
  %xor = xor i32 %add, %b.1
  br label %for.cond18
}
