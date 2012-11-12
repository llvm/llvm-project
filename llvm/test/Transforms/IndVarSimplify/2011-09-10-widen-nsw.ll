; RUN: opt < %s -indvars -S | FileCheck %s
; Test WidenIV::GetExtendedOperandRecurrence.
; add219 should be extended to i64 because it is nsw, even though its
; sext cannot be hoisted outside the loop.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @test() nounwind {
entry:
  br i1 undef, label %for.body11, label %for.end285

for.body11:                                       ; preds = %entry
  %shl = shl i32 1, 1
  %shl132 = shl i32 %shl, 1
  br label %for.body153

for.body153:                                      ; preds = %for.body153, %for.body11
  br i1 undef, label %for.body170, label %for.body153

; CHECK: add nsw i64 %indvars.iv, 1
for.body170:                                      ; preds = %for.body170, %for.body153
  %i2.19 = phi i32 [ %add249, %for.body170 ], [ 0, %for.body153 ]
  %add219 = add nsw i32 %i2.19, 1
  %idxprom220 = sext i32 %add219 to i64
  %add249 = add nsw i32 %i2.19, %shl132
  br label %for.body170

for.end285:                                       ; preds = %entry
  ret void
}
