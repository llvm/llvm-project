; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s

; CHECK: Valid Region for Scop: bb10 => bb16

; Verify that -polly-scops does not crash. At some point this piece of
; code crashed as we extracted from the SCEV expression:
;
;    ((8 * ((%a * %b) + %c)) + (-8 * %a))'
;
; the constant 8, which resulted in the new expression:
;
;    (((-1 + %b) * %a) + %c)
;
; which introduced a new parameter (-1 + %b) * %a which was not registered
; correctly and consequently caused a crash due to an expression not being
; registered as a parameter.

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

define void @barney(ptr %arg, ptr %arg1, ptr %arg2, ptr %arg3, i32 %a, i32 %b, i32 %c) {
bb:
  br label %bb10

bb10:                                             ; preds = %bb
  br i1 true, label %bb11, label %bb16

bb11:                                             ; preds = %bb10
  %tmp4 = add nsw i32 1, %a
  %tmp5 = sub i32 0, %tmp4
  %tmp8 = add nsw i32 %c, 1
  %tmp12 = mul nsw i32 %b, %a
  %tmp13 = add nsw i32 %tmp8, %tmp12
  %tmp6 = getelementptr inbounds double, ptr %arg2, i32 %tmp5
  %tmp14 = getelementptr inbounds double, ptr %tmp6, i32 %tmp13
  %tmp15 = load double, ptr %tmp14
  br label %bb16

bb16:                                             ; preds = %bb11, %bb10
  ret void
}
