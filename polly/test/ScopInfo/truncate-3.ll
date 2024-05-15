; RUN: opt %loadPolly -polly-scops -pass-remarks-analysis="polly-scops" \
; RUN:                -disable-output < %s 2>&1 | FileCheck %s

; CHECK: Signed-unsigned restriction: [p] -> {  : p <= -129 or p >= 128 }

; Verify that this test case does not crash when we try to model it.
; At some point we tried to insert a restriction:
;                                      [p] -> {  : p <= -129 or p >= 128 }
; which resulted in a crash.

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

define void @wobble(ptr %A, i32 %p) {
bb:
  %tmp1 = and i32 %p, 255
  br label %bb4

bb4:                                              ; preds = %bb4, %bb
  %indvar = phi ptr [ %A, %bb ], [ %indvar.next, %bb4 ]
  %val = load i16, ptr %indvar
  %indvar.next = getelementptr inbounds i16, ptr %indvar, i32 %tmp1
  br i1 false, label %bb4, label %bb9

bb9:                                              ; preds = %bb4
  ret void
}
