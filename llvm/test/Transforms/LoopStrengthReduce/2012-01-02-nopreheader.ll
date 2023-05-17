; RUN: opt -loop-reduce -S < %s | FileCheck %s
;
; <rdar://10619599> "SelectionDAGBuilder shouldn't visit PHI nodes!" assert.
; <rdar://10655343> SCEVExpander segfault on simple test case

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-f128:128:128-n8:16:32"
target triple = "i386-apple-darwin"

; LSR should convert the inner loop (bb7.us) IV (j.01.us) into float*.
; This involves a nested AddRec, the outer AddRec's loop invariant components
; cannot find a preheader, so they should be expanded in the loop header
; (bb7.lr.ph.us) below the existing phi i.12.us.
; Currently, LSR won't kick in on such loops.
; CHECK-LABEL: @nopreheader(
; CHECK: bb7.us:
; CHECK-NOT: phi ptr
; CHECK: %j.01.us = phi i32
; CHECK-NOT: phi ptr
define void @nopreheader(ptr nocapture %a, i32 %n) nounwind {
entry:
  %0 = sdiv i32 %n, undef
  indirectbr ptr undef, [label %bb10.preheader]

bb10.preheader:                                   ; preds = %bb4
  indirectbr ptr undef, [label %bb8.preheader.lr.ph, label %return]

bb8.preheader.lr.ph:                              ; preds = %bb10.preheader
  indirectbr ptr null, [label %bb7.lr.ph.us, label %bb9]

bb7.lr.ph.us:                                     ; preds = %bb9.us, %bb8.preheader.lr.ph
  %i.12.us = phi i32 [ %2, %bb9.us ], [ 0, %bb8.preheader.lr.ph ]
  %tmp30 = mul i32 %0, %i.12.us
  indirectbr ptr undef, [label %bb7.us]

bb7.us:                                           ; preds = %bb7.lr.ph.us, %bb7.us
  %j.01.us = phi i32 [ 0, %bb7.lr.ph.us ], [ %1, %bb7.us ]
  %tmp31 = add i32 %tmp30, %j.01.us
  %scevgep9 = getelementptr float, ptr %a, i32 %tmp31
  store float undef, ptr %scevgep9, align 1
  %1 = add nsw i32 %j.01.us, 1
  indirectbr ptr undef, [label %bb9.us, label %bb7.us]

bb9.us:                                           ; preds = %bb7.us
  %2 = add nsw i32 %i.12.us, 1
  indirectbr ptr undef, [label %bb7.lr.ph.us, label %return]

bb9:                                              ; preds = %bb9, %bb8.preheader.lr.ph
  indirectbr ptr undef, [label %bb9, label %return]

return:                                           ; preds = %bb9, %bb9.us, %bb10.preheader
  ret void
}

; In this case, SCEVExpander simply cannot materialize the AddRecExpr
; that LSR picks. We must detect that %bb8.preheader does not have a
; preheader and avoid performing LSR on %bb7.
; CHECK-LABEL: @nopreheader2(
; CHECK: bb7:
; CHECK: %indvar = phi i32
define fastcc void @nopreheader2(ptr nocapture %Array2) nounwind {
entry:
  indirectbr ptr undef, [label %bb]

bb:                                               ; preds = %bb, %entry
  indirectbr ptr undef, [label %bb3, label %bb]

bb3:                                              ; preds = %bb3, %bb
  indirectbr ptr undef, [label %bb8.preheader, label %bb3]

bb8.preheader:                                    ; preds = %bb9, %bb3
  %indvar5 = phi i32 [ %indvar.next6, %bb9 ], [ 0, %bb3 ]
  %tmp26 = add i32 %indvar5, 13
  indirectbr ptr null, [label %bb7]

bb7:                                              ; preds = %bb8.preheader, %bb7
  %indvar = phi i32 [ 0, %bb8.preheader ], [ %indvar.next, %bb7 ]
  %scevgep = getelementptr [200 x i32], ptr %Array2, i32 %tmp26, i32 %indvar
  store i32 undef, ptr %scevgep, align 4
  %indvar.next = add i32 %indvar, 1
  indirectbr ptr undef, [label %bb9, label %bb7]

bb9:                                              ; preds = %bb7
  %indvar.next6 = add i32 %indvar5, 1
  indirectbr ptr undef, [label %return, label %bb8.preheader]

return:                                           ; preds = %bb9
  ret void
}
