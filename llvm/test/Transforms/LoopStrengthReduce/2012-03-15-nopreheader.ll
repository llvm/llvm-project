; RUN: opt -loop-reduce -S < %s | FileCheck %s
;
; <rdar://problem/11049788> Segmentation fault: 11 in LoopStrengthReduce

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; IVUsers should not consider tmp128 a valid user because it is not in a
; simplified loop nest.
; CHECK-LABEL: @nopreheader(
; CHECK: for.cond:
; CHECK: %tmp128 = add i64 %0, %indvar65
define void @nopreheader(i8* %cmd) nounwind ssp {
entry:
  indirectbr i8* undef, [label %while.cond]

while.cond:                                       ; preds = %while.body, %entry
  %0 = phi i64 [ %indvar.next48, %while.body ], [ 0, %entry ]
  indirectbr i8* undef, [label %while.end, label %while.body]

while.body:                                       ; preds = %lor.rhs, %lor.lhs.false17, %lor.lhs.false11, %lor.lhs.false, %land.rhs
  %indvar.next48 = add i64 %0, 1
  indirectbr i8* undef, [label %while.cond]

while.end:                                        ; preds = %lor.rhs, %while.cond
  indirectbr i8* undef, [label %if.end152]

if.end152:                                        ; preds = %lor.lhs.false144, %if.end110
  indirectbr i8* undef, [label %lor.lhs.false184, label %for.cond]

lor.lhs.false184:                                 ; preds = %lor.lhs.false177
  indirectbr i8* undef, [label %return, label %for.cond]

for.cond:                                         ; preds = %for.inc, %lor.lhs.false184, %if.end152
  %indvar65 = phi i64 [ %indvar.next66, %for.inc ], [ 0, %lor.lhs.false184 ], [ 0, %if.end152 ]
  %tmp128 = add i64 %0, %indvar65
  %s.4 = getelementptr i8, i8* %cmd, i64 %tmp128
  %tmp195 = load i8, i8* %s.4, align 1
  indirectbr i8* undef, [label %return, label %land.rhs198]

land.rhs198:                                      ; preds = %for.cond
  indirectbr i8* undef, [label %return, label %for.inc]

for.inc:                                          ; preds = %lor.rhs234, %land.lhs.true228, %land.lhs.true216, %land.lhs.true204
  %indvar.next66 = add i64 %indvar65, 1
  indirectbr i8* undef, [label %for.cond]

return:                                           ; preds = %if.end677, %doshell, %if.then96
  ret void
}

; Another case with a dominating loop that does not contain the IV
; User. Just make sure it doesn't assert.
define void @nopreheader2() nounwind ssp {
entry:
  indirectbr i8* undef, [label %while.cond, label %return]

while.cond:                                       ; preds = %while.cond.backedge, %entry
  indirectbr i8* undef, [label %while.cond.backedge, label %lor.rhs]

lor.rhs:                                          ; preds = %while.cond
  indirectbr i8* undef, [label %while.cond.backedge, label %while.end]

while.cond.backedge:                              ; preds = %lor.rhs, %while.cond
  indirectbr i8* undef, [label %while.cond]

while.end:                                        ; preds = %lor.rhs
  indirectbr i8* undef, [label %if.then18, label %return]

if.then18:                                        ; preds = %while.end
  indirectbr i8* undef, [label %if.end35, label %lor.lhs.false]

lor.lhs.false:                                    ; preds = %if.then18
  indirectbr i8* undef, [label %if.end35, label %return]

if.end35:                                         ; preds = %lor.lhs.false, %if.then18
  indirectbr i8* undef, [label %while.cond36]

while.cond36:                                     ; preds = %while.body49, %if.end35
  %0 = phi i64 [ %indvar.next13, %while.body49 ], [ 0, %if.end35 ]
  indirectbr i8* undef, [label %while.body49, label %lor.rhs42]

lor.rhs42:                                        ; preds = %while.cond36
  indirectbr i8* undef, [label %while.body49, label %while.end52]

while.body49:                                     ; preds = %lor.rhs42, %while.cond36
  %indvar.next13 = add i64 %0, 1
  indirectbr i8* undef, [label %while.cond36]

while.end52:                                      ; preds = %lor.rhs42
  indirectbr i8* undef, [label %land.lhs.true, label %return]

land.lhs.true:                                    ; preds = %while.end52
  indirectbr i8* undef, [label %while.cond66.preheader, label %return]

while.cond66.preheader:                           ; preds = %land.lhs.true
  indirectbr i8* undef, [label %while.cond66]

while.cond66:                                     ; preds = %while.body77, %while.cond66.preheader
  indirectbr i8* undef, [label %land.rhs, label %while.cond81.preheader]

land.rhs:                                         ; preds = %while.cond66
  indirectbr i8* undef, [label %while.body77, label %while.cond81.preheader]

while.cond81.preheader:                           ; preds = %land.rhs, %while.cond66
  %tmp45 = add i64 undef, %0
  %tmp46 = add i64 %tmp45, undef
  indirectbr i8* undef, [label %while.cond81]

while.body77:                                     ; preds = %land.rhs
  indirectbr i8* undef, [label %while.cond66]

while.cond81:                                     ; preds = %while.body94, %while.cond81.preheader
  %tmp25 = add i64 %tmp46, undef
  indirectbr i8* undef, [label %while.body94, label %lor.rhs87]

lor.rhs87:                                        ; preds = %while.cond81
  indirectbr i8* undef, [label %while.body94, label %return]

while.body94:                                     ; preds = %lor.rhs87, %while.cond81
  indirectbr i8* undef, [label %while.cond81]

return:                                           ; preds = %if.end216, %land.lhs.true183, %land.lhs.true, %while.end52, %lor.lhs.false, %while.end, %entry
  ret void
}

; Test a phi operand IV User dominated by a no-preheader loop.
define void @nopreheader3() nounwind uwtable ssp align 2 {
entry:
  indirectbr i8* blockaddress(@nopreheader3, %if.end10), [label %if.end22, label %if.end10]

if.end10:                                         ; preds = %entry
  indirectbr i8* blockaddress(@nopreheader3, %if.end6.i), [label %if.end22, label %if.end6.i]

if.end6.i:                                        ; preds = %if.end10
  indirectbr i8* blockaddress(@nopreheader3, %while.cond2.preheader.i.i), [label %if.then12, label %while.cond2.preheader.i.i]

while.cond2.preheader.i.i:                        ; preds = %while.end.i18.i, %if.end6.i
  indirectbr i8* blockaddress(@nopreheader3, %while.cond2.i.i), [label %while.cond2.i.i]

while.cond2.i.i:                                  ; preds = %while.cond2.i.i, %while.cond2.preheader.i.i
  %i1.1.i14.i = phi i32 [ %add.i15.i, %while.cond2.i.i ], [ undef, %while.cond2.preheader.i.i ]
  %add.i15.i = add nsw i32 %i1.1.i14.i, undef
  indirectbr i8* blockaddress(@nopreheader3, %while.end.i18.i), [label %while.cond2.i.i, label %while.end.i18.i]

while.end.i18.i:                                  ; preds = %while.cond2.i.i
  indirectbr i8* blockaddress(@nopreheader3, %while.cond2.preheader.i.i), [label %if.then12, label %while.cond2.preheader.i.i]

if.then12:                                        ; preds = %while.end.i18.i, %if.end6.i
  %i1.0.lcssa.i.i = phi i32 [ undef, %if.end6.i ], [ %i1.1.i14.i, %while.end.i18.i ]
  indirectbr i8* blockaddress(@nopreheader3, %if.end22), [label %if.end22]

if.end22:                                         ; preds = %if.then12, %if.end10, %entry
  ret void
}
