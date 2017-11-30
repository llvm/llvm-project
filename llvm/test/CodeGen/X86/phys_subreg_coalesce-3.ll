; RUN: llc < %s -verify-machineinstrs -mtriple=i386-apple-darwin -mcpu=corei7 | FileCheck %s
; rdar://5571034

; This requires physreg joining, %13 is live everywhere:
; 304L		%cl<def> = COPY %13:sub_8bit; GR32_ABCD:%13
; 320L		%15<def> = COPY %19; GR32:%15 GR32_NOSP:%19
; 336L		%15<def> = SAR32rCL %15, %eflags<imp-def,dead>, %cl<imp-use,kill>; GR32:%15

define void @foo(i32* nocapture %quadrant, i32* nocapture %ptr, i32 %bbSize, i32 %bbStart, i32 %shifts) nounwind ssp {
; CHECK-LABEL: foo:
entry:
  %j.03 = add i32 %bbSize, -1                     ; <i32> [#uses=2]
  %0 = icmp sgt i32 %j.03, -1                     ; <i1> [#uses=1]
  br i1 %0, label %bb.nph, label %return

bb.nph:                                           ; preds = %entry
  %tmp9 = add i32 %bbStart, %bbSize               ; <i32> [#uses=1]
  %tmp10 = add i32 %tmp9, -1                      ; <i32> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
; CHECK: %bb
; CHECK-NOT: movb {{.*}}l, %cl
; CHECK: sarl %cl
  %indvar = phi i32 [ 0, %bb.nph ], [ %indvar.next, %bb ] ; <i32> [#uses=3]
  %j.06 = sub i32 %j.03, %indvar                  ; <i32> [#uses=1]
  %tmp11 = sub i32 %tmp10, %indvar                ; <i32> [#uses=1]
  %scevgep = getelementptr i32, i32* %ptr, i32 %tmp11  ; <i32*> [#uses=1]
  %1 = load i32, i32* %scevgep, align 4                ; <i32> [#uses=1]
  %2 = ashr i32 %j.06, %shifts                    ; <i32> [#uses=1]
  %3 = and i32 %2, 65535                          ; <i32> [#uses=1]
  %4 = getelementptr inbounds i32, i32* %quadrant, i32 %1 ; <i32*> [#uses=1]
  store i32 %3, i32* %4, align 4
  %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %indvar.next, %bbSize   ; <i1> [#uses=1]
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}
