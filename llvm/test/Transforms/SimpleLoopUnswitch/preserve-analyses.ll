; RUN: opt -simple-loop-unswitch -verify-loop-info -verify-dom-info -verify-memoryssa -disable-output < %s

; Loop unswitch should be able to unswitch these loops and
; preserve LCSSA and LoopSimplify forms.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv6-apple-darwin9"

@delim1 = external global i32                     ; <i32*> [#uses=1]
@delim2 = external global i32                     ; <i32*> [#uses=1]

define i32 @ineqn(i8* %s, i8* %p) nounwind readonly {
entry:
  %0 = load i32, i32* @delim1, align 4                 ; <i32> [#uses=1]
  %1 = load i32, i32* @delim2, align 4                 ; <i32> [#uses=1]
  br label %bb8.outer

bb:                                               ; preds = %bb8
  %2 = icmp eq i8* %p_addr.0, %s                  ; <i1> [#uses=1]
  br i1 %2, label %bb10, label %bb2

bb2:                                              ; preds = %bb
  %3 = getelementptr inbounds i8, i8* %p_addr.0, i32 1 ; <i8*> [#uses=3]
  switch i32 %ineq.0.ph, label %bb8.backedge [
    i32 0, label %bb3
    i32 1, label %bb6
  ]

bb8.backedge:                                     ; preds = %bb6, %bb5, %bb2
  br label %bb8

bb3:                                              ; preds = %bb2
  %4 = icmp eq i32 %8, %0                         ; <i1> [#uses=1]
  br i1 %4, label %bb8.outer.loopexit, label %bb5

bb5:                                              ; preds = %bb3
  br i1 %6, label %bb6, label %bb8.backedge

bb6:                                              ; preds = %bb5, %bb2
  %5 = icmp eq i32 %8, %1                         ; <i1> [#uses=1]
  br i1 %5, label %bb7, label %bb8.backedge

bb7:                                              ; preds = %bb6
  %.lcssa1 = phi i8* [ %3, %bb6 ]                 ; <i8*> [#uses=1]
  br label %bb8.outer.backedge

bb8.outer.backedge:                               ; preds = %bb8.outer.loopexit, %bb7
  %.lcssa2 = phi i8* [ %.lcssa1, %bb7 ], [ %.lcssa, %bb8.outer.loopexit ] ; <i8*> [#uses=1]
  %ineq.0.ph.be = phi i32 [ 0, %bb7 ], [ 1, %bb8.outer.loopexit ] ; <i32> [#uses=1]
  br label %bb8.outer

bb8.outer.loopexit:                               ; preds = %bb3
  %.lcssa = phi i8* [ %3, %bb3 ]                  ; <i8*> [#uses=1]
  br label %bb8.outer.backedge

bb8.outer:                                        ; preds = %bb8.outer.backedge, %entry
  %ineq.0.ph = phi i32 [ 0, %entry ], [ %ineq.0.ph.be, %bb8.outer.backedge ] ; <i32> [#uses=3]
  %p_addr.0.ph = phi i8* [ %p, %entry ], [ %.lcssa2, %bb8.outer.backedge ] ; <i8*> [#uses=1]
  %6 = icmp eq i32 %ineq.0.ph, 1                  ; <i1> [#uses=1]
  br label %bb8

bb8:                                              ; preds = %bb8.outer, %bb8.backedge
  %p_addr.0 = phi i8* [ %p_addr.0.ph, %bb8.outer ], [ %3, %bb8.backedge ] ; <i8*> [#uses=3]
  %7 = load i8, i8* %p_addr.0, align 1                ; <i8> [#uses=2]
  %8 = sext i8 %7 to i32                          ; <i32> [#uses=2]
  %9 = icmp eq i8 %7, 0                           ; <i1> [#uses=1]
  br i1 %9, label %bb10, label %bb

bb10:                                             ; preds = %bb8, %bb
  %.0 = phi i32 [ %ineq.0.ph, %bb ], [ 0, %bb8 ]  ; <i32> [#uses=1]
  ret i32 %.0
}

; This is a simplified form of ineqn from above. It triggers some
; different cases in the loop-unswitch code.

define void @simplified_ineqn() nounwind readonly {
entry:
  br label %bb8.outer

bb8.outer:                                        ; preds = %bb6, %bb2, %entry
  %x = phi i32 [ 0, %entry ], [ 0, %bb6 ], [ 1, %bb2 ] ; <i32> [#uses=1]
  br i1 undef, label %return, label %bb2

bb2:                                              ; preds = %bb
  switch i32 %x, label %bb6 [
    i32 0, label %bb8.outer
  ]

bb6:                                              ; preds = %bb2
  br i1 undef, label %bb8.outer, label %bb2

return:                                             ; preds = %bb8, %bb
  ret void
}

; This function requires special handling to preserve LCSSA form.
; PR4934

define void @pnp_check_irq() nounwind noredzone {
entry:
  %conv56 = trunc i64 undef to i32                ; <i32> [#uses=1]
  br label %while.cond.i

while.cond.i:                                     ; preds = %while.cond.i.backedge, %entry
  %call.i25 = call i8* @pci_get_device() nounwind noredzone ; <i8*> [#uses=2]
  br i1 undef, label %if.then65, label %while.body.i

while.body.i:                                     ; preds = %while.cond.i
  br i1 undef, label %if.then31.i.i, label %while.cond.i.backedge

while.cond.i.backedge:                            ; preds = %if.then31.i.i, %while.body.i
  br label %while.cond.i

if.then31.i.i:                                    ; preds = %while.body.i
  switch i32 %conv56, label %while.cond.i.backedge [
    i32 14, label %if.then42.i.i
    i32 15, label %if.then42.i.i
  ]

if.then42.i.i:                                    ; preds = %if.then31.i.i, %if.then31.i.i
  %call.i25.lcssa48 = phi i8* [ %call.i25, %if.then31.i.i ], [ %call.i25, %if.then31.i.i ] ; <i8*> [#uses=0]
  unreachable

if.then65:                                        ; preds = %while.cond.i
  unreachable
}

declare i8* @pci_get_device() noredzone
