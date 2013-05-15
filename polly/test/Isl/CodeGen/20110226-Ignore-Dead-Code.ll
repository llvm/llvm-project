; RUN: opt %loadPolly %defaultOpts -polly-codegen-isl %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @main() nounwind {
.split:
  br label %0

.loopexit.loopexit:                               ; preds = %.preheader.us
  br label %.loopexit.simregexit

.loopexit.simregexit:                             ; preds = %.loopexit.loopexit, %._crit_edge
  br label %.loopexit

.loopexit:                                        ; preds = %.loopexit.simregexit
  %indvar.next16 = add i64 %indvar15, 1
  %exitcond53 = icmp eq i64 %indvar.next16, 2048
  br i1 %exitcond53, label %2, label %0

; <label>:0                                       ; preds = %.loopexit, %.split
  %indvar15 = phi i64 [ 0, %.split ], [ %indvar.next16, %.loopexit ]
  br label %.simregentry

.simregentry:                                     ; preds = %0
  %indvar15.ph = phi i64 [ %indvar15, %0 ]
  %tmp67 = add i64 %indvar15, 1
  %i.06 = trunc i64 %tmp67 to i32
  %tmp25 = add i64 undef, 1
  %1 = icmp slt i32 %i.06, 2048
  br i1 %1, label %.lr.ph.preheader, label %._crit_edge.simregexit

.lr.ph.preheader:                                 ; preds = %.simregentry
  br label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph, %.lr.ph.preheader
  %indvar33 = phi i64 [ %indvar.next34, %.lr.ph ], [ 0, %.lr.ph.preheader ]
  %indvar.next34 = add i64 %indvar33, 1
  %exitcond40 = icmp eq i64 %indvar.next34, 0
  br i1 %exitcond40, label %._crit_edge.loopexit, label %.lr.ph

._crit_edge.loopexit:                             ; preds = %.lr.ph
  br label %._crit_edge.simregexit

._crit_edge.simregexit:                           ; preds = %.simregentry, %._crit_edge.loopexit
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.simregexit
  br i1 false, label %.loopexit.simregexit, label %.preheader.us.preheader

.preheader.us.preheader:                          ; preds = %._crit_edge
  br label %.preheader.us

.preheader.us:                                    ; preds = %.preheader.us, %.preheader.us.preheader
  %exitcond26.old = icmp eq i64 undef, %tmp25
  br i1 false, label %.loopexit.loopexit, label %.preheader.us

; <label>:2                                       ; preds = %.loopexit
  ret void
}
