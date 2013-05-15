; RUN: opt %loadPolly -polly-codegen-isl -polly-codegen-scev %s
; -polly-independent causes: Cannot generate independent blocks
;
; XFAIL:*
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @main() nounwind {
.split:
  br label %.start

.start:
  %indvar15 = phi i64 [ 0, %.split ], [ %indvar.next16, %.loopexit ]
  %tmp25 = add i64 %indvar15, 1
  br i1 true, label %.preheader, label %.loop2

.preheader:
  br label %.loop1

.loop1:
  %indvar33 = phi i64 [ %indvar.next34, %.loop1 ], [ 0, %.preheader ]
  %indvar.next34 = add i64 %indvar33, 1
  %exitcond40 = icmp eq i64 %indvar.next34, 0
  br i1 %exitcond40, label %.loop2, label %.loop1

.loop2:
  %exitcond26.old = icmp eq i64 undef, %tmp25
  br i1 false, label %.loopexit, label %.loop2

.loopexit:
  %indvar.next16 = add i64 %indvar15, 1
  %exitcond53 = icmp eq i64 %indvar.next16, 2048
  br i1 %exitcond53, label %.start, label %.end

.end:
  ret void
}
