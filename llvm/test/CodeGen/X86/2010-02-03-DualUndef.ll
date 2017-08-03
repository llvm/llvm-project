; RUN: llc < %s -march=x86-64
; PR6086
define fastcc void @prepOutput() nounwind {
bb:                                               ; preds = %output.exit
  br label %bb.i1

bb.i1:                                            ; preds = %bb7.i, %bb
  br i1 undef, label %bb7.i, label %bb.nph.i

bb.nph.i:                                         ; preds = %bb.i1
  br label %bb3.i

bb3.i:                                            ; preds = %bb5.i6, %bb.nph.i
  %tmp10.i = trunc i64 undef to i32               ; <i32> [#uses=1]
  br i1 undef, label %bb4.i, label %bb5.i6

bb4.i:                                            ; preds = %bb3.i
  br label %bb5.i6

bb5.i6:                                           ; preds = %bb4.i, %bb3.i
  %0 = phi i32 [ undef, %bb4.i ], [ undef, %bb3.i ] ; <i32> [#uses=1]
  %1 = icmp slt i32 %0, %tmp10.i                  ; <i1> [#uses=1]
  br i1 %1, label %bb7.i, label %bb3.i

bb7.i:                                            ; preds = %bb5.i6, %bb.i1
  br label %bb.i1
}
