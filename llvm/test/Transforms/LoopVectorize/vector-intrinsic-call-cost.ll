; RUN: opt -S -loop-vectorize -force-vector-width=4 %s | FileCheck %s

; CHECK-LABEL: @test_fshl
; CHECK-LABEL: vector.body:
; CHECK-NEXT:    %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK-NEXT:    %0 = add i32 %index, 0
; CHECK-NEXT:    %1 = call <4 x i16> @llvm.fshl.v4i16(<4 x i16> undef, <4 x i16> undef, <4 x i16> <i16 15, i16 15, i16 15, i16 15>)
; CHECK-NEXT:    %index.next = add nuw i32 %index, 4
; CHECK-NEXT:    %2 = icmp eq i32 %index.next, %n.vec
; CHECK-NEXT:     br i1 %2, label %middle.block, label %vector.body, !llvm.loop !0
;
define void @test_fshl(i32 %width) {
entry:
  br label %for.body9.us.us

for.cond6.for.cond.cleanup8_crit_edge.us.us:      ; preds = %for.body9.us.us
  ret void

for.body9.us.us:                                  ; preds = %for.body9.us.us, %entry
  %x.020.us.us = phi i32 [ 0, %entry ], [ %inc.us.us, %for.body9.us.us ]
  %conv4.i.us.us = tail call i16 @llvm.fshl.i16(i16 undef, i16 undef, i16 15)
  %inc.us.us = add nuw i32 %x.020.us.us, 1
  %exitcond50 = icmp eq i32 %inc.us.us, %width
  br i1 %exitcond50, label %for.cond6.for.cond.cleanup8_crit_edge.us.us, label %for.body9.us.us
}

declare i16 @llvm.fshl.i16(i16, i16, i16)
