; REQUIRES: asserts
; RUN: not --crash opt -passes=loop-vectorize -force-vector-width=8 -disable-output %s

@postscale = external constant [64 x float]

define void @test(ptr %data) {
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %or.iv.1 = or disjoint i64 %iv, 1
  %gep.postscale = getelementptr [64 x float], ptr @postscale, i64 0, i64 %or.iv.1
  %load.postscale = load float, ptr %gep.postscale, align 4, !tbaa !0
  %lrint = tail call i64 @llvm.lrint.i64.f32(float %load.postscale)
  %lrint.trunc = trunc i64 %lrint to i16
  store i16 %lrint.trunc, ptr %data, align 2, !tbaa !4
  %iv.next = add i64 %iv, 1
  %exit.cond = icmp eq i64 %iv.next, 8
  br i1 %exit.cond, label %end, label %loop

end:                                              ; preds = %loop
  ret void
}

!0 = !{!1, !1, i64 0}
!1 = !{!"float", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"short", !2, i64 0}
