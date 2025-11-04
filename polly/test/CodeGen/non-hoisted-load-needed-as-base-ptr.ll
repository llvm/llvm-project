; RUN: opt %loadNPMPolly -aa-pipeline=tbaa -passes=polly-codegen -disable-output %s
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.1 = type { ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, i32, [38 x i8], [128 x i8], [38 x i32], [256 x i8], [256 x i8], [256 x i8], %struct.4, [25 x [16 x %struct.4]], [128 x [64 x i16]] }
%struct.2 = type { i16, i16, i32, i32 }
%struct.3 = type { i8, i8, i16, i16 }
%struct.4 = type { i16, i8, i8 }

define void @AllocUnitsRare(ptr %p, i32 %indx) {
entry:
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %i.0 = phi i32 [ %inc, %do.body ], [ %indx, %entry ]
  %inc = add i32 %i.0, 1
  br i1 undef, label %do.body, label %do.end

do.end:                                           ; preds = %do.body
  %Base.i = getelementptr inbounds %struct.1, ptr %p, i32 0, i32 12
  %tmp.i = load ptr, ptr %Base.i, align 8, !tbaa !0
  %idxprom.i = zext i32 %inc to i64
  %FreeList.i = getelementptr inbounds %struct.1, ptr %p, i32 0, i32 20
  %arrayidx.i = getelementptr inbounds [38 x i32], ptr %FreeList.i, i64 0, i64 %idxprom.i
  %tmp2.i = load i32, ptr %tmp.i, align 4, !tbaa !8
  store i32 %tmp2.i, ptr %arrayidx.i, align 4, !tbaa !8
  %Indx2Units.i = getelementptr inbounds %struct.1, ptr %p, i32 0, i32 18
  %cmp.i = icmp ne i32 0, 3
  br i1 %cmp.i, label %if.then.i, label %SplitBlock.exit

if.then.i:                                        ; preds = %do.end
  br label %SplitBlock.exit

SplitBlock.exit:                                  ; preds = %if.then.i, %do.end
  ret void
}

!0 = !{!1, !2, i64 64}
!1 = !{!"", !2, i64 0, !2, i64 8, !2, i64 16, !5, i64 24, !5, i64 28, !5, i64 32, !5, i64 36, !5, i64 40, !5, i64 44, !5, i64 48, !5, i64 52, !5, i64 56, !2, i64 64, !2, i64 72, !2, i64 80, !2, i64 88, !2, i64 96, !5, i64 104, !3, i64 108, !3, i64 146, !3, i64 276, !3, i64 428, !3, i64 684, !3, i64 940, !6, i64 1196, !3, i64 1200, !3, i64 2800}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!"int", !3, i64 0}
!6 = !{!"", !7, i64 0, !3, i64 2, !3, i64 3}
!7 = !{!"short", !3, i64 0}
!8 = !{!5, !5, i64 0}
