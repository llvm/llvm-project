;; This test is really dependent on propagating a lot of memory info around, but in the end, not
;; screwing up a single add.
; RUN: opt < %s -basic-aa -newgvn -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.Letter = type { i32, i32, i32, i32 }

@alPhrase = external local_unnamed_addr global [26 x %struct.Letter], align 16
@aqMainMask = external local_unnamed_addr global [2 x i64], align 16
@aqMainSign = external local_unnamed_addr global [2 x i64], align 16
@cchPhraseLength = external local_unnamed_addr global i32, align 4
@auGlobalFrequency = external local_unnamed_addr global [26 x i32], align 16
@.str.7 = external hidden unnamed_addr constant [28 x i8], align 1

; Function Attrs: nounwind uwtable
declare void @Fatal(i8*, i32) local_unnamed_addr #0

; Function Attrs: nounwind readnone
declare i16** @__ctype_b_loc() local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define void @BuildMask(i8* nocapture readonly) local_unnamed_addr #0 {
  tail call void @llvm.memset.p0i8.i64(i8* align 16 bitcast ([26 x %struct.Letter]* @alPhrase to i8*), i8 0, i64 416, i1 false)
  tail call void @llvm.memset.p0i8.i64(i8* align 16 bitcast ([2 x i64]* @aqMainMask to i8*), i8 0, i64 16, i1 false)
  tail call void @llvm.memset.p0i8.i64(i8* align 16 bitcast ([2 x i64]* @aqMainSign to i8*), i8 0, i64 16, i1 false)
  br label %.sink.split

.sink.split:                                      ; preds = %14, %1
  %.0 = phi i8* [ %0, %1 ], [ %.lcssa67, %14 ]
  %.sink = phi i32 [ 0, %1 ], [ %23, %14 ]
  store i32 %.sink, i32* @cchPhraseLength, align 4, !tbaa !1
  br label %2

; <label>:2:                                      ; preds = %6, %.sink.split
  %.1 = phi i8* [ %.0, %.sink.split ], [ %3, %6 ]
  %3 = getelementptr inbounds i8, i8* %.1, i64 1
  %4 = load i8, i8* %.1, align 1, !tbaa !5
  %5 = icmp eq i8 %4, 0
  br i1 %5, label %.preheader.preheader, label %6

.preheader.preheader:                             ; preds = %2
  br label %.preheader

; <label>:6:                                      ; preds = %2
  %7 = tail call i16** @__ctype_b_loc() #4
  %8 = load i16*, i16** %7, align 8, !tbaa !6
  %9 = sext i8 %4 to i64
  %10 = getelementptr inbounds i16, i16* %8, i64 %9
  %11 = load i16, i16* %10, align 2, !tbaa !8
  %12 = and i16 %11, 1024
  %13 = icmp eq i16 %12, 0
  br i1 %13, label %2, label %14

; <label>:14:                                     ; preds = %6
  %.lcssa67 = phi i8* [ %3, %6 ]
  %.lcssa65 = phi i8 [ %4, %6 ]
  %15 = sext i8 %.lcssa65 to i32
  %16 = tail call i32 @tolower(i32 %15) #5
  %17 = add nsw i32 %16, -97
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %18, i32 0
  %20 = load i32, i32* %19, align 16, !tbaa !10
  %21 = add i32 %20, 1
  store i32 %21, i32* %19, align 16, !tbaa !10
  %22 = load i32, i32* @cchPhraseLength, align 4, !tbaa !1
  %23 = add nsw i32 %22, 1
  br label %.sink.split

.preheader:                                       ; preds = %58, %.preheader.preheader
  %indvars.iv = phi i64 [ 0, %.preheader.preheader ], [ %indvars.iv.next, %58 ]
  %.04961 = phi i32 [ %.2, %58 ], [ 0, %.preheader.preheader ]
  %.05160 = phi i32 [ %.253, %58 ], [ 0, %.preheader.preheader ]
  %24 = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %indvars.iv, i32 0
  %25 = load i32, i32* %24, align 16, !tbaa !10
  %26 = icmp eq i32 %25, 0
  %27 = getelementptr inbounds [26 x i32], [26 x i32]* @auGlobalFrequency, i64 0, i64 %indvars.iv
  br i1 %26, label %28, label %29

; <label>:28:                                     ; preds = %.preheader
  store i32 -1, i32* %27, align 4, !tbaa !1
  br label %58

; <label>:29:                                     ; preds = %.preheader
  store i32 0, i32* %27, align 4, !tbaa !1
  %30 = zext i32 %25 to i64
  br i1 false, label %._crit_edge, label %.lr.ph.preheader

.lr.ph.preheader:                                 ; preds = %29
  br label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph, %.lr.ph.preheader
  %.04658 = phi i64 [ %32, %.lr.ph ], [ 1, %.lr.ph.preheader ]
  %.04857 = phi i32 [ %31, %.lr.ph ], [ 1, %.lr.ph.preheader ]
  %31 = add nuw nsw i32 %.04857, 1
  %32 = shl i64 %.04658, 1
  %33 = icmp ult i64 %30, %32
  br i1 %33, label %._crit_edge.loopexit, label %.lr.ph

._crit_edge.loopexit:                             ; preds = %.lr.ph
  %.lcssa63 = phi i32 [ %31, %.lr.ph ]
  %.lcssa = phi i64 [ %32, %.lr.ph ]
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %29
  %.048.lcssa = phi i32 [ 1, %29 ], [ %.lcssa63, %._crit_edge.loopexit ]
  %.046.lcssa = phi i64 [ 1, %29 ], [ %.lcssa, %._crit_edge.loopexit ]
  %34 = add nsw i32 %.048.lcssa, %.04961
  %35 = icmp ugt i32 %34, 64
  br i1 %35, label %36, label %40

; <label>:36:                                     ; preds = %._crit_edge
; This testcase essentially comes down to this little add.
; If we screw up the revisitation of the users of store of %sink above
; we will end up propagating and simplifying this to 1 in the final output
; because we keep an optimistic assumption we should not.
; CHECK:  add i32 %.05160, 1
  %37 = add i32 %.05160, 1
  %38 = icmp ugt i32 %37, 1
  br i1 %38, label %39, label %40

; <label>:39:                                     ; preds = %36
  tail call void @Fatal(i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str.7, i64 0, i64 0), i32 0)
  br label %40

; <label>:40:                                     ; preds = %39, %36, %._crit_edge
  %.152 = phi i32 [ %.05160, %._crit_edge ], [ %37, %39 ], [ %37, %36 ]
  %.150 = phi i32 [ %.04961, %._crit_edge ], [ 0, %39 ], [ 0, %36 ]
  %41 = add i64 %.046.lcssa, 4294967295
  %42 = trunc i64 %41 to i32
  %43 = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %indvars.iv, i32 2
  store i32 %42, i32* %43, align 8, !tbaa !12
  %44 = zext i32 %.150 to i64
  %.046. = shl i64 %.046.lcssa, %44
  %45 = zext i32 %.152 to i64
  %46 = getelementptr inbounds [2 x i64], [2 x i64]* @aqMainSign, i64 0, i64 %45
  %47 = load i64, i64* %46, align 8, !tbaa !13
  %48 = or i64 %47, %.046.
  store i64 %48, i64* %46, align 8, !tbaa !13
  %49 = load i32, i32* %24, align 16, !tbaa !10
  %50 = zext i32 %49 to i64
  %51 = shl i64 %50, %44
  %52 = getelementptr inbounds [2 x i64], [2 x i64]* @aqMainMask, i64 0, i64 %45
  %53 = load i64, i64* %52, align 8, !tbaa !13
  %54 = or i64 %51, %53
  store i64 %54, i64* %52, align 8, !tbaa !13
  %55 = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %indvars.iv, i32 1
  store i32 %.150, i32* %55, align 4, !tbaa !15
  %56 = getelementptr inbounds [26 x %struct.Letter], [26 x %struct.Letter]* @alPhrase, i64 0, i64 %indvars.iv, i32 3
  store i32 %.152, i32* %56, align 4, !tbaa !16
  %57 = add nsw i32 %.150, %.048.lcssa
  br label %58

; <label>:58:                                     ; preds = %40, %28
  %.253 = phi i32 [ %.05160, %28 ], [ %.152, %40 ]
  %.2 = phi i32 [ %.04961, %28 ], [ %57, %40 ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 26
  br i1 %exitcond, label %.preheader, label %59

; <label>:59:                                     ; preds = %58
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #2

; Function Attrs: inlinehint nounwind readonly uwtable
declare i32 @tolower(i32) local_unnamed_addr #3

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { inlinehint nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone }
attributes #5 = { nounwind readonly }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!3, !3, i64 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !3, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"short", !3, i64 0}
!10 = !{!11, !2, i64 0}
!11 = !{!"", !2, i64 0, !2, i64 4, !2, i64 8, !2, i64 12}
!12 = !{!11, !2, i64 8}
!13 = !{!14, !14, i64 0}
!14 = !{!"long", !3, i64 0}
!15 = !{!11, !2, i64 4}
!16 = !{!11, !2, i64 12}
