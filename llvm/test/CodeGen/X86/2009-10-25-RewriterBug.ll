; RUN: llc < %s -mtriple=x86_64-apple-darwin -relocation-model=pic -frame-pointer=all

%struct.DecRefPicMarking_t = type { i32, i32, i32, i32, i32, %struct.DecRefPicMarking_t* }
%struct.FrameStore = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %struct.StorablePicture*, %struct.StorablePicture*, %struct.StorablePicture* }
%struct.StorablePicture = type { i32, i32, i32, i32, i32, [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16**, i16***, i8*, i16**, i8***, i64***, i64***, i16****, i8**, i8**, %struct.StorablePicture*, %struct.StorablePicture*, %struct.StorablePicture*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], i32, %struct.DecRefPicMarking_t*, i32 }

define fastcc void @insert_picture_in_dpb(%struct.FrameStore* nocapture %fs, %struct.StorablePicture* %p) nounwind ssp {
entry:
  %0 = getelementptr inbounds %struct.FrameStore, %struct.FrameStore* %fs, i64 0, i32 12 ; <%struct.StorablePicture**> [#uses=1]
  %1 = icmp eq i32 undef, 0                       ; <i1> [#uses=1]
  br i1 %1, label %bb.i, label %bb36.i

bb.i:                                             ; preds = %entry
  br i1 undef, label %bb3.i, label %bb14.preheader.i

bb3.i:                                            ; preds = %bb.i
  unreachable

bb14.preheader.i:                                 ; preds = %bb.i
  br i1 undef, label %bb9.i, label %bb20.preheader.i

bb9.i:                                            ; preds = %bb9.i, %bb14.preheader.i
  br i1 undef, label %bb9.i, label %bb20.preheader.i

bb20.preheader.i:                                 ; preds = %bb9.i, %bb14.preheader.i
  br i1 undef, label %bb18.i, label %bb29.preheader.i

bb18.i:                                           ; preds = %bb20.preheader.i
  unreachable

bb29.preheader.i:                                 ; preds = %bb20.preheader.i
  br i1 undef, label %bb24.i, label %bb30.i

bb24.i:                                           ; preds = %bb29.preheader.i
  unreachable

bb30.i:                                           ; preds = %bb29.preheader.i
  store i32 undef, i32* undef, align 8
  br label %bb67.preheader.i

bb36.i:                                           ; preds = %entry
  br label %bb67.preheader.i

bb67.preheader.i:                                 ; preds = %bb36.i, %bb30.i
  %2 = phi %struct.StorablePicture* [ null, %bb36.i ], [ undef, %bb30.i ] ; <%struct.StorablePicture*> [#uses=2]
  %3 = phi %struct.StorablePicture* [ null, %bb36.i ], [ undef, %bb30.i ] ; <%struct.StorablePicture*> [#uses=2]
  %4 = phi %struct.StorablePicture* [ null, %bb36.i ], [ undef, %bb30.i ] ; <%struct.StorablePicture*> [#uses=2]
  %5 = phi %struct.StorablePicture* [ null, %bb36.i ], [ undef, %bb30.i ] ; <%struct.StorablePicture*> [#uses=1]
  %6 = phi %struct.StorablePicture* [ null, %bb36.i ], [ undef, %bb30.i ] ; <%struct.StorablePicture*> [#uses=1]
  %7 = phi %struct.StorablePicture* [ null, %bb36.i ], [ undef, %bb30.i ] ; <%struct.StorablePicture*> [#uses=1]
  %8 = phi %struct.StorablePicture* [ null, %bb36.i ], [ undef, %bb30.i ] ; <%struct.StorablePicture*> [#uses=1]
  %9 = phi %struct.StorablePicture* [ null, %bb36.i ], [ undef, %bb30.i ] ; <%struct.StorablePicture*> [#uses=1]
  %10 = phi %struct.StorablePicture* [ null, %bb36.i ], [ undef, %bb30.i ] ; <%struct.StorablePicture*> [#uses=1]
  %11 = phi %struct.StorablePicture* [ null, %bb36.i ], [ undef, %bb30.i ] ; <%struct.StorablePicture*> [#uses=1]
  %12 = phi %struct.StorablePicture* [ null, %bb36.i ], [ undef, %bb30.i ] ; <%struct.StorablePicture*> [#uses=1]
  br i1 undef, label %bb38.i, label %bb68.i

bb38.i:                                           ; preds = %bb66.i, %bb67.preheader.i
  %13 = phi %struct.StorablePicture* [ %37, %bb66.i ], [ %2, %bb67.preheader.i ] ; <%struct.StorablePicture*> [#uses=1]
  %14 = phi %struct.StorablePicture* [ %38, %bb66.i ], [ %3, %bb67.preheader.i ] ; <%struct.StorablePicture*> [#uses=1]
  %15 = phi %struct.StorablePicture* [ %39, %bb66.i ], [ %4, %bb67.preheader.i ] ; <%struct.StorablePicture*> [#uses=1]
  %16 = phi %struct.StorablePicture* [ %40, %bb66.i ], [ %5, %bb67.preheader.i ] ; <%struct.StorablePicture*> [#uses=1]
  %17 = phi %struct.StorablePicture* [ %40, %bb66.i ], [ %6, %bb67.preheader.i ] ; <%struct.StorablePicture*> [#uses=1]
  %18 = phi %struct.StorablePicture* [ %40, %bb66.i ], [ %7, %bb67.preheader.i ] ; <%struct.StorablePicture*> [#uses=1]
  %19 = phi %struct.StorablePicture* [ %40, %bb66.i ], [ %8, %bb67.preheader.i ] ; <%struct.StorablePicture*> [#uses=1]
  %20 = phi %struct.StorablePicture* [ %40, %bb66.i ], [ %9, %bb67.preheader.i ] ; <%struct.StorablePicture*> [#uses=1]
  %21 = phi %struct.StorablePicture* [ %40, %bb66.i ], [ %10, %bb67.preheader.i ] ; <%struct.StorablePicture*> [#uses=1]
  %22 = phi %struct.StorablePicture* [ %40, %bb66.i ], [ %11, %bb67.preheader.i ] ; <%struct.StorablePicture*> [#uses=1]
  %23 = phi %struct.StorablePicture* [ %40, %bb66.i ], [ %12, %bb67.preheader.i ] ; <%struct.StorablePicture*> [#uses=1]
  %indvar248.i = phi i64 [ %indvar.next249.i, %bb66.i ], [ 0, %bb67.preheader.i ] ; <i64> [#uses=3]
  %storemerge52.i = trunc i64 %indvar248.i to i32 ; <i32> [#uses=1]
  %24 = getelementptr inbounds %struct.StorablePicture, %struct.StorablePicture* %23, i64 0, i32 19 ; <i32*> [#uses=0]
  br i1 undef, label %bb.nph51.i, label %bb66.i

bb.nph51.i:                                       ; preds = %bb38.i
  %25 = sdiv i32 %storemerge52.i, 8               ; <i32> [#uses=0]
  br label %bb39.i

bb39.i:                                           ; preds = %bb64.i, %bb.nph51.i
  %26 = phi %struct.StorablePicture* [ %17, %bb.nph51.i ], [ null, %bb64.i ] ; <%struct.StorablePicture*> [#uses=1]
  %27 = phi %struct.StorablePicture* [ %18, %bb.nph51.i ], [ null, %bb64.i ] ; <%struct.StorablePicture*> [#uses=0]
  %28 = phi %struct.StorablePicture* [ %19, %bb.nph51.i ], [ null, %bb64.i ] ; <%struct.StorablePicture*> [#uses=0]
  %29 = phi %struct.StorablePicture* [ %20, %bb.nph51.i ], [ null, %bb64.i ] ; <%struct.StorablePicture*> [#uses=0]
  %30 = phi %struct.StorablePicture* [ %21, %bb.nph51.i ], [ null, %bb64.i ] ; <%struct.StorablePicture*> [#uses=0]
  %31 = phi %struct.StorablePicture* [ %22, %bb.nph51.i ], [ null, %bb64.i ] ; <%struct.StorablePicture*> [#uses=0]
  br i1 undef, label %bb57.i, label %bb40.i

bb40.i:                                           ; preds = %bb39.i
  br i1 undef, label %bb57.i, label %bb41.i

bb41.i:                                           ; preds = %bb40.i
  %storemerge10.i = select i1 undef, i32 2, i32 4 ; <i32> [#uses=1]
  %32 = zext i32 %storemerge10.i to i64           ; <i64> [#uses=1]
  br i1 undef, label %bb45.i, label %bb47.i

bb45.i:                                           ; preds = %bb41.i
  %33 = getelementptr inbounds %struct.StorablePicture, %struct.StorablePicture* %26, i64 0, i32 5, i64 undef, i64 %32, i64 undef ; <i64*> [#uses=1]
  %34 = load i64, i64* %33, align 8                    ; <i64> [#uses=1]
  br label %bb47.i

bb47.i:                                           ; preds = %bb45.i, %bb41.i
  %storemerge11.i = phi i64 [ %34, %bb45.i ], [ 0, %bb41.i ] ; <i64> [#uses=0]
  %scevgep246.i = getelementptr i64, i64* undef, i64 undef ; <i64*> [#uses=0]
  br label %bb64.i

bb57.i:                                           ; preds = %bb40.i, %bb39.i
  br i1 undef, label %bb58.i, label %bb60.i

bb58.i:                                           ; preds = %bb57.i
  br label %bb60.i

bb60.i:                                           ; preds = %bb58.i, %bb57.i
  %35 = load i64**, i64*** undef, align 8                ; <i64**> [#uses=1]
  %scevgep256.i = getelementptr i64*, i64** %35, i64 %indvar248.i ; <i64**> [#uses=1]
  %36 = load i64*, i64** %scevgep256.i, align 8         ; <i64*> [#uses=1]
  %scevgep243.i = getelementptr i64, i64* %36, i64 undef ; <i64*> [#uses=1]
  store i64 -1, i64* %scevgep243.i, align 8
  br label %bb64.i

bb64.i:                                           ; preds = %bb60.i, %bb47.i
  br i1 undef, label %bb39.i, label %bb66.i

bb66.i:                                           ; preds = %bb64.i, %bb38.i
  %37 = phi %struct.StorablePicture* [ %13, %bb38.i ], [ null, %bb64.i ] ; <%struct.StorablePicture*> [#uses=2]
  %38 = phi %struct.StorablePicture* [ %14, %bb38.i ], [ null, %bb64.i ] ; <%struct.StorablePicture*> [#uses=2]
  %39 = phi %struct.StorablePicture* [ %15, %bb38.i ], [ null, %bb64.i ] ; <%struct.StorablePicture*> [#uses=2]
  %40 = phi %struct.StorablePicture* [ %16, %bb38.i ], [ null, %bb64.i ] ; <%struct.StorablePicture*> [#uses=8]
  %indvar.next249.i = add i64 %indvar248.i, 1     ; <i64> [#uses=1]
  br i1 undef, label %bb38.i, label %bb68.i

bb68.i:                                           ; preds = %bb66.i, %bb67.preheader.i
  %41 = phi %struct.StorablePicture* [ %2, %bb67.preheader.i ], [ %37, %bb66.i ] ; <%struct.StorablePicture*> [#uses=0]
  %42 = phi %struct.StorablePicture* [ %3, %bb67.preheader.i ], [ %38, %bb66.i ] ; <%struct.StorablePicture*> [#uses=1]
  %43 = phi %struct.StorablePicture* [ %4, %bb67.preheader.i ], [ %39, %bb66.i ] ; <%struct.StorablePicture*> [#uses=1]
  br i1 undef, label %bb.nph48.i, label %bb108.i

bb.nph48.i:                                       ; preds = %bb68.i
  br label %bb80.i

bb80.i:                                           ; preds = %bb104.i, %bb.nph48.i
  %44 = phi %struct.StorablePicture* [ %42, %bb.nph48.i ], [ null, %bb104.i ] ; <%struct.StorablePicture*> [#uses=1]
  %45 = phi %struct.StorablePicture* [ %43, %bb.nph48.i ], [ null, %bb104.i ] ; <%struct.StorablePicture*> [#uses=1]
  br i1 undef, label %bb.nph39.i, label %bb104.i

bb.nph39.i:                                       ; preds = %bb80.i
  br label %bb81.i

bb81.i:                                           ; preds = %bb102.i, %bb.nph39.i
  %46 = phi %struct.StorablePicture* [ %44, %bb.nph39.i ], [ %48, %bb102.i ] ; <%struct.StorablePicture*> [#uses=0]
  %47 = phi %struct.StorablePicture* [ %45, %bb.nph39.i ], [ %48, %bb102.i ] ; <%struct.StorablePicture*> [#uses=0]
  br i1 undef, label %bb83.i, label %bb82.i

bb82.i:                                           ; preds = %bb81.i
  br i1 undef, label %bb83.i, label %bb101.i

bb83.i:                                           ; preds = %bb82.i, %bb81.i
  br label %bb102.i

bb101.i:                                          ; preds = %bb82.i
  br label %bb102.i

bb102.i:                                          ; preds = %bb101.i, %bb83.i
  %48 = load %struct.StorablePicture*, %struct.StorablePicture** %0, align 8 ; <%struct.StorablePicture*> [#uses=2]
  br i1 undef, label %bb81.i, label %bb104.i

bb104.i:                                          ; preds = %bb102.i, %bb80.i
  br label %bb80.i

bb108.i:                                          ; preds = %bb68.i
  unreachable
}
