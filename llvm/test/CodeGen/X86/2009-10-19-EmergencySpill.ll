; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -frame-pointer=all
; rdar://7291624

%union.RtreeCoord = type { float }
%struct.RtreeCell = type { i64, [10 x %union.RtreeCoord] }
%struct.Rtree = type { i32, i32*, i32, i32, i32, i32, i8*, i8* }
%struct.RtreeNode = type { i32*, i64, i32, i32, i8*, i32* }

define fastcc void @nodeOverwriteCell(%struct.Rtree* nocapture %pRtree, %struct.RtreeNode* nocapture %pNode, %struct.RtreeCell* nocapture %pCell, i32 %iCell) nounwind ssp {
entry:
  %0 = load i8*, i8** undef, align 8                   ; <i8*> [#uses=2]
  %1 = load i32, i32* undef, align 8                   ; <i32> [#uses=1]
  %2 = mul i32 %1, %iCell                         ; <i32> [#uses=1]
  %3 = add nsw i32 %2, 4                          ; <i32> [#uses=1]
  %4 = sext i32 %3 to i64                         ; <i64> [#uses=2]
  %5 = load i64, i64* null, align 8                    ; <i64> [#uses=2]
  %6 = lshr i64 %5, 48                            ; <i64> [#uses=1]
  %7 = trunc i64 %6 to i8                         ; <i8> [#uses=1]
  store i8 %7, i8* undef, align 1
  %8 = lshr i64 %5, 8                             ; <i64> [#uses=1]
  %9 = trunc i64 %8 to i8                         ; <i8> [#uses=1]
  %.sum4 = add i64 %4, 6                          ; <i64> [#uses=1]
  %10 = getelementptr inbounds i8, i8* %0, i64 %.sum4 ; <i8*> [#uses=1]
  store i8 %9, i8* %10, align 1
  %11 = getelementptr inbounds %struct.Rtree, %struct.Rtree* %pRtree, i64 0, i32 3 ; <i32*> [#uses=1]
  br i1 undef, label %bb.nph, label %bb2

bb.nph:                                           ; preds = %entry
  %tmp25 = add i64 %4, 11                         ; <i64> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb ] ; <i64> [#uses=3]
  %scevgep = getelementptr %struct.RtreeCell, %struct.RtreeCell* %pCell, i64 0, i32 1, i64 %indvar ; <%union.RtreeCoord*> [#uses=1]
  %scevgep12 = bitcast %union.RtreeCoord* %scevgep to i32* ; <i32*> [#uses=1]
  %tmp = shl i64 %indvar, 2                       ; <i64> [#uses=1]
  %tmp26 = add i64 %tmp, %tmp25                   ; <i64> [#uses=1]
  %scevgep27 = getelementptr i8, i8* %0, i64 %tmp26   ; <i8*> [#uses=1]
  %12 = load i32, i32* %scevgep12, align 4             ; <i32> [#uses=1]
  %13 = lshr i32 %12, 24                          ; <i32> [#uses=1]
  %14 = trunc i32 %13 to i8                       ; <i8> [#uses=1]
  store i8 %14, i8* undef, align 1
  store i8 undef, i8* %scevgep27, align 1
  %15 = load i32, i32* %11, align 4                    ; <i32> [#uses=1]
  %16 = shl i32 %15, 1                            ; <i32> [#uses=1]
  %17 = icmp sgt i32 %16, undef                   ; <i1> [#uses=1]
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br i1 %17, label %bb, label %bb2

bb2:                                              ; preds = %bb, %entry
  %18 = getelementptr inbounds %struct.RtreeNode, %struct.RtreeNode* %pNode, i64 0, i32 3 ; <i32*> [#uses=1]
  store i32 1, i32* %18, align 4
  ret void
}
