; RUN: llc < %s -mtriple=x86_64-apple-darwin10 -frame-pointer=all
; rdar://7291624

%union.RtreeCoord = type { float }
%struct.RtreeCell = type { i64, [10 x %union.RtreeCoord] }
%struct.Rtree = type { i32, ptr, i32, i32, i32, i32, ptr, ptr }
%struct.RtreeNode = type { ptr, i64, i32, i32, ptr, ptr }

define fastcc void @nodeOverwriteCell(ptr nocapture %pRtree, ptr nocapture %pNode, ptr nocapture %pCell, i32 %iCell) nounwind ssp {
entry:
  %0 = load ptr, ptr undef, align 8                   ; <ptr> [#uses=2]
  %1 = load i32, ptr undef, align 8                   ; <i32> [#uses=1]
  %2 = mul i32 %1, %iCell                         ; <i32> [#uses=1]
  %3 = add nsw i32 %2, 4                          ; <i32> [#uses=1]
  %4 = sext i32 %3 to i64                         ; <i64> [#uses=2]
  %5 = load i64, ptr null, align 8                    ; <i64> [#uses=2]
  %6 = lshr i64 %5, 48                            ; <i64> [#uses=1]
  %7 = trunc i64 %6 to i8                         ; <i8> [#uses=1]
  store i8 %7, ptr undef, align 1
  %8 = lshr i64 %5, 8                             ; <i64> [#uses=1]
  %9 = trunc i64 %8 to i8                         ; <i8> [#uses=1]
  %.sum4 = add i64 %4, 6                          ; <i64> [#uses=1]
  %10 = getelementptr inbounds i8, ptr %0, i64 %.sum4 ; <ptr> [#uses=1]
  store i8 %9, ptr %10, align 1
  %11 = getelementptr inbounds %struct.Rtree, ptr %pRtree, i64 0, i32 3 ; <ptr> [#uses=1]
  br i1 undef, label %bb.nph, label %bb2

bb.nph:                                           ; preds = %entry
  %tmp25 = add i64 %4, 11                         ; <i64> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb ] ; <i64> [#uses=3]
  %scevgep = getelementptr %struct.RtreeCell, ptr %pCell, i64 0, i32 1, i64 %indvar ; <ptr> [#uses=1]
  %tmp = shl i64 %indvar, 2                       ; <i64> [#uses=1]
  %tmp26 = add i64 %tmp, %tmp25                   ; <i64> [#uses=1]
  %scevgep27 = getelementptr i8, ptr %0, i64 %tmp26   ; <ptr> [#uses=1]
  %12 = load i32, ptr %scevgep, align 4             ; <i32> [#uses=1]
  %13 = lshr i32 %12, 24                          ; <i32> [#uses=1]
  %14 = trunc i32 %13 to i8                       ; <i8> [#uses=1]
  store i8 %14, ptr undef, align 1
  store i8 undef, ptr %scevgep27, align 1
  %15 = load i32, ptr %11, align 4                    ; <i32> [#uses=1]
  %16 = shl i32 %15, 1                            ; <i32> [#uses=1]
  %17 = icmp sgt i32 %16, undef                   ; <i1> [#uses=1]
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br i1 %17, label %bb, label %bb2

bb2:                                              ; preds = %bb, %entry
  %18 = getelementptr inbounds %struct.RtreeNode, ptr %pNode, i64 0, i32 3 ; <ptr> [#uses=1]
  store i32 1, ptr %18, align 4
  ret void
}
