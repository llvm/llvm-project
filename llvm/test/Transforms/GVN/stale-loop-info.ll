; RUN: opt -passes='require<loops>,gvn' -S < %s | FileCheck %s

; This used to fail with ASAN enabled and if for some reason LoopInfo remained
; available during GVN.  In this case BasicAA will use LI but
; MergeBlockIntoPredecessor in GVN failed to update LI.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

%struct.wibble.1028 = type { i32, i32, %struct.barney.881 }
%struct.barney.881 = type { %struct.zot.882 }
%struct.zot.882 = type { [64 x i8] }

; Function Attrs: argmemonly
declare void @snork.1(ptr) local_unnamed_addr #0

define hidden zeroext i1 @eggs(ptr %arg, i1 %arg2) unnamed_addr align 2 {
bb:
  br i1 %arg2, label %bb14, label %bb3

bb3:                                              ; preds = %bb
  %tmp = getelementptr inbounds %struct.wibble.1028, ptr %arg, i64 0, i32 2, i32 0, i32 0, i64 0
  br label %bb6

bb6:                                              ; preds = %bb12, %bb3
  br label %bb7

bb7:                                              ; preds = %bb6
  br i1 undef, label %bb11, label %bb8

bb8:                                              ; preds = %bb7
  %tmp9 = load ptr, ptr %tmp, align 8
; CHECK: %tmp9 = load ptr, ptr %tmp, align 8
  br label %bb12

bb11:                                             ; preds = %bb7
  br label %bb12

bb12:                                             ; preds = %bb11, %bb8
  %tmp13 = phi ptr [ %tmp, %bb11 ], [ %tmp9, %bb8 ]
  call void @snork.1(ptr %tmp13) #1
  br label %bb6

bb14:                                             ; preds = %bb
  ret i1 false
}

attributes #0 = { argmemonly }
attributes #1 = { nounwind }
