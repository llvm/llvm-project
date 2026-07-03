; RUN: opt -passes=licm -S < %s | FileCheck %s
; REQUIRES: asserts

@global1 = external global i64, align 8
@global2 = external global [3 x [8 x [8 x { i32, i64, i8, i8, i16, i32 }]]], align 8

; CHECK-LABEL: @f0(i1 %arg)
define void @f0(i1 %arg) {
bb:
  br label %bb18.i

bb18.i:                                           ; preds = %bb49.us.i.us, %bb
  tail call void @f1()
  br i1 %arg, label %.exit.loopexit, label %bb49.preheader.i

bb49.preheader.i:                                 ; preds = %bb18.i
  br i1 %arg, label %bb49.us.preheader.i, label %bb78.loopexit3.i

bb49.us.preheader.i:                              ; preds = %bb49.preheader.i
  br label %bb49.us.i.us

bb49.us.i.preheader:                              ; No predecessors!
  br label %.exit

bb49.us.i.us:                                     ; preds = %bb49.us.preheader.i
  br label %bb18.i

bb78.loopexit3.i:                                 ; preds = %bb49.preheader.i
  store i64 0, ptr @global1, align 8
  br label %.exit

.exit.loopexit:                                   ; preds = %bb18.i
  br label %.exit

.exit:                                            ; preds = %.exit.loopexit, %bb78.loopexit3.i, %bb49.us.i.preheader
  br i1 %arg, label %bb4.i.us.preheader, label %bb4.i

bb4.i.us.preheader:                               ; preds = %.exit
  br label %bb4.i.us

bb4.i.us:                                         ; preds = %bb4.i.us, %bb4.i.us.preheader
  store i32 0, ptr undef, align 4
  store i32 undef, ptr getelementptr inbounds ([3 x [8 x [8 x { i32, i64, i8, i8, i16, i32 }]]], ptr @global2, i64 0, i64 0, i64 6, i64 6, i32 0), align 8
  br label %bb4.i.us

bb4.i:                                            ; preds = %.exit
  ret void
}

declare void @f1()


