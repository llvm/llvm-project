; RUN: opt < %s -loop-reduce -S | FileCheck %s

; LSR shouldn't consider %t8 to be an interesting user of %t6, and it
; should be able to form pretty GEPs.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @test(i8* %ptr.i8, float** %ptr.float) {
; CHECK-LABEL: define void @test(
bb:
  br label %bb3

bb1:                                              ; preds = %bb3
  br i1 undef, label %bb10, label %bb2

bb2:                                              ; preds = %bb1
  %t = add i64 %t4, 1                         ; <i64> [#uses=1]
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  %t4 = phi i64 [ %t, %bb2 ], [ 0, %bb ]      ; <i64> [#uses=3]
  br label %bb1

; CHECK: bb10:
; CHECK-NEXT: %t7 = icmp eq i64 %t4, 0
; Host %t2 computation outside the loop.
; CHECK-NEXT: [[SCEVGEP:%[^ ]+]] = getelementptr i8, i8* %ptr.i8, i64 %t4
; CHECK-NEXT: br label %bb14
bb10:                                             ; preds = %bb9
  %t7 = icmp eq i64 %t4, 0                    ; <i1> [#uses=1]
  %t3 = add i64 %t4, 16                     ; <i64> [#uses=1]
  br label %bb14

; CHECK: bb14:
; CHECK-NEXT: store i8 undef, i8* [[SCEVGEP]]
; CHECK-NEXT: %t6 = load float*, float** %ptr.float
; Fold %t3's add within the address.
; CHECK-NEXT: [[SCEVGEP1:%[^ ]+]] = getelementptr float, float* %t6, i64 4
; CHECK-NEXT: [[SCEVGEP2:%[^ ]+]] = bitcast float* [[SCEVGEP1]] to i8*
; Use the induction variable (%t4) to access the right element
; CHECK-NEXT: [[ADDRESS:%[^ ]+]] = getelementptr i8, i8* [[SCEVGEP2]], i64 %t4
; CHECK-NEXT: store i8 undef, i8* [[ADDRESS]]
; CHECK-NEXT: br label %bb14
bb14:                                             ; preds = %bb14, %bb10
  %t2 = getelementptr inbounds i8, i8* %ptr.i8, i64 %t4 ; <i8*> [#uses=1]
  store i8 undef, i8* %t2
  %t6 = load float*, float** %ptr.float
  %t8 = bitcast float* %t6 to i8*              ; <i8*> [#uses=1]
  %t9 = getelementptr inbounds i8, i8* %t8, i64 %t3 ; <i8*> [#uses=1]
  store i8 undef, i8* %t9
  br label %bb14
}

define fastcc void @TransformLine() nounwind {
; CHECK-LABEL: @TransformLine(
bb:
  br label %loop0

; CHECK: loop0:
; Induction variable is initialized to -2.
; CHECK-NEXT: [[PHIIV:%[^ ]+]] = phi i32 [ [[IVNEXT:%[^ ]+]], %loop0 ], [ -2, %bb ]
; CHECK-NEXT: [[IVNEXT]] = add nuw nsw i32 [[PHIIV]], 1
; CHECK-NEXT: br i1 false, label %loop0, label %bb0
loop0:                                            ; preds = %loop0, %bb
  %i0 = phi i32 [ %i0.next, %loop0 ], [ 0, %bb ]  ; <i32> [#uses=2]
  %i0.next = add i32 %i0, 1                       ; <i32> [#uses=1]
  br i1 false, label %loop0, label %bb0

bb0:                                              ; preds = %loop0
  br label %loop1

; CHECK: loop1:
; CHECK-NEXT: %i1 = phi i32 [ 0, %bb0 ], [ %i1.next, %bb5 ]
; IVNEXT covers the uses of %i0 and %t0.
; Therefore, %t0 has been removed.
; The critical edge has been split.
; CHECK-NEXT: br i1 false, label %bb2, label %[[LOOP1BB6:.+]]
loop1:                                            ; preds = %bb5, %bb0
  %i1 = phi i32 [ 0, %bb0 ], [ %i1.next, %bb5 ]   ; <i32> [#uses=4]
  %t0 = add i32 %i0, %i1                          ; <i32> [#uses=1]
  br i1 false, label %bb2, label %bb6

; CHECK: bb2:
; Critical edge split.
; CHECK-NEXT: br i1 true, label %[[BB2BB6:[^,]+]], label %bb5
bb2:                                              ; preds = %loop1
  br i1 true, label %bb6, label %bb5

; CHECK: bb5:
; CHECK-NEXT: %i1.next = add i32 %i1, 1
; CHECK-NEXT: br i1 true, label %[[BB5BB6:[^,]+]], label %loop1
bb5:                                              ; preds = %bb2
  %i1.next = add i32 %i1, 1                       ; <i32> [#uses=1]
  br i1 true, label %bb6, label %loop1

; bb5 to bb6 split basic block.
; CHECK: [[BB5BB6]]:
; CHECK-NEXT: [[INITIALVAL:%[^ ]+]] = add i32 [[IVNEXT]], %i1.next
; CHECK-NEXT: br label %[[SPLITTOBB6:.+]]

; bb2 to bb6 split basic block.
; CHECK: [[BB2BB6]]:
; CHECK-NEXT: br label %[[SPLITTOBB6]]

; Split basic blocks to bb6.
; CHECK: [[SPLITTOBB6]]:
; CHECK-NEXT: [[INITP8:%[^ ]+]] = phi i32 [ [[INITIALVAL]], %[[BB5BB6]] ], [ undef, %[[BB2BB6]] ]
; CHECK-NEXT: [[INITP9:%[^ ]+]] = phi i32 [ undef, %[[BB5BB6]] ], [ %i1, %[[BB2BB6]] ]
; CHECK-NEXT: br label %bb6
  
; CHECK: [[LOOP1BB6]]:
; CHECK-NEXT: br label %bb6

; CHECK: bb6:
; CHECK-NEXT: %p8 = phi i32 [ undef, %[[LOOP1BB6]] ], [ [[INITP8]], %[[SPLITTOBB6]] ]
; CHECK-NEXT: %p9 = phi i32 [ %i1, %[[LOOP1BB6]] ], [ [[INITP9]], %[[SPLITTOBB6]] ]
; CHECK-NEXT: unreachable
bb6:                                              ; preds = %bb5, %bb2, %loop1
  %p8 = phi i32 [ %t0, %bb5 ], [ undef, %loop1 ], [ undef, %bb2 ] ; <i32> [#uses=0]
  %p9 = phi i32 [ undef, %bb5 ], [ %i1, %loop1 ], [ %i1, %bb2 ] ; <i32> [#uses=0]
  unreachable
}
