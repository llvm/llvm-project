; RUN: opt -passes=gvn-sink -S < %s | FileCheck %s

%struct = type {i32, i32, i32}
%struct2 = type { [ 2 x i32], i32 }

; Struct indices cannot be variant.

; CHECK-LABEL: @f(i1 %arg) {
; CHECK: getelementptr
; CHECK: getelementptr
define void @f(i1 %arg) {
bb:
  br i1 %arg, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  %tmp = getelementptr inbounds %struct, ptr null, i64 0, i32 1
  br label %bb4

bb2:                                              ; preds = %bb
  %tmp3 = getelementptr inbounds %struct, ptr null, i64 0, i32 2
  br label %bb4

bb4:                                              ; preds = %bb2, %bb1
  %tmp5 = phi i32 [ 1, %bb1 ], [ 2, %bb2 ]
  ret void
}

; Struct indices cannot be variant.

; CHECK-LABEL: @g(i1 %arg) {
; CHECK: getelementptr
; CHECK: getelementptr
define void @g(i1 %arg) {
bb:
  br i1 %arg, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  %tmp = getelementptr inbounds %struct2, ptr null, i64 0, i32 0, i32 1
  br label %bb4

bb2:                                              ; preds = %bb
  %tmp3 = getelementptr inbounds %struct2, ptr null, i64 0, i32 0, i32 2
  br label %bb4

bb4:                                              ; preds = %bb2, %bb1
  %tmp5 = phi i32 [ 1, %bb1 ], [ 2, %bb2 ]
  ret void
}


; ... but the first parameter to a GEP can.

; CHECK-LABEL: @h(i1 %arg) {
; CHECK: getelementptr
; CHECK-NOT: getelementptr
define void @h(i1 %arg) {
bb:
  br i1 %arg, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  %tmp = getelementptr inbounds %struct, ptr null, i32 1, i32 0
  br label %bb4

bb2:                                              ; preds = %bb
  %tmp3 = getelementptr inbounds %struct, ptr null, i32 2, i32 0
  br label %bb4

bb4:                                              ; preds = %bb2, %bb1
  %tmp5 = phi i32 [ 1, %bb1 ], [ 2, %bb2 ]
  ret void
}
