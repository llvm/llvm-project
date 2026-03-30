; RUN: opt < %s -disable-output "-passes=print<da>" 2>&1 | FileCheck %s

define void @fun(ptr %arg) {
; CHECK-LABEL: 'fun'
; CHECK-NEXT:  Src:  store double 0.000000e+00, ptr %i8, align 8 --> Dst:  store double 0.000000e+00, ptr %i8, align 8
; CHECK-NEXT:    da analyze - output [* *]!
; CHECK-NEXT:  Src:  store double 0.000000e+00, ptr %i8, align 8 --> Dst:  %i16 = load double, ptr %i15, align 8
; CHECK-NEXT:    da analyze - flow [|<]!
; CHECK-NEXT:  Src:  %i16 = load double, ptr %i15, align 8 --> Dst:  %i16 = load double, ptr %i15, align 8
; CHECK-NEXT:    da analyze - input [* *]!

bb:
  br i1 false, label %bb1, label %bb10

bb1:                                              ; preds = %bb2, %bb
  %i = phi i64 [ %i3, %bb2 ], [ 0, %bb ]
  br label %bb5

bb2:                                              ; preds = %bb5
  %i3 = add i64 %i, 1
  %i4 = icmp eq i64 %i3, 10
  br i1 %i4, label %bb21, label %bb1

bb5:                                              ; preds = %bb5, %bb1
  %i6 = phi i64 [ %i, %bb1 ], [ %i7, %bb5 ]
  %i7 = add i64 %i6, 1
  %i8 = getelementptr inbounds i64, ptr %arg, i64 %i7
  store double 0.000000e+00, ptr %i8, align 8
  %i9 = icmp eq i64 %i6, 100
  br i1 %i9, label %bb2, label %bb5

bb10:                                             ; preds = %bb18, %bb
  %i11 = phi i64 [ %i19, %bb18 ], [ 0, %bb ]
  br label %bb12

bb12:                                             ; preds = %bb12, %bb10
  %i13 = phi i64 [ %i11, %bb10 ], [ %i14, %bb12 ]
  %i14 = add i64 %i13, 1
  %i15 = getelementptr inbounds i64, ptr %arg, i64 %i14
  %i16 = load double, ptr %i15, align 8
  %i17 = icmp eq i64 %i13, 100
  br i1 %i17, label %bb18, label %bb12

bb18:                                             ; preds = %bb12
  %i19 = add i64 %i11, 1
  %i20 = icmp eq i64 %i19, 10
  br i1 %i20, label %bb21, label %bb10

bb21:                                             ; preds = %bb18, %bb2
  ret void
}
