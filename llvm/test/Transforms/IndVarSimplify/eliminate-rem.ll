; RUN: opt -passes=indvars -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; Indvars should be able to eliminate this srem.
; CHECK-LABEL: @simple(
; CHECK-NOT: rem
; CHECK: ret

define void @simple(i64 %arg, double* %arg3) nounwind {
bb:
  %t = icmp slt i64 0, %arg                     ; <i1> [#uses=1]
  br i1 %t, label %bb4, label %bb12

bb4:                                              ; preds = %bb
  br label %bb5

bb5:                                              ; preds = %bb4, %bb5
  %t6 = phi i64 [ %t9, %bb5 ], [ 0, %bb4 ]    ; <i64> [#uses=2]
  %t7 = srem i64 %t6, %arg                    ; <i64> [#uses=1]
  %t8 = getelementptr inbounds double, double* %arg3, i64 %t7 ; <double*> [#uses=1]
  store double 0.000000e+00, double* %t8
  %t9 = add nsw i64 %t6, 1                    ; <i64> [#uses=2]
  %t10 = icmp slt i64 %t9, %arg               ; <i1> [#uses=1]
  br i1 %t10, label %bb5, label %bb11

bb11:                                             ; preds = %bb5
  br label %bb12

bb12:                                             ; preds = %bb11, %bb
  ret void
}

; Indvars should be able to eliminate the (i+1)%n.
; CHECK-LABEL: @f(
; CHECK-NOT: {{[us]}}rem
; CHECK: {{[us]}}rem
; CHECK-NOT: {{[us]}}rem
; CHECK: ret

define i32 @f(i64* %arg, i64 %arg1, i64 %arg2, i64 %arg3) nounwind {
bb:
  %t = icmp sgt i64 %arg1, 0                      ; <i1> [#uses=1]
  br i1 %t, label %bb4, label %bb54

bb4:                                              ; preds = %bb
  br label %bb5

bb5:                                              ; preds = %bb49, %bb4
  %t6 = phi i64 [ %t51, %bb49 ], [ 0, %bb4 ]      ; <i64> [#uses=4]
  %t7 = phi i32 [ %t50, %bb49 ], [ 0, %bb4 ]      ; <i32> [#uses=2]
  %t8 = add nsw i64 %t6, %arg1                    ; <i64> [#uses=1]
  %t9 = add nsw i64 %t8, -2                       ; <i64> [#uses=1]
  %t10 = srem i64 %t9, %arg1                      ; <i64> [#uses=1]
  %t11 = add nsw i64 %t10, 1                      ; <i64> [#uses=1]
  %t12 = add nsw i64 %t6, 1                       ; <i64> [#uses=1]
  %t13 = srem i64 %t12, %arg1                     ; <i64> [#uses=1]
  %t14 = icmp sgt i64 %arg1, 0                    ; <i1> [#uses=1]
  br i1 %t14, label %bb15, label %bb49

bb15:                                             ; preds = %bb5
  br label %bb16

bb16:                                             ; preds = %bb44, %bb15
  %t17 = phi i64 [ %t46, %bb44 ], [ 0, %bb15 ]    ; <i64> [#uses=1]
  %t18 = phi i32 [ %t45, %bb44 ], [ %t7, %bb15 ]  ; <i32> [#uses=2]
  %t19 = icmp sgt i64 %arg1, 0                    ; <i1> [#uses=1]
  br i1 %t19, label %bb20, label %bb44

bb20:                                             ; preds = %bb16
  br label %bb21

bb21:                                             ; preds = %bb21, %bb20
  %t22 = phi i64 [ %t41, %bb21 ], [ 0, %bb20 ]    ; <i64> [#uses=4]
  %t23 = phi i32 [ %t40, %bb21 ], [ %t18, %bb20 ] ; <i32> [#uses=1]
  %t24 = mul i64 %t6, %arg1                       ; <i64> [#uses=1]
  %t25 = mul i64 %t13, %arg1                      ; <i64> [#uses=1]
  %t26 = add nsw i64 %t24, %t22                   ; <i64> [#uses=1]
  %t27 = mul i64 %t11, %arg1                      ; <i64> [#uses=1]
  %t28 = add nsw i64 %t25, %t22                   ; <i64> [#uses=1]
  %t29 = getelementptr inbounds i64, i64* %arg, i64 %t26 ; <i64*> [#uses=1]
  %t30 = add nsw i64 %t27, %t22                   ; <i64> [#uses=1]
  %t31 = getelementptr inbounds i64, i64* %arg, i64 %t28 ; <i64*> [#uses=1]
  %t32 = zext i32 %t23 to i64                     ; <i64> [#uses=1]
  %t33 = load i64, i64* %t29                           ; <i64> [#uses=1]
  %t34 = getelementptr inbounds i64, i64* %arg, i64 %t30 ; <i64*> [#uses=1]
  %t35 = load i64, i64* %t31                           ; <i64> [#uses=1]
  %t36 = add nsw i64 %t32, %t33                   ; <i64> [#uses=1]
  %t37 = add nsw i64 %t36, %t35                   ; <i64> [#uses=1]
  %t38 = load i64, i64* %t34                           ; <i64> [#uses=1]
  %t39 = add nsw i64 %t37, %t38                   ; <i64> [#uses=1]
  %t40 = trunc i64 %t39 to i32                    ; <i32> [#uses=2]
  %t41 = add nsw i64 %t22, 1                      ; <i64> [#uses=2]
  %t42 = icmp slt i64 %t41, %arg1                 ; <i1> [#uses=1]
  br i1 %t42, label %bb21, label %bb43

bb43:                                             ; preds = %bb21
  br label %bb44

bb44:                                             ; preds = %bb43, %bb16
  %t45 = phi i32 [ %t18, %bb16 ], [ %t40, %bb43 ] ; <i32> [#uses=2]
  %t46 = add nsw i64 %t17, 1                      ; <i64> [#uses=2]
  %t47 = icmp slt i64 %t46, %arg1                 ; <i1> [#uses=1]
  br i1 %t47, label %bb16, label %bb48

bb48:                                             ; preds = %bb44
  br label %bb49

bb49:                                             ; preds = %bb48, %bb5
  %t50 = phi i32 [ %t7, %bb5 ], [ %t45, %bb48 ]   ; <i32> [#uses=2]
  %t51 = add nsw i64 %t6, 1                       ; <i64> [#uses=2]
  %t52 = icmp slt i64 %t51, %arg1                 ; <i1> [#uses=1]
  br i1 %t52, label %bb5, label %bb53

bb53:                                             ; preds = %bb49
  br label %bb54

bb54:                                             ; preds = %bb53, %bb
  %t55 = phi i32 [ 0, %bb ], [ %t50, %bb53 ]      ; <i32> [#uses=1]
  ret i32 %t55
}
