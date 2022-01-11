; RUN: opt < %s -loop-reduce -S | FileCheck %s

; LSR shouldn't consider %t8 to be an interesting user of %t6, and it
; should be able to form pretty GEPs.

target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

; Copy of uglygep with a different address space
; This tests expandAddToGEP uses the right smaller integer type for
; another address space
define void @Z4(i8 addrspace(1)* %ptr.i8, float addrspace(1)* addrspace(1)* %ptr.float) {
; CHECK: define void @Z4
bb:
  br label %bb3

bb1:                                              ; preds = %bb3
  br i1 undef, label %bb10, label %bb2

bb2:                                              ; preds = %bb1
  %t = add i16 %t4, 1                         ; <i16> [#uses=1]
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  %t4 = phi i16 [ %t, %bb2 ], [ 0, %bb ]      ; <i16> [#uses=3]
  br label %bb1

; CHECK: bb10:
; CHECK-NEXT: %t7 = icmp eq i16 %t4, 0
; Host %t2 computation outside the loop.
; CHECK-NEXT: [[SCEVGEP:%[^ ]+]] = getelementptr i8, i8 addrspace(1)* %ptr.i8, i16 %t4
; CHECK-NEXT: br label %bb14
bb10:                                             ; preds = %bb9
  %t7 = icmp eq i16 %t4, 0                    ; <i1> [#uses=1]
  %t3 = add i16 %t4, 16                     ; <i16> [#uses=1]
  br label %bb14

; CHECK: bb14:
; CHECK-NEXT: store i8 undef, i8 addrspace(1)* [[SCEVGEP]]
; CHECK-NEXT: %t6 = load float addrspace(1)*, float addrspace(1)* addrspace(1)* %ptr.float
; Fold %t3's add within the address.
; CHECK-NEXT: [[SCEVGEP1:%[^ ]+]] = getelementptr float, float addrspace(1)* %t6, i16 4
; CHECK-NEXT: [[SCEVGEP2:%[^ ]+]] = bitcast float addrspace(1)* [[SCEVGEP1]] to i8 addrspace(1)*
; Use the induction variable (%t4) to access the right element
; CHECK-NEXT: [[ADDRESS:%[^ ]+]] = getelementptr i8, i8 addrspace(1)* [[SCEVGEP2]], i16 %t4
; CHECK-NEXT: store i8 undef, i8 addrspace(1)* [[ADDRESS]]
; CHECK-NEXT: br label %bb14
bb14:                                             ; preds = %bb14, %bb10
  %t2 = getelementptr inbounds i8, i8 addrspace(1)* %ptr.i8, i16 %t4 ; <i8*> [#uses=1]
  store i8 undef, i8 addrspace(1)* %t2
  %t6 = load float addrspace(1)*, float addrspace(1)* addrspace(1)* %ptr.float
  %t8 = bitcast float addrspace(1)* %t6 to i8 addrspace(1)*              ; <i8*> [#uses=1]
  %t9 = getelementptr inbounds i8, i8 addrspace(1)* %t8, i16 %t3 ; <i8*> [#uses=1]
  store i8 undef, i8 addrspace(1)* %t9
  br label %bb14
}

