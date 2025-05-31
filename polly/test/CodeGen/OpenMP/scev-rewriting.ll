; RUN: opt %loadNPMPolly < %s -polly-vectorizer=stripmine -polly-parallel -polly-parallel-force -polly-process-unprofitable -passes=polly-codegen -S | FileCheck %s
; CHECK: define internal void @DoStringSort_polly_subfn
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnueabi"

define void @DoStringSort() {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %i = phi i32 [ 0, %bb ], [ %i2, %bb1 ]
  %i2 = add i32 %i, 1
  br i1 undef, label %bb1, label %bb3

bb3:                                              ; preds = %bb1
  br i1 undef, label %bb6, label %bb4

bb4:                                              ; preds = %bb3
  %i5 = bitcast i8* undef to i32*
  br label %bb6

bb6:                                              ; preds = %bb4, %bb3
  %i7 = phi i32* [ %i5, %bb4 ], [ undef, %bb3 ]
  br i1 undef, label %bb21, label %bb8

bb8:                                              ; preds = %bb20, %bb6
  %i9 = phi i32* [ %i7, %bb6 ], [ %i10, %bb20 ]
  %i10 = getelementptr inbounds i32, i32* %i9, i32 %i2
  br i1 undef, label %bb11, label %bb20

bb11:                                             ; preds = %bb8
  br label %bb12

bb12:                                             ; preds = %bb11
  br label %bb13

bb13:                                             ; preds = %bb12
  br label %bb14

bb14:                                             ; preds = %bb14, %bb13
  %i15 = phi i32 [ %i17, %bb14 ], [ 1, %bb13 ]
  %i16 = getelementptr inbounds i32, i32* %i9, i32 %i15
  store i32 undef, i32* %i16, align 4
  %i17 = add i32 %i15, 1
  %i18 = icmp eq i32 %i15, %i
  br i1 %i18, label %bb19, label %bb14

bb19:                                             ; preds = %bb14
  br label %bb20

bb20:                                             ; preds = %bb19, %bb8
  br i1 undef, label %bb21, label %bb8

bb21:                                             ; preds = %bb20, %bb6
  unreachable
}
