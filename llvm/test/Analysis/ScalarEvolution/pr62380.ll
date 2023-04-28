; RUN: opt -passes='loop(loop-deletion),loop-mssa(loop-predication,licm<allowspeculation>,simple-loop-unswitch<nontrivial>),loop(loop-predication)' -S < %s | FileCheck %s

; REQUIRES: asserts
; XFAIL: *

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

define void @test(i32 %arg) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb2, %bb
  %load = load i32, ptr null, align 4
  br label %bb2

bb2:                                              ; preds = %bb1
  br i1 false, label %bb3, label %bb1

bb3:                                              ; preds = %bb6, %bb2
  %phi = phi i32 [ %add, %bb6 ], [ 0, %bb2 ]
  %add = add i32 %phi, 1
  %icmp = icmp ult i32 %phi, %load
  br i1 %icmp, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  ret void

bb5:                                              ; preds = %bb3
  %call = call i1 @llvm.experimental.widenable.condition()
  br i1 %call, label %bb9, label %bb14

bb6:                                              ; preds = %bb9
  %add7 = add i32 %phi10, 1
  %icmp8 = icmp ugt i32 %phi10, 1
  br i1 %icmp8, label %bb3, label %bb9

bb9:                                              ; preds = %bb6, %bb5
  %phi10 = phi i32 [ %add7, %bb6 ], [ %phi, %bb5 ]
  %icmp11 = icmp ult i32 %phi10, %arg
  %call12 = call i1 @llvm.experimental.widenable.condition()
  %and = and i1 %icmp11, %call12
  br i1 %and, label %bb6, label %bb13

bb13:                                             ; preds = %bb9
  ret void

bb14:                                             ; preds = %bb5
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(inaccessiblemem: readwrite)
declare noundef i1 @llvm.experimental.widenable.condition() #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(inaccessiblemem: readwrite) }
