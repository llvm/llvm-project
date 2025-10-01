; RUN: opt %loadNPMPolly -passes=polly-codegen -polly-parallel -S < %s | FileCheck %s

; CHECK: define ptr @ham(ptr %arg, i64 %arg1, i1 %arg2)

; This test is added to verify if the following IR does not crash on using different Dominator Tree when using polly parallel flag.

; ModuleID = '<stdin>'
source_filename = "<stdin>"

define ptr @ham(ptr %arg, i64 %arg1, i1 %arg2) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb8, %bb
  %phi = phi i64 [ 0, %bb ], [ %add9, %bb8 ]
  %getelementptr = getelementptr [64 x i16], ptr %arg, i64 %phi
  br label %bb4

bb4:                                              ; preds = %bb7, %bb3
  %phi5 = phi i64 [ %add, %bb7 ], [ 0, %bb3 ]
  %load = load i16, ptr null, align 2
  br i1 %arg2, label %bb7, label %bb6

bb6:                                              ; preds = %bb4
  store i16 0, ptr %getelementptr, align 2
  br label %bb7

bb7:                                              ; preds = %bb6, %bb4
  %add = add i64 %phi5, 1
  %icmp = icmp ne i64 %phi5, 64
  br i1 %icmp, label %bb4, label %bb8

bb8:                                              ; preds = %bb7
  %add9 = add i64 %phi, 1
  %icmp10 = icmp ult i64 %phi, %arg1
  br i1 %icmp10, label %bb3, label %bb11

bb11:                                             ; preds = %bb8
  ret ptr null
}

