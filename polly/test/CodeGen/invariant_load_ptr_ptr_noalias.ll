; RUN: opt -opaque-pointers=0 %loadPolly -polly-process-unprofitable -polly-codegen -polly-invariant-load-hoisting=true -polly-ignore-aliasing -S  < %s | FileCheck %s
;
; CHECK-LABEL: polly.preload.begin:
; CHECK:   %polly.access.A = getelementptr i32**, i32*** %A, i64 42
; CHECK:   %polly.access.A.load = load i32**, i32*** %polly.access.A
; CHECK:   %polly.access.polly.access.A.load = getelementptr i32*, i32** %polly.access.A.load, i64 32
; CHECK:   %polly.access.polly.access.A.load.load = load i32*, i32** %polly.access.polly.access.A.load
;
; CHECK: polly.stmt.bb2:
; CHECK: %scevgep = getelementptr i32, i32* %polly.access.polly.access.A.load.load, i64 %polly.indvar
; CHECK:   store i32 0, i32* %scevgep, align 4
;
;    void f(int ***A) {
;      for (int i = 0; i < 1024; i++)
;        A[42][32][i] = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32*** %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb7, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb7 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i32**, i32*** %A, i64 42
  %tmp3 = load i32**, i32*** %tmp, align 8
  %tmp4 = getelementptr inbounds i32*, i32** %tmp3, i64 32
  %tmp5 = load i32*, i32** %tmp4, align 8
  %tmp6 = getelementptr inbounds i32, i32* %tmp5, i64 %indvars.iv
  store i32 0, i32* %tmp6, align 4
  br label %bb7

bb7:                                              ; preds = %bb2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}
