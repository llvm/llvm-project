; RUN: llc -march=hexagon -O3 -verify-machineinstrs < %s | FileCheck %s
;
; Make sure that this testcase passes the verifier.
; CHECK: call f1

target triple = "hexagon"

%s.0 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i64, i32, i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

@g0 = external global %s.0, align 8
@g1 = external hidden unnamed_addr constant [3 x i8], align 1

; Function Attrs: nounwind
define void @f0() local_unnamed_addr #0 {
b0:
  br label %b1

b1:                                               ; preds = %b0
  switch i8 undef, label %b3 [
    i8 35, label %b2
    i8 10, label %b2
  ]

b2:                                               ; preds = %b1, %b1
  unreachable

b3:                                               ; preds = %b1
  br label %b4

b4:                                               ; preds = %b3
  switch i8 undef, label %b6 [
    i8 35, label %b5
    i8 10, label %b5
  ]

b5:                                               ; preds = %b4, %b4
  unreachable

b6:                                               ; preds = %b4
  call void (i8*, i8*, ...) @f1(i8* nonnull undef, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @g1, i32 0, i32 0), i32* getelementptr inbounds (%s.0, %s.0* @g0, i32 0, i32 45)) #0
  br label %b7

b7:                                               ; preds = %b6
  switch i8 undef, label %b9 [
    i8 35, label %b8
    i8 10, label %b8
  ]

b8:                                               ; preds = %b7, %b7
  unreachable

b9:                                               ; preds = %b7
  br label %b10

b10:                                              ; preds = %b9
  switch i8 undef, label %b12 [
    i8 35, label %b11
    i8 10, label %b11
  ]

b11:                                              ; preds = %b10, %b10
  unreachable

b12:                                              ; preds = %b10
  br label %b13

b13:                                              ; preds = %b12
  switch i8 undef, label %b14 [
    i8 35, label %b15
    i8 10, label %b15
  ]

b14:                                              ; preds = %b13
  br label %b16

b15:                                              ; preds = %b13, %b13
  unreachable

b16:                                              ; preds = %b17, %b14
  %v0 = phi i8* [ %v2, %b17 ], [ undef, %b14 ]
  %v1 = load i8, i8* %v0, align 1
  switch i8 %v1, label %b17 [
    i8 32, label %b18
    i8 9, label %b18
  ]

b17:                                              ; preds = %b16
  %v2 = getelementptr inbounds i8, i8* %v0, i32 1
  br label %b16

b18:                                              ; preds = %b16, %b16
  unreachable
}

; Function Attrs: nounwind
declare void @f1(i8* nocapture readonly, i8* nocapture readonly, ...) local_unnamed_addr #0

attributes #0 = { nounwind "target-cpu"="hexagonv62" }
