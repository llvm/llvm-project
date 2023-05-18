; RUN: llc < %s -asm-verbose=false -O3 -relocation-model=pic -frame-pointer=all -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -post-RA-scheduler

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-apple-darwin9"

%struct.Hosp = type { i32, i32, i32, %struct.List, %struct.List, %struct.List, %struct.List }
%struct.List = type { ptr, ptr, ptr }
%struct.Patient = type { i32, i32, i32, ptr }
%struct.Village = type { [4 x ptr], ptr, %struct.List, %struct.Hosp, i32, i32 }

define ptr @alloc_tree(i32 %level, i32 %label, ptr %back, i1 %p) nounwind {
entry:
  br i1 %p, label %bb8, label %bb1

bb1:                                              ; preds = %entry
  %malloccall = tail call ptr @malloc(i32 ptrtoint (ptr getelementptr (%struct.Village, ptr null, i32 1) to i32))
  %exp2 = call double @ldexp(double 1.000000e+00, i32 %level) nounwind ; <double> [#uses=1]
  %.c = fptosi double %exp2 to i32                ; <i32> [#uses=1]
  store i32 %.c, ptr null
  %0 = getelementptr %struct.Village, ptr %malloccall, i32 0, i32 3, i32 6, i32 0 ; <ptr> [#uses=1]
  store ptr null, ptr %0
  %1 = getelementptr %struct.Village, ptr %malloccall, i32 0, i32 3, i32 6, i32 2 ; <ptr> [#uses=1]
  store ptr null, ptr %1
  ret ptr %malloccall

bb8:                                              ; preds = %entry
  ret ptr null
}

declare double @ldexp(double, i32)
declare noalias ptr @malloc(i32)
