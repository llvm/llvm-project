; RUN: llc < %s -asm-verbose=false -O3 -relocation-model=pic -disable-fp-elim -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 -post-RA-scheduler

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-apple-darwin9"

%struct.Hosp = type { i32, i32, i32, %struct.List, %struct.List, %struct.List, %struct.List }
%struct.List = type { %struct.List*, %struct.Patient*, %struct.List* }
%struct.Patient = type { i32, i32, i32, %struct.Village* }
%struct.Village = type { [4 x %struct.Village*], %struct.Village*, %struct.List, %struct.Hosp, i32, i32 }

define %struct.Village* @alloc_tree(i32 %level, i32 %label, %struct.Village* %back, i1 %p) nounwind {
entry:
  br i1 %p, label %bb8, label %bb1

bb1:                                              ; preds = %entry
  %malloccall = tail call i8* @malloc(i32 ptrtoint (%struct.Village* getelementptr (%struct.Village* null, i32 1) to i32))
  %0 = bitcast i8* %malloccall to %struct.Village*
  %exp2 = call double @ldexp(double 1.000000e+00, i32 %level) nounwind ; <double> [#uses=1]
  %.c = fptosi double %exp2 to i32                ; <i32> [#uses=1]
  store i32 %.c, i32* null
  %1 = getelementptr %struct.Village* %0, i32 0, i32 3, i32 6, i32 0 ; <%struct.List**> [#uses=1]
  store %struct.List* null, %struct.List** %1
  %2 = getelementptr %struct.Village* %0, i32 0, i32 3, i32 6, i32 2 ; <%struct.List**> [#uses=1]
  store %struct.List* null, %struct.List** %2
  ret %struct.Village* %0

bb8:                                              ; preds = %entry
  ret %struct.Village* null
}

declare double @ldexp(double, i32)
declare noalias i8* @malloc(i32)
