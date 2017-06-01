; RUN: opt < %s -gvn -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

define i32 @test1(i32* %b, i32* %c) nounwind {
; CHECK-LABEL: @test1(
entry:
	%g = alloca i32
	%t1 = icmp eq i32* %b, null
	br i1 %t1, label %bb, label %bb1

bb:
	%t2 = load i32, i32* %c, align 4
	%t3 = add i32 %t2, 1
	store i32 %t3, i32* %g, align 4
	br label %bb2

bb1:		; preds = %entry
	%t5 = load i32, i32* %b, align 4
	%t6 = add i32 %t5, 1
	store i32 %t6, i32* %g, align 4
	br label %bb2

bb2:		; preds = %bb1, %bb
	%c_addr.0 = phi i32* [ %g, %bb1 ], [ %c, %bb ]
	%b_addr.0 = phi i32* [ %b, %bb1 ], [ %g, %bb ]
	%cv = load i32, i32* %c_addr.0, align 4
	%bv = load i32, i32* %b_addr.0, align 4
; CHECK: %bv = phi i32
; CHECK: %cv = phi i32
; CHECK-NOT: load
; CHECK: ret i32
	%ret = add i32 %cv, %bv
	ret i32 %ret
}

define i8 @test2(i1 %cond, i32* %b, i32* %c) nounwind {
; CHECK-LABEL: @test2(
entry:
  br i1 %cond, label %bb, label %bb1

bb:
  %b1 = bitcast i32* %b to i8*
  store i8 4, i8* %b1
  br label %bb2

bb1:
  %c1 = bitcast i32* %c to i8*
  store i8 92, i8* %c1
  br label %bb2

bb2:
  %d = phi i32* [ %c, %bb1 ], [ %b, %bb ]
  %d1 = bitcast i32* %d to i8*
  %dv = load i8, i8* %d1
; CHECK: %dv = phi i8 [ 92, %bb1 ], [ 4, %bb ]
; CHECK-NOT: load
; CHECK: ret i8 %dv
  ret i8 %dv
}

define i32 @test3(i1 %cond, i32* %b, i32* %c) nounwind {
; CHECK-LABEL: @test3(
entry:
  br i1 %cond, label %bb, label %bb1

bb:
  %b1 = getelementptr i32, i32* %b, i32 17
  store i32 4, i32* %b1
  br label %bb2

bb1:
  %c1 = getelementptr i32, i32* %c, i32 7
  store i32 82, i32* %c1
  br label %bb2

bb2:
  %d = phi i32* [ %c, %bb1 ], [ %b, %bb ]
  %i = phi i32 [ 7, %bb1 ], [ 17, %bb ]
  %d1 = getelementptr i32, i32* %d, i32 %i
  %dv = load i32, i32* %d1
; CHECK: %dv = phi i32 [ 82, %bb1 ], [ 4, %bb ]
; CHECK-NOT: load
; CHECK: ret i32 %dv
  ret i32 %dv
}

; PR5313
define i32 @test4(i1 %cond, i32* %b, i32* %c) nounwind {
; CHECK-LABEL: @test4(
entry:
  br i1 %cond, label %bb, label %bb1

bb:
  store i32 4, i32* %b
  br label %bb2

bb1:
  %c1 = getelementptr i32, i32* %c, i32 7
  store i32 82, i32* %c1
  br label %bb2

bb2:
  %d = phi i32* [ %c, %bb1 ], [ %b, %bb ]
  %i = phi i32 [ 7, %bb1 ], [ 0, %bb ]
  %d1 = getelementptr i32, i32* %d, i32 %i
  %dv = load i32, i32* %d1
; CHECK: %dv = phi i32 [ 82, %bb1 ], [ 4, %bb ]
; CHECK-NOT: load
; CHECK: ret i32 %dv
  ret i32 %dv
}



; void test5(int N, double* G) {
;   for (long j = 1; j < 1000; j++)
;     G[j] = G[j] + G[j-1];
; }
;
; Should compile into one load in the loop.
define void @test5(i32 %N, double* nocapture %G) nounwind ssp {
; CHECK-LABEL: @test5(
bb.nph:
  br label %for.body

for.body:
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp, %for.body ]
  %arrayidx6 = getelementptr double, double* %G, i64 %indvar
  %tmp = add i64 %indvar, 1
  %arrayidx = getelementptr double, double* %G, i64 %tmp
  %tmp3 = load double, double* %arrayidx
  %tmp7 = load double, double* %arrayidx6
  %add = fadd double %tmp3, %tmp7
  store double %add, double* %arrayidx
  %exitcond = icmp eq i64 %tmp, 999
  br i1 %exitcond, label %for.end, label %for.body
; CHECK: for.body:
; CHECK: phi double
; CHECK: load double
; CHECK-NOT: load double
; CHECK: br i1
for.end:
  ret void
}
