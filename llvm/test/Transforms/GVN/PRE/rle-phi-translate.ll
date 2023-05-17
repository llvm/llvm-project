; RUN: opt < %s -passes=gvn -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

define i32 @test1(ptr %b, ptr %c) nounwind {
; CHECK-LABEL: @test1(
entry:
	%g = alloca i32
	%t1 = icmp eq ptr %b, null
	br i1 %t1, label %bb, label %bb1

bb:
	%t2 = load i32, ptr %c, align 4
	%t3 = add i32 %t2, 1
	store i32 %t3, ptr %g, align 4
	br label %bb2

bb1:		; preds = %entry
	%t5 = load i32, ptr %b, align 4
	%t6 = add i32 %t5, 1
	store i32 %t6, ptr %g, align 4
	br label %bb2

bb2:		; preds = %bb1, %bb
	%c_addr.0 = phi ptr [ %g, %bb1 ], [ %c, %bb ]
	%b_addr.0 = phi ptr [ %b, %bb1 ], [ %g, %bb ]
	%cv = load i32, ptr %c_addr.0, align 4
	%bv = load i32, ptr %b_addr.0, align 4
; CHECK: %bv = phi i32
; CHECK: %cv = phi i32
; CHECK-NOT: load
; CHECK: ret i32
	%ret = add i32 %cv, %bv
	ret i32 %ret
}

define i8 @test2(i1 %cond, ptr %b, ptr %c) nounwind {
; CHECK-LABEL: @test2(
entry:
  br i1 %cond, label %bb, label %bb1

bb:
  store i8 4, ptr %b
  br label %bb2

bb1:
  store i8 92, ptr %c
  br label %bb2

bb2:
  %d = phi ptr [ %c, %bb1 ], [ %b, %bb ]
  %dv = load i8, ptr %d
; CHECK: %dv = phi i8 [ 92, %bb1 ], [ 4, %bb ]
; CHECK-NOT: load
; CHECK: ret i8 %dv
  ret i8 %dv
}

define i32 @test3(i1 %cond, ptr %b, ptr %c) nounwind {
; CHECK-LABEL: @test3(
entry:
  br i1 %cond, label %bb, label %bb1

bb:
  %b1 = getelementptr i32, ptr %b, i32 17
  store i32 4, ptr %b1
  br label %bb2

bb1:
  %c1 = getelementptr i32, ptr %c, i32 7
  store i32 82, ptr %c1
  br label %bb2

bb2:
  %d = phi ptr [ %c, %bb1 ], [ %b, %bb ]
  %i = phi i32 [ 7, %bb1 ], [ 17, %bb ]
  %d1 = getelementptr i32, ptr %d, i32 %i
  %dv = load i32, ptr %d1
; CHECK: %dv = phi i32 [ 82, %bb1 ], [ 4, %bb ]
; CHECK-NOT: load
; CHECK: ret i32 %dv
  ret i32 %dv
}

; PR5313
define i32 @test4(i1 %cond, ptr %b, ptr %c) nounwind {
; CHECK-LABEL: @test4(
entry:
  br i1 %cond, label %bb, label %bb1

bb:
  store i32 4, ptr %b
  br label %bb2

bb1:
  %c1 = getelementptr i32, ptr %c, i32 7
  store i32 82, ptr %c1
  br label %bb2

bb2:
  %d = phi ptr [ %c, %bb1 ], [ %b, %bb ]
  %i = phi i32 [ 7, %bb1 ], [ 0, %bb ]
  %d1 = getelementptr i32, ptr %d, i32 %i
  %dv = load i32, ptr %d1
; CHECK: %dv = phi i32 [ 82, %bb1 ], [ 4, %bb ]
; CHECK-NOT: load
; CHECK: ret i32 %dv
  ret i32 %dv
}



; void test5(int N, ptr G) {
;   for (long j = 1; j < 1000; j++)
;     G[j] = G[j] + G[j-1];
; }
;
; Should compile into one load in the loop.
define void @test5(i32 %N, ptr nocapture %G) nounwind ssp {
; CHECK-LABEL: @test5(
bb.nph:
  br label %for.body

for.body:
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp, %for.body ]
  %arrayidx6 = getelementptr double, ptr %G, i64 %indvar
  %tmp = add i64 %indvar, 1
  %arrayidx = getelementptr double, ptr %G, i64 %tmp
  %tmp3 = load double, ptr %arrayidx
  %tmp7 = load double, ptr %arrayidx6
  %add = fadd double %tmp3, %tmp7
  store double %add, ptr %arrayidx
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
