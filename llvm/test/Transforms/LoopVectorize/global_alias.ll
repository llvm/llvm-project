; RUN: opt < %s -O1 -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"

%struct.anon = type { [100 x i32], i32, [100 x i32] }
%struct.anon.0 = type { [100 x [100 x i32]], i32, [100 x [100 x i32]] }

@Foo = common global %struct.anon zeroinitializer, align 4
@Bar = common global %struct.anon.0 zeroinitializer, align 4

@PB = external global i32*
@PA = external global i32*


;; === First, the tests that should always vectorize, whether statically or by adding run-time checks ===


; /// Different objects, positive induction, constant distance
; int noAlias01 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i] = Foo.B[i] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @noAlias01(
; CHECK: add nsw <4 x i32>
; CHECK: ret

define i32 @noAlias01(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %arrayidx1 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %4
  store i32 %add, i32* %arrayidx1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx2, align 4
  ret i32 %7
}

; /// Different objects, positive induction with widening slide
; int noAlias02 (int a) {
;   int i;
;   for (i=0; i<SIZE-10; i++)
;     Foo.A[i] = Foo.B[i+10] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @noAlias02(
; CHECK: add nsw <4 x i32>
; CHECK: ret

define i32 @noAlias02(i32 %a) {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 90
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %add = add nsw i32 %1, 10
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %add
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add1 = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %4
  store i32 %add1, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx3, align 4
  ret i32 %7
}

; /// Different objects, positive induction with shortening slide
; int noAlias03 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i+10] = Foo.B[i] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @noAlias03(
; CHECK: add nsw <4 x i32>
; CHECK: ret

define i32 @noAlias03(i32 %a) {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %add1 = add nsw i32 %4, 10
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %add1
  store i32 %add, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx3, align 4
  ret i32 %7
}

; /// Pointer access, positive stride, run-time check added
; int noAlias04 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     *(PA+i) = *(PB+i) + a;
;   return *(PA+a);
; }
; CHECK-LABEL: define i32 @noAlias04(
; CHECK-NOT: add nsw <4 x i32>
; CHECK: ret
;
; TODO: This test vectorizes (with run-time check) on real targets with -O3)
; Check why it's not being vectorized even when forcing vectorization

define i32 @noAlias04(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32*, i32** @PB, align 4
  %2 = load i32, i32* %i, align 4
  %add.ptr = getelementptr inbounds i32, i32* %1, i32 %2
  %3 = load i32, i32* %add.ptr, align 4
  %4 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %3, %4
  %5 = load i32*, i32** @PA, align 4
  %6 = load i32, i32* %i, align 4
  %add.ptr1 = getelementptr inbounds i32, i32* %5, i32 %6
  store i32 %add, i32* %add.ptr1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %8 = load i32*, i32** @PA, align 4
  %9 = load i32, i32* %a.addr, align 4
  %add.ptr2 = getelementptr inbounds i32, i32* %8, i32 %9
  %10 = load i32, i32* %add.ptr2, align 4
  ret i32 %10
}

; /// Different objects, positive induction, multi-array
; int noAlias05 (int a) {
;   int i, N=10;
;   for (i=0; i<SIZE; i++)
;     Bar.A[N][i] = Bar.B[N][i] + a;
;   return Bar.A[N][a];
; }
; CHECK-LABEL: define i32 @noAlias05(
; CHECK: add nsw <4 x i32>
; CHECK: ret

define i32 @noAlias05(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %N = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 10, i32* %N, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %2 = load i32, i32* %N, align 4
  %arrayidx = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @Bar, i32 0, i32 2), i32 0, i32 %2
  %arrayidx1 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx, i32 0, i32 %1
  %3 = load i32, i32* %arrayidx1, align 4
  %4 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %3, %4
  %5 = load i32, i32* %i, align 4
  %6 = load i32, i32* %N, align 4
  %arrayidx2 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @Bar, i32 0, i32 0), i32 0, i32 %6
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx2, i32 0, i32 %5
  store i32 %add, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %8 = load i32, i32* %a.addr, align 4
  %9 = load i32, i32* %N, align 4
  %arrayidx4 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @Bar, i32 0, i32 0), i32 0, i32 %9
  %arrayidx5 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx4, i32 0, i32 %8
  %10 = load i32, i32* %arrayidx5, align 4
  ret i32 %10
}

; /// Same objects, positive induction, multi-array, different sub-elements
; int noAlias06 (int a) {
;   int i, N=10;
;   for (i=0; i<SIZE; i++)
;     Bar.A[N][i] = Bar.A[N+1][i] + a;
;   return Bar.A[N][a];
; }
; CHECK-LABEL: define i32 @noAlias06(
; CHECK: add nsw <4 x i32>
; CHECK: ret

define i32 @noAlias06(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %N = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 10, i32* %N, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %2 = load i32, i32* %N, align 4
  %add = add nsw i32 %2, 1
  %arrayidx = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @Bar, i32 0, i32 0), i32 0, i32 %add
  %arrayidx1 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx, i32 0, i32 %1
  %3 = load i32, i32* %arrayidx1, align 4
  %4 = load i32, i32* %a.addr, align 4
  %add2 = add nsw i32 %3, %4
  %5 = load i32, i32* %i, align 4
  %6 = load i32, i32* %N, align 4
  %arrayidx3 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @Bar, i32 0, i32 0), i32 0, i32 %6
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx3, i32 0, i32 %5
  store i32 %add2, i32* %arrayidx4, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %8 = load i32, i32* %a.addr, align 4
  %9 = load i32, i32* %N, align 4
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @Bar, i32 0, i32 0), i32 0, i32 %9
  %arrayidx6 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx5, i32 0, i32 %8
  %10 = load i32, i32* %arrayidx6, align 4
  ret i32 %10
}

; /// Different objects, negative induction, constant distance
; int noAlias07 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[SIZE-i-1] = Foo.B[SIZE-i-1] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @noAlias07(
; CHECK: store <4 x i32>
; CHECK: ret
define i32 @noAlias07(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 1
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %sub1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %sub2 = sub nsw i32 100, %4
  %sub3 = sub nsw i32 %sub2, 1
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %sub3
  store i32 %add, i32* %arrayidx4, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx5 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx5, align 4
  ret i32 %7
}

; /// Different objects, negative induction, shortening slide
; int noAlias08 (int a) {
;   int i;
;   for (i=0; i<SIZE-10; i++)
;     Foo.A[SIZE-i-1] = Foo.B[SIZE-i-10] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @noAlias08(
; CHECK: load <4 x i32>
; CHECK: ret

define i32 @noAlias08(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 90
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 10
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %sub1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %sub2 = sub nsw i32 100, %4
  %sub3 = sub nsw i32 %sub2, 1
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %sub3
  store i32 %add, i32* %arrayidx4, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx5 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx5, align 4
  ret i32 %7
}

; /// Different objects, negative induction, widening slide
; int noAlias09 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[SIZE-i-10] = Foo.B[SIZE-i-1] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @noAlias09(
; CHECK: load <4 x i32>
; CHECK: ret

define i32 @noAlias09(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 1
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %sub1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %sub2 = sub nsw i32 100, %4
  %sub3 = sub nsw i32 %sub2, 10
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %sub3
  store i32 %add, i32* %arrayidx4, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx5 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx5, align 4
  ret i32 %7
}

; /// Pointer access, negative stride, run-time check added
; int noAlias10 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     *(PA+SIZE-i-1) = *(PB+SIZE-i-1) + a;
;   return *(PA+a);
; }
; CHECK-LABEL: define i32 @noAlias10(
; CHECK-NOT: sub {{.*}} <4 x i32>
; CHECK: ret
;
; TODO: This test vectorizes (with run-time check) on real targets with -O3)
; Check why it's not being vectorized even when forcing vectorization

define i32 @noAlias10(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32*, i32** @PB, align 4
  %add.ptr = getelementptr inbounds i32, i32* %1, i32 100
  %2 = load i32, i32* %i, align 4
  %idx.neg = sub i32 0, %2
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 %idx.neg
  %add.ptr2 = getelementptr inbounds i32, i32* %add.ptr1, i32 -1
  %3 = load i32, i32* %add.ptr2, align 4
  %4 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %3, %4
  %5 = load i32*, i32** @PA, align 4
  %add.ptr3 = getelementptr inbounds i32, i32* %5, i32 100
  %6 = load i32, i32* %i, align 4
  %idx.neg4 = sub i32 0, %6
  %add.ptr5 = getelementptr inbounds i32, i32* %add.ptr3, i32 %idx.neg4
  %add.ptr6 = getelementptr inbounds i32, i32* %add.ptr5, i32 -1
  store i32 %add, i32* %add.ptr6, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %8 = load i32*, i32** @PA, align 4
  %9 = load i32, i32* %a.addr, align 4
  %add.ptr7 = getelementptr inbounds i32, i32* %8, i32 %9
  %10 = load i32, i32* %add.ptr7, align 4
  ret i32 %10
}

; /// Different objects, negative induction, multi-array
; int noAlias11 (int a) {
;   int i, N=10;
;   for (i=0; i<SIZE; i++)
;     Bar.A[N][SIZE-i-1] = Bar.B[N][SIZE-i-1] + a;
;   return Bar.A[N][a];
; }
; CHECK-LABEL: define i32 @noAlias11(
; CHECK: store <4 x i32>
; CHECK: ret

define i32 @noAlias11(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %N = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 10, i32* %N, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 1
  %2 = load i32, i32* %N, align 4
  %arrayidx = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @Bar, i32 0, i32 2), i32 0, i32 %2
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx, i32 0, i32 %sub1
  %3 = load i32, i32* %arrayidx2, align 4
  %4 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %3, %4
  %5 = load i32, i32* %i, align 4
  %sub3 = sub nsw i32 100, %5
  %sub4 = sub nsw i32 %sub3, 1
  %6 = load i32, i32* %N, align 4
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @Bar, i32 0, i32 0), i32 0, i32 %6
  %arrayidx6 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx5, i32 0, i32 %sub4
  store i32 %add, i32* %arrayidx6, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %8 = load i32, i32* %a.addr, align 4
  %9 = load i32, i32* %N, align 4
  %arrayidx7 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @Bar, i32 0, i32 0), i32 0, i32 %9
  %arrayidx8 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx7, i32 0, i32 %8
  %10 = load i32, i32* %arrayidx8, align 4
  ret i32 %10
}

; /// Same objects, negative induction, multi-array, different sub-elements
; int noAlias12 (int a) {
;   int i, N=10;
;   for (i=0; i<SIZE; i++)
;     Bar.A[N][SIZE-i-1] = Bar.A[N+1][SIZE-i-1] + a;
;   return Bar.A[N][a];
; }
; CHECK-LABEL: define i32 @noAlias12(
; CHECK: store <4 x i32>
; CHECK: ret

define i32 @noAlias12(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %N = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 10, i32* %N, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 1
  %2 = load i32, i32* %N, align 4
  %add = add nsw i32 %2, 1
  %arrayidx = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @Bar, i32 0, i32 0), i32 0, i32 %add
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx, i32 0, i32 %sub1
  %3 = load i32, i32* %arrayidx2, align 4
  %4 = load i32, i32* %a.addr, align 4
  %add3 = add nsw i32 %3, %4
  %5 = load i32, i32* %i, align 4
  %sub4 = sub nsw i32 100, %5
  %sub5 = sub nsw i32 %sub4, 1
  %6 = load i32, i32* %N, align 4
  %arrayidx6 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @Bar, i32 0, i32 0), i32 0, i32 %6
  %arrayidx7 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx6, i32 0, i32 %sub5
  store i32 %add3, i32* %arrayidx7, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %8 = load i32, i32* %a.addr, align 4
  %9 = load i32, i32* %N, align 4
  %arrayidx8 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @Bar, i32 0, i32 0), i32 0, i32 %9
  %arrayidx9 = getelementptr inbounds [100 x i32], [100 x i32]* %arrayidx8, i32 0, i32 %8
  %10 = load i32, i32* %arrayidx9, align 4
  ret i32 %10
}

; /// Same objects, positive induction, constant distance, just enough for vector size
; int noAlias13 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i] = Foo.A[i+4] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @noAlias13(
; CHECK: add nsw <4 x i32>
; CHECK: ret

define i32 @noAlias13(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %add = add nsw i32 %1, 4
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %add
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add1 = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %4
  store i32 %add1, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx3, align 4
  ret i32 %7
}

; /// Same objects, negative induction, constant distance, just enough for vector size
; int noAlias14 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[SIZE-i-1] = Foo.A[SIZE-i-5] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @noAlias14(
; CHECK: load <4 x i32>
; CHECK: ret

define i32 @noAlias14(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 5
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %sub1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %sub2 = sub nsw i32 100, %4
  %sub3 = sub nsw i32 %sub2, 1
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %sub3
  store i32 %add, i32* %arrayidx4, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx5 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx5, align 4
  ret i32 %7
}


;; === Now, the tests that we could vectorize with induction changes or run-time checks ===


; /// Different objects, swapped induction, alias at the end
; int mayAlias01 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i] = Foo.B[SIZE-i-1] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @mayAlias01(
; CHECK-NOT: add nsw <4 x i32>
; CHECK: ret

define i32 @mayAlias01(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 1
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %sub1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %4
  store i32 %add, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx3, align 4
  ret i32 %7
}

; /// Different objects, swapped induction, alias at the beginning
; int mayAlias02 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[SIZE-i-1] = Foo.B[i] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @mayAlias02(
; CHECK-NOT: add nsw <4 x i32>
; CHECK: ret

define i32 @mayAlias02(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %sub = sub nsw i32 100, %4
  %sub1 = sub nsw i32 %sub, 1
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %sub1
  store i32 %add, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx3, align 4
  ret i32 %7
}

; /// Pointer access, run-time check added
; int mayAlias03 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     *(PA+i) = *(PB+SIZE-i-1) + a;
;   return *(PA+a);
; }
; CHECK-LABEL: define i32 @mayAlias03(
; CHECK-NOT: add nsw <4 x i32>
; CHECK: ret

define i32 @mayAlias03(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32*, i32** @PB, align 4
  %add.ptr = getelementptr inbounds i32, i32* %1, i32 100
  %2 = load i32, i32* %i, align 4
  %idx.neg = sub i32 0, %2
  %add.ptr1 = getelementptr inbounds i32, i32* %add.ptr, i32 %idx.neg
  %add.ptr2 = getelementptr inbounds i32, i32* %add.ptr1, i32 -1
  %3 = load i32, i32* %add.ptr2, align 4
  %4 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %3, %4
  %5 = load i32*, i32** @PA, align 4
  %6 = load i32, i32* %i, align 4
  %add.ptr3 = getelementptr inbounds i32, i32* %5, i32 %6
  store i32 %add, i32* %add.ptr3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %8 = load i32*, i32** @PA, align 4
  %9 = load i32, i32* %a.addr, align 4
  %add.ptr4 = getelementptr inbounds i32, i32* %8, i32 %9
  %10 = load i32, i32* %add.ptr4, align 4
  ret i32 %10
}


;; === Finally, the tests that should only vectorize with care (or if we ignore undefined behaviour at all) ===


; int mustAlias01 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i+10] = Foo.B[SIZE-i-1] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @mustAlias01(
; CHECK-NOT: add nsw <4 x i32>
; CHECK: ret

define i32 @mustAlias01(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 1
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %sub1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %add2 = add nsw i32 %4, 10
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %add2
  store i32 %add, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx4, align 4
  ret i32 %7
}

; int mustAlias02 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i] = Foo.B[SIZE-i-10] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @mustAlias02(
; CHECK-NOT: add nsw <4 x i32>
; CHECK: ret

define i32 @mustAlias02(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 10
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %sub1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %4
  store i32 %add, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx3, align 4
  ret i32 %7
}

; int mustAlias03 (int a) {
;   int i;
;   for (i=0; i<SIZE; i++)
;     Foo.A[i+10] = Foo.B[SIZE-i-10] + a;
;   return Foo.A[a];
; }
; CHECK-LABEL: define i32 @mustAlias03(
; CHECK-NOT: add nsw <4 x i32>
; CHECK: ret

define i32 @mustAlias03(i32 %a) nounwind {
entry:
  %a.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %sub = sub nsw i32 100, %1
  %sub1 = sub nsw i32 %sub, 10
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 2), i32 0, i32 %sub1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %2, %3
  %4 = load i32, i32* %i, align 4
  %add2 = add nsw i32 %4, 10
  %arrayidx3 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %add2
  store i32 %add, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %6 = load i32, i32* %a.addr, align 4
  %arrayidx4 = getelementptr inbounds [100 x i32], [100 x i32]* getelementptr inbounds (%struct.anon, %struct.anon* @Foo, i32 0, i32 0), i32 0, i32 %6
  %7 = load i32, i32* %arrayidx4, align 4
  ret i32 %7
}
