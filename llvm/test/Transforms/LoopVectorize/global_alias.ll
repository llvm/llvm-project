; RUN: opt -passes='loop-vectorize,dce,instcombine' -force-vector-interleave=1 -force-vector-width=4 -S %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"

%struct.anon = type { [100 x i32], i32, [100 x i32] }
%struct.anon.0 = type { [100 x [100 x i32]], i32, [100 x [100 x i32]] }

@Foo = common global %struct.anon zeroinitializer, align 4
@Bar = common global %struct.anon.0 zeroinitializer, align 4

@PB = external global ptr
@PA = external global ptr


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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds %struct.anon, ptr @Foo, i32 0, i32 2, i32 %i.05
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a
  %arrayidx1 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %i.05
  store i32 %add, ptr %arrayidx1, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx2 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx2, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %add = add nuw nsw i32 %i.05, 10
  %arrayidx = getelementptr inbounds %struct.anon, ptr @Foo, i32 0, i32 2, i32 %add
  %0 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %0, %a
  %arrayidx2 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %i.05
  store i32 %add1, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 90
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx3 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx3, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds %struct.anon, ptr @Foo, i32 0, i32 2, i32 %i.05
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a
  %add1 = add nuw nsw i32 %i.05, 10
  %arrayidx2 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %add1
  store i32 %add, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx3 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx3, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load ptr, ptr @PB, align 4
  %add.ptr = getelementptr inbounds i32, ptr %0, i32 %i.05
  %1 = load i32, ptr %add.ptr, align 4
  %add = add nsw i32 %1, %a
  %2 = load ptr, ptr @PA, align 4
  %add.ptr1 = getelementptr inbounds i32, ptr %2, i32 %i.05
  store i32 %add, ptr %add.ptr1, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %3 = load ptr, ptr @PA, align 4
  %add.ptr2 = getelementptr inbounds i32, ptr %3, i32 %a
  %4 = load i32, ptr %add.ptr2, align 4
  ret i32 %4
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx1 = getelementptr inbounds %struct.anon.0, ptr @Bar, i32 0, i32 2, i32 10, i32 %i.07
  %0 = load i32, ptr %arrayidx1, align 4
  %add = add nsw i32 %0, %a
  %arrayidx3 = getelementptr inbounds %struct.anon.0, ptr @Bar, i32 0, i32 0, i32 10, i32 %i.07
  store i32 %add, ptr %arrayidx3, align 4
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx5 = getelementptr inbounds %struct.anon.0, ptr @Bar, i32 0, i32 0, i32 10, i32 %a
  %1 = load i32, ptr %arrayidx5, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx1 = getelementptr inbounds %struct.anon.0, ptr @Bar, i32 0, i32 0, i32 11, i32 %i.07
  %0 = load i32, ptr %arrayidx1, align 4
  %add2 = add nsw i32 %0, %a
  %arrayidx4 = getelementptr inbounds %struct.anon.0, ptr @Bar, i32 0, i32 0, i32 10, i32 %i.07
  store i32 %add2, ptr %arrayidx4, align 4
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx6 = getelementptr inbounds %struct.anon.0, ptr @Bar, i32 0, i32 0, i32 10, i32 %a
  %1 = load i32, ptr %arrayidx6, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sub1 = sub nuw nsw i32 99, %i.05
  %arrayidx = getelementptr inbounds %struct.anon, ptr @Foo, i32 0, i32 2, i32 %sub1
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a
  %arrayidx4 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %sub1
  store i32 %add, ptr %arrayidx4, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx5 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx5, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sub1 = sub nuw nsw i32 90, %i.05
  %arrayidx = getelementptr inbounds %struct.anon, ptr @Foo, i32 0, i32 2, i32 %sub1
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a
  %sub3 = sub nuw nsw i32 99, %i.05
  %arrayidx4 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %sub3
  store i32 %add, ptr %arrayidx4, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 90
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx5 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx5, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sub1 = sub nuw nsw i32 99, %i.05
  %arrayidx = getelementptr inbounds %struct.anon, ptr @Foo, i32 0, i32 2, i32 %sub1
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a
  %sub3 = sub nsw i32 90, %i.05
  %arrayidx4 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %sub3
  store i32 %add, ptr %arrayidx4, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx5 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx5, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load ptr, ptr @PB, align 4
  %add.ptr = getelementptr inbounds i8, ptr %0, i32 400
  %idx.neg = sub nsw i32 0, %i.05
  %add.ptr1 = getelementptr inbounds i32, ptr %add.ptr, i32 %idx.neg
  %add.ptr2 = getelementptr inbounds i8, ptr %add.ptr1, i32 -4
  %1 = load i32, ptr %add.ptr2, align 4
  %add = add nsw i32 %1, %a
  %2 = load ptr, ptr @PA, align 4
  %add.ptr3 = getelementptr inbounds i8, ptr %2, i32 400
  %add.ptr5 = getelementptr inbounds i32, ptr %add.ptr3, i32 %idx.neg
  %add.ptr6 = getelementptr inbounds i8, ptr %add.ptr5, i32 -4
  store i32 %add, ptr %add.ptr6, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %3 = load ptr, ptr @PA, align 4
  %add.ptr7 = getelementptr inbounds i32, ptr %3, i32 %a
  %4 = load i32, ptr %add.ptr7, align 4
  ret i32 %4
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sub1 = sub nuw nsw i32 99, %i.07
  %arrayidx2 = getelementptr inbounds %struct.anon.0, ptr @Bar, i32 0, i32 2, i32 10, i32 %sub1
  %0 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %0, %a
  %arrayidx6 = getelementptr inbounds %struct.anon.0, ptr @Bar, i32 0, i32 0, i32 10, i32 %sub1
  store i32 %add, ptr %arrayidx6, align 4
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx8 = getelementptr inbounds %struct.anon.0, ptr @Bar, i32 0, i32 0, i32 10, i32 %a
  %1 = load i32, ptr %arrayidx8, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sub1 = sub nuw nsw i32 99, %i.07
  %arrayidx2 = getelementptr inbounds %struct.anon.0, ptr @Bar, i32 0, i32 0, i32 11, i32 %sub1
  %0 = load i32, ptr %arrayidx2, align 4
  %add3 = add nsw i32 %0, %a
  %arrayidx7 = getelementptr inbounds %struct.anon.0, ptr @Bar, i32 0, i32 0, i32 10, i32 %sub1
  store i32 %add3, ptr %arrayidx7, align 4
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx9 = getelementptr inbounds %struct.anon.0, ptr @Bar, i32 0, i32 0, i32 10, i32 %a
  %1 = load i32, ptr %arrayidx9, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %add = add nuw nsw i32 %i.05, 4
  %arrayidx = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %add
  %0 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %0, %a
  %arrayidx2 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %i.05
  store i32 %add1, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx3 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx3, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sub1 = sub nsw i32 95, %i.05
  %arrayidx = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %sub1
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a
  %sub3 = sub nuw nsw i32 99, %i.05
  %arrayidx4 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %sub3
  store i32 %add, ptr %arrayidx4, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx5 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx5, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sub1 = sub nuw nsw i32 99, %i.05
  %arrayidx = getelementptr inbounds %struct.anon, ptr @Foo, i32 0, i32 2, i32 %sub1
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a
  %arrayidx2 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %i.05
  store i32 %add, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx3 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx3, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds %struct.anon, ptr @Foo, i32 0, i32 2, i32 %i.05
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a
  %sub1 = sub nuw nsw i32 99, %i.05
  %arrayidx2 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %sub1
  store i32 %add, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx3 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx3, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load ptr, ptr @PB, align 4
  %add.ptr = getelementptr inbounds i8, ptr %0, i32 400
  %idx.neg = sub nsw i32 0, %i.05
  %add.ptr1 = getelementptr inbounds i32, ptr %add.ptr, i32 %idx.neg
  %add.ptr2 = getelementptr inbounds i8, ptr %add.ptr1, i32 -4
  %1 = load i32, ptr %add.ptr2, align 4
  %add = add nsw i32 %1, %a
  %2 = load ptr, ptr @PA, align 4
  %add.ptr3 = getelementptr inbounds i32, ptr %2, i32 %i.05
  store i32 %add, ptr %add.ptr3, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %3 = load ptr, ptr @PA, align 4
  %add.ptr4 = getelementptr inbounds i32, ptr %3, i32 %a
  %4 = load i32, ptr %add.ptr4, align 4
  ret i32 %4
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sub1 = sub nuw nsw i32 99, %i.05
  %arrayidx = getelementptr inbounds %struct.anon, ptr @Foo, i32 0, i32 2, i32 %sub1
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a
  %add2 = add nuw nsw i32 %i.05, 10
  %arrayidx3 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %add2
  store i32 %add, ptr %arrayidx3, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx4 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx4, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sub1 = sub nsw i32 90, %i.05
  %arrayidx = getelementptr inbounds %struct.anon, ptr @Foo, i32 0, i32 2, i32 %sub1
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a
  %arrayidx2 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %i.05
  store i32 %add, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx3 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx3, align 4
  ret i32 %1
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sub1 = sub nsw i32 90, %i.05
  %arrayidx = getelementptr inbounds %struct.anon, ptr @Foo, i32 0, i32 2, i32 %sub1
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %a
  %add2 = add nuw nsw i32 %i.05, 10
  %arrayidx3 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %add2
  store i32 %add, ptr %arrayidx3, align 4
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, 100
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx4 = getelementptr inbounds [100 x i32], ptr @Foo, i32 0, i32 %a
  %1 = load i32, ptr %arrayidx4, align 4
  ret i32 %1
}
