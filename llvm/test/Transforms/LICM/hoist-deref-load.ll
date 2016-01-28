; RUN: opt -S -basicaa -licm < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; This test represents the following function:
; void test1(int * __restrict__ a, int * __restrict__ b, int &c, int n) {
;   for (int i = 0; i < n; ++i)
;     if (a[i] > 0)
;       a[i] = c*b[i];
; }
; and we want to hoist the load of %c out of the loop. This can be done only
; because the dereferenceable attribute is on %c.

; CHECK-LABEL: @test1
; CHECK: load i32, i32* %c, align 4
; CHECK: for.body:

define void @test1(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* nocapture readonly nonnull dereferenceable(4) %c, i32 %n) #0 {
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; This is the same as @test1, but without the dereferenceable attribute on %c.
; Without this attribute, we should not hoist the load of %c.

; CHECK-LABEL: @test2
; CHECK: if.then:
; CHECK: load i32, i32* %c, align 4

define void @test2(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* nocapture readonly nonnull %c, i32 %n) #0 {
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; This test represents the following function:
; void test3(int * restrict a, int * restrict b, int c[static 3], int n) {
;   for (int i = 0; i < n; ++i)
;     if (a[i] > 0)
;       a[i] = c[2]*b[i];
; }
; and we want to hoist the load of c[2] out of the loop. This can be done only
; because the dereferenceable attribute is on %c.

; CHECK-LABEL: @test3
; CHECK: load i32, i32* %c2, align 4
; CHECK: for.body:

define void @test3(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* nocapture readonly dereferenceable(12) %c, i32 %n) #0 {
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %c2 = getelementptr inbounds i32, i32* %c, i64 2
  %1 = load i32, i32* %c2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; This is the same as @test3, but with a dereferenceable attribute on %c with a
; size too small to cover c[2] (and so we should not hoist it).

; CHECK-LABEL: @test4
; CHECK: if.then:
; CHECK: load i32, i32* %c2, align 4

define void @test4(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32* nocapture readonly dereferenceable(11) %c, i32 %n) #0 {
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %c2 = getelementptr inbounds i32, i32* %c, i64 2
  %1 = load i32, i32* %c2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; This test represents the following function:
; void test1(int * __restrict__ a, int *b, int &c, int n) {
;   if (c != null)
;     for (int i = 0; i < n; ++i)
;       if (a[i] > 0)
;         a[i] = c*b[i];
; }
; and we want to hoist the load of %c out of the loop. This can be done only
; because the dereferenceable_or_null attribute is on %c and there is a null
; check on %c.

; CHECK-LABEL: @test5
; CHECK: load i32, i32* %c, align 4
; CHECK: for.body:

define void @test5(i32* noalias %a, i32* %b, i32* dereferenceable_or_null(4) %c, i32 %n) #0 {
entry:
  %not_null = icmp ne i32* %c, null
  br i1 %not_null, label %not.null, label %for.end

not.null:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %not.null, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %not.null ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry, %not.null
  ret void
}

; This is the same as @test5, but without the null check on %c.
; Without this check, we should not hoist the load of %c.

; This test case has an icmp on c but the use of this comparison is
; not a branch. 

; CHECK-LABEL: @test6
; CHECK: if.then:
; CHECK: load i32, i32* %c, align 4

define i1 @test6(i32* noalias %a, i32* %b, i32* dereferenceable_or_null(4) %c, i32 %n) #0 {
entry:
  %not_null = icmp ne i32* %c, null
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret i1 %not_null
}

; This test represents the following function:
; void test1(int * __restrict__ a, int *b, int **cptr, int n) {
;   c = *cptr;
;   for (int i = 0; i < n; ++i)
;     if (a[i] > 0)
;       a[i] = (*c)*b[i];
; }
; and we want to hoist the load of %c out of the loop. This can be done only
; because the dereferenceable meatdata on the c = *cptr load.

; CHECK-LABEL: @test7
; CHECK: load i32, i32* %c, align 4
; CHECK: for.body:

define void @test7(i32* noalias %a, i32* %b, i32** %cptr, i32 %n) #0 {
entry:
  %c = load i32*, i32** %cptr, !dereferenceable !0
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; This test represents the following function:
; void test1(int * __restrict__ a, int *b, int **cptr, int n) {
;   c = *cptr;
;   if (c != null)
;     for (int i = 0; i < n; ++i)
;       if (a[i] > 0)
;         a[i] = (*c)*b[i];
; }
; and we want to hoist the load of %c out of the loop. This can be done only
; because the dereferenceable_or_null meatdata on the c = *cptr load and there 
; is a null check on %c.

; CHECK-LABEL: @test8
; CHECK: load i32, i32* %c, align 4
; CHECK: for.body:

define void @test8(i32* noalias %a, i32* %b, i32** %cptr, i32 %n) #0 {
entry:
  %c = load i32*, i32** %cptr, !dereferenceable_or_null !0
  %not_null = icmp ne i32* %c, null
  br i1 %not_null, label %not.null, label %for.end

not.null:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %not.null, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %not.null ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry, %not.null
  ret void
}

; This is the same as @test8, but without the null check on %c.
; Without this check, we should not hoist the load of %c.

; CHECK-LABEL: @test9
; CHECK: if.then:
; CHECK: load i32, i32* %c, align 4

define void @test9(i32* noalias %a, i32* %b, i32** %cptr, i32 %n) #0 {
entry:
  %c = load i32*, i32** %cptr, !dereferenceable_or_null !0
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

; In this test we should be able to only hoist load from %cptr. We can't hoist
; load from %c because it's dereferenceability can depend on %cmp1 condition.
; By moving it out of the loop we break this dependency and can not rely
; on the dereferenceability anymore.
; In other words this test checks that we strip dereferenceability  metadata
; after hoisting an instruction.

; CHECK-LABEL: @test10
; CHECK: %c = load i32*, i32** %cptr
; CHECK-NOT: dereferenceable
; CHECK: if.then:
; CHECK: load i32, i32* %c, align 4

define void @test10(i32* noalias %a, i32* %b, i32** dereferenceable(8) %cptr, i32 %n) #0 {
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %c = load i32*, i32** %cptr, !dereferenceable !0
  %1 = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %2, %1
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

define void @test11(i32* noalias %a, i32* %b, i32** dereferenceable(8) %cptr, i32 %n) #0 {
; CHECK-LABEL: @test11(
entry:
  %cmp11 = icmp sgt i32 %n, 0
  br i1 %cmp11, label %for.body, label %for.end

; CHECK: for.body.preheader:
; CHECK:  %c = load i32*, i32** %cptr, !dereferenceable !0
; CHECK:  %d = load i32, i32* %c, align 4


for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 0
  %c = load i32*, i32** %cptr, !dereferenceable !0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %d = load i32, i32* %c, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %e = load i32, i32* %arrayidx3, align 4
  %mul = mul nsw i32 %e, %d
  store i32 %mul, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  ret void
}

attributes #0 = { nounwind uwtable }
!0 = !{i64 4}
