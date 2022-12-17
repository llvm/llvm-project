; RUN: opt -passes=loop-idiom -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1() {
entry:
  br label %for.body.preheader

for.body.preheader:                               ; preds = %for.cond
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i32 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %add.ptr3 = getelementptr inbounds i32, i32* null, i32 %indvars.iv
  %add.ptr4 = getelementptr inbounds i32, i32* %add.ptr3, i32 1
  %0 = load i32, i32* %add.ptr4, align 4
  store i32 %0, i32* %add.ptr3, align 4
  %indvars.iv.next = add nsw i32 %indvars.iv, 1
  %exitcond = icmp ne i32 %indvars.iv.next, 6
  br i1 %exitcond, label %for.body, label %for.body.preheader
}

; CHECK-LABEL: define void @test1(
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 null, i8* align 4 inttoptr (i64 4 to i8*), i64 24, i1 false)
; CHECK-NOT: store

define void @test1_no_null_opt() #0 {
entry:
  br label %for.body.preheader

for.body.preheader:                               ; preds = %for.cond
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i32 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %add.ptr3 = getelementptr inbounds i32, i32* null, i32 %indvars.iv
  %add.ptr4 = getelementptr inbounds i32, i32* %add.ptr3, i32 1
  %0 = load i32, i32* %add.ptr4, align 4
  store i32 %0, i32* %add.ptr3, align 4
  %indvars.iv.next = add nsw i32 %indvars.iv, 1
  %exitcond = icmp ne i32 %indvars.iv.next, 6
  br i1 %exitcond, label %for.body, label %for.body.preheader
}

; CHECK-LABEL: define void @test1_no_null_opt(
; CHECK-NOT: call void @llvm.memcpy
; CHECK: getelementptr
; CHECK: getelementptr
; CHECK: load
; CHECK: store

attributes #0 = { null_pointer_is_valid }
