; RUN: opt -S -loop-vectorize -force-vector-width=2 -force-vector-interleave=1 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure we can vectorize loops which contain lifetime markers.

; CHECK-LABEL: @test(
; CHECK: call void @llvm.lifetime.end
; CHECK: store <2 x i32>
; CHECK: call void @llvm.lifetime.start

define void @test(i32 *%d) {
entry:
  %arr = alloca [1024 x i32], align 16
  %0 = bitcast [1024 x i32]* %arr to i8*
  call void @llvm.lifetime.start.p0i8(i64 4096, i8* %0) #1
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  call void @llvm.lifetime.end.p0i8(i64 4096, i8* %0) #1
  %arrayidx = getelementptr inbounds i32, i32* %d, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx, align 8
  store i32 100, i32* %arrayidx, align 8
  call void @llvm.lifetime.start.p0i8(i64 4096, i8* %0) #1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  call void @llvm.lifetime.end.p0i8(i64 4096, i8* %0) #1
  ret void
}

; CHECK-LABEL: @testbitcast(
; CHECK: call void @llvm.lifetime.end
; CHECK: store <2 x i32>
; CHECK: call void @llvm.lifetime.start

define void @testbitcast(i32 *%d) {
entry:
  %arr = alloca [1024 x i32], align 16
  %0 = bitcast [1024 x i32]* %arr to i8*
  call void @llvm.lifetime.start.p0i8(i64 4096, i8* %0) #1
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %1 = bitcast [1024 x i32]* %arr to i8*
  call void @llvm.lifetime.end.p0i8(i64 4096, i8* %1) #1
  %arrayidx = getelementptr inbounds i32, i32* %d, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx, align 8
  store i32 100, i32* %arrayidx, align 8
  call void @llvm.lifetime.start.p0i8(i64 4096, i8* %1) #1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  call void @llvm.lifetime.end.p0i8(i64 4096, i8* %0) #1
  ret void
}

; CHECK-LABEL: @testloopvariant(
; CHECK: call void @llvm.lifetime.end
; CHECK: store <2 x i32>
; CHECK: call void @llvm.lifetime.start

define void @testloopvariant(i32 *%d) {
entry:
  %arr = alloca [1024 x i32], align 16
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = getelementptr [1024 x i32], [1024 x i32]* %arr, i32 0, i64 %indvars.iv
  %1 = bitcast [1024 x i32]* %arr to i8*
  call void @llvm.lifetime.end.p0i8(i64 4096, i8* %1) #1
  %arrayidx = getelementptr inbounds i32, i32* %d, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx, align 8
  store i32 100, i32* %arrayidx, align 8
  call void @llvm.lifetime.start.p0i8(i64 4096, i8* %1) #1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1
