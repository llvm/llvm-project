; RUN: opt -S -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure we can vectorize loops which contain lifetime markers.

; CHECK-LABEL: @test(
; CHECK: call void @llvm.lifetime.end
; CHECK: store <2 x i32>
; CHECK: call void @llvm.lifetime.start

define void @test(ptr %d) {
entry:
  %arr = alloca [1024 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 4096, ptr %arr) #1
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  call void @llvm.lifetime.end.p0(i64 4096, ptr %arr) #1
  %arrayidx = getelementptr inbounds i32, ptr %d, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 8
  store i32 100, ptr %arrayidx, align 8
  call void @llvm.lifetime.start.p0(i64 4096, ptr %arr) #1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  call void @llvm.lifetime.end.p0(i64 4096, ptr %arr) #1
  ret void
}

; CHECK-LABEL: @testbitcast(
; CHECK: call void @llvm.lifetime.end
; CHECK: store <2 x i32>
; CHECK: call void @llvm.lifetime.start

define void @testbitcast(ptr %d) {
entry:
  %arr = alloca [1024 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 4096, ptr %arr) #1
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  call void @llvm.lifetime.end.p0(i64 4096, ptr %arr) #1
  %arrayidx = getelementptr inbounds i32, ptr %d, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 8
  store i32 100, ptr %arrayidx, align 8
  call void @llvm.lifetime.start.p0(i64 4096, ptr %arr) #1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  call void @llvm.lifetime.end.p0(i64 4096, ptr %arr) #1
  ret void
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1
