; RUN: opt %loadPolly -polly-opt-isl -polly-parallel -polly-codegen -S < %s | FileCheck %s
; llvm.org/PR51960

; CHECK-LABEL: define internal void @foo_polly_subfn
; CHECK: polly.stmt.for.body3:
; CHECK:   tail call i32 asm "664:\0A", "={ax},{di},~{dirflag},~{fpsr},~{flags}"(i32 0)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(ptr %bar) {
for.cond1.preheader.preheader:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv16 = phi i64 [ 0, %for.cond1.preheader.preheader ], [ %indvars.iv.next17, %for.inc6 ]
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 0, %for.cond1.preheader ]
  %xyzzy = tail call i32 asm "664:\0A", "={ax},{di},~{dirflag},~{fpsr},~{flags}"(i32 0) #0
  %arrayidx5 = getelementptr inbounds [1 x i32], ptr %bar, i64 0, i64 %indvars.iv
  store i32 %xyzzy, ptr %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv, 1
  br i1 %exitcond.not, label %for.inc6, label %for.body3

for.inc6:
  %indvars.iv.next17 = add nuw nsw i64 %indvars.iv16, 1
  %exitcond19.not = icmp eq i64 %indvars.iv16, 1
  br i1 %exitcond19.not, label %for.end8, label %for.cond1.preheader

for.end8:
  ret i32 0
}

attributes #0 = { readnone }
