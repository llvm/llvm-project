; RUN: opt -S -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -mtriple=mips-linux-gnu < %s | FileCheck %s

define i32 @ctlz(i32 %A) {
; CHECK-LABEL: @ctlz(
; CHECK: [[ICMP:%[A-Za-z0-9]+]] = icmp eq i32 %A, 0
; CHECK-NEXT: [[CTZ:%[A-Za-z0-9]+]] = tail call i32 @llvm.ctlz.i32(i32 %A, i1 true)
; CHECK-NEXT: [[SEL:%[A-Za-z0-9.]+]] = select i1 [[ICMP]], i32 32, i32 [[CTZ]]
; CHECK-NEXT: ret i32 [[SEL]]
entry:
  %tobool = icmp eq i32 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:
  %0 = tail call i32 @llvm.ctlz.i32(i32 %A, i1 true)
  br label %cond.end

cond.end:
  %cond = phi i32 [ %0, %cond.true ], [ 32, %entry ]
  ret i32 %cond
}

define i32 @cttz(i32 %A) {
; CHECK-LABEL: @cttz(
; CHECK: [[ICMP:%[A-Za-z0-9]+]] = icmp eq i32 %A, 0
; CHECK-NEXT: [[CTZ:%[A-Za-z0-9]+]] = tail call i32 @llvm.cttz.i32(i32 %A, i1 true)
; CHECK-NEXT: [[SEL:%[A-Za-z0-9.]+]] = select i1 [[ICMP]], i32 32, i32 [[CTZ]]
; CHECK-NEXT: ret i32 [[SEL]]
entry:
  %tobool = icmp eq i32 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:
  %0 = tail call i32 @llvm.cttz.i32(i32 %A, i1 true)
  br label %cond.end

cond.end:
  %cond = phi i32 [ %0, %cond.true ], [ 32, %entry ]
  ret i32 %cond
}

declare i32 @llvm.ctlz.i32(i32, i1)
declare i32 @llvm.cttz.i32(i32, i1)

