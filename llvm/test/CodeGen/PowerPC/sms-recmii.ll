; REQUIRES: asserts
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -verify-machineinstrs\
; RUN:       -mcpu=pwr9 --ppc-enable-pipeliner --debug-only=pipeliner 2>&1 | FileCheck %s

; Test that the pipeliner doesn't overestimate the recurrence MII when evaluating circuits.
; CHECK: MII = 16 MAX_II = 26 (rec=16, res=5)
define dso_local void @comp_method(ptr noalias nocapture noundef readonly %0, ptr nocapture noundef writeonly %1, ptr nocapture noundef writeonly %2, i32 noundef %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, i64 %v1) local_unnamed_addr {
  %8 = icmp sgt i32 %3, 64
  tail call void @llvm.assume(i1 %8)
  %9 = and i32 %3, 1
  %10 = icmp eq i32 %9, 0
  tail call void @llvm.assume(i1 %10)
  %11 = sext i32 %5 to i64
  %12 = sext i32 %6 to i64
  %13 = zext nneg i32 %3 to i64
  %14 = getelementptr i8, ptr %2, i64 %12
  br label %16

15:
  ret void

16:
  %17 = phi i64 [ 0, %7 ], [ %24, %16 ]
  %18 = getelementptr inbounds i8, ptr %0, i64 %17
  %19 = load i8, ptr %18, align 1
  %20 = sext i8 %19 to i64
  %21 = getelementptr inbounds i8, ptr %1, i64 %20
  store i8 2, ptr %21, align 1
  %22 = mul nsw i64 %17, %11
  %a1 = ashr i64 %22, 2
  %a2 = add i64 %a1, %v1
  %a3 = add i64 %20, %a2
  %a4 = mul nsw i64 %a3, 5
  %23 = getelementptr i8, ptr %14, i64 %a4
  %a5 = load i8, ptr %23, align 1
  %a4_truncated = trunc i64 %a4 to i8
  %min = call i8 @llvm.smin.i8(i8 %a5, i8 %a4_truncated)
  %res = mul i8 %min, %a5
  store i8 %res, ptr %23, align 1
  %24 = add nuw nsw i64 %17, 1
  %25 = icmp eq i64 %24, %13
  br i1 %25, label %15, label %16
}

declare void @llvm.assume(i1 noundef) #1
declare i8 @llvm.smin.i8(i8, i8)

attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
