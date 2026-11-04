; This test checks if two similar functions, @0 and @1, are not merged as they are unnamed.

; RUN: opt -mtriple=arm64-apple-darwin -S --passes=global-merge-func %s | FileCheck %s
; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true < %s | FileCheck %s

; CHECK-NOT: .Tgm

@g = external local_unnamed_addr global [0 x i32], align 4
@g1 = external global i32, align 4
@g2 = external global i32, align 4

define i32 @0(i32 %a) {
entry:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [0 x i32], ptr @g, i64 0, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %1 = load volatile i32, ptr @g1, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 1
  ret i32 %add
}

define i32 @1(i32 %a) {
entry:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [0 x i32], ptr @g, i64 0, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %1 = load volatile i32, ptr @g2, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 1
  ret i32 %add
}
