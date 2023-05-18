;RUN: opt -S -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -mtriple=riscv32 < %s | FileCheck %s

; Test case taken from test/Transforms/SimplifyCFG/ARM/select-trunc-i64.ll.
; A correct implementation of isTruncateFree allows this test case to be
; reduced to a single basic block.

; CHECK-LABEL: select_trunc_i64
; CHECK-NOT: br
; CHECK: select
; CHECK: select
define i32 @select_trunc_i64(i32 %a, i32 %b) {
entry:
  %conv = sext i32 %a to i64
  %conv1 = sext i32 %b to i64
  %add = add nsw i64 %conv1, %conv
  %cmp = icmp sgt i64 %add, 2147483647
  br i1 %cmp, label %cond.end7, label %cond.false

cond.false:                                       ; preds = %entry
  %0 = icmp sgt i64 %add, -2147483648
  %cond = select i1 %0, i64 %add, i64 -2147483648
  %extract.t = trunc i64 %cond to i32
  br label %cond.end7

cond.end7:                                        ; preds = %cond.false, %entry
  %cond8.off0 = phi i32 [ 2147483647, %entry ], [ %extract.t, %cond.false ]
  ret i32 %cond8.off0
}
