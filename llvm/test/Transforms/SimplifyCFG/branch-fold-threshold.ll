; RUN: opt %s -simplifycfg -S | FileCheck %s --check-prefix=NORMAL
; RUN: opt %s -simplifycfg -S -bonus-inst-threshold=2 | FileCheck %s --check-prefix=AGGRESSIVE
; RUN: opt %s -simplifycfg -S -bonus-inst-threshold=4 | FileCheck %s --check-prefix=WAYAGGRESSIVE
; RUN: opt %s -passes=simplify-cfg -S | FileCheck %s --check-prefix=NORMAL
; RUN: opt %s -passes='simplify-cfg<bonus-inst-threshold=2>' -S | FileCheck %s --check-prefix=AGGRESSIVE
; RUN: opt %s -passes='simplify-cfg<bonus-inst-threshold=4>' -S | FileCheck %s --check-prefix=WAYAGGRESSIVE

define i32 @foo(i32 %a, i32 %b, i32 %c, i32 %d, i32* %input) {
; NORMAL-LABEL: @foo(
; AGGRESSIVE-LABEL: @foo(
entry:
  %cmp = icmp sgt i32 %d, 3
  br i1 %cmp, label %cond.end, label %lor.lhs.false
; NORMAL: br i1
; AGGRESSIVE: br i1

lor.lhs.false:
  %mul = shl i32 %c, 1
  %add = add nsw i32 %mul, %a
  %cmp1 = icmp slt i32 %add, %b
  br i1 %cmp1, label %cond.false, label %cond.end
; NORMAL: br i1
; AGGRESSIVE-NOT: br i1

cond.false:
  %0 = load i32, i32* %input, align 4
  br label %cond.end

cond.end:
  %cond = phi i32 [ %0, %cond.false ], [ 0, %lor.lhs.false ], [ 0, %entry ]
  ret i32 %cond
}

declare void @distinct_a();
declare void @distinct_b();

;; Like foo, but have to duplicate into multiple predecessors
define i32 @bar(i32 %a, i32 %b, i32 %c, i32 %d, i32* %input) {
; NORMAL-LABEL: @bar(
; AGGRESSIVE-LABEL: @bar(
entry:
  %cmp_split = icmp slt i32 %d, %b
  %cmp = icmp sgt i32 %d, 3
  br i1 %cmp_split, label %pred_a, label %pred_b
pred_a:
; NORMAL-LABEL: pred_a:
; AGGRESSIVE-LABEL: pred_a:
; WAYAGGRESSIVE-LABEL: pred_a:
; NORMAL: br i1
; AGGRESSIVE: br i1
; WAYAGGRESSIVE: br i1
  call void @distinct_a();
  br i1 %cmp, label %cond.end, label %lor.lhs.false
pred_b:
; NORMAL-LABEL: pred_b:
; AGGRESSIVE-LABEL: pred_b:
; WAYAGGRESSIVE-LABEL: pred_b:
; NORMAL: br i1
; AGGRESSIVE: br i1
; WAYAGGRESSIVE: br i1
  call void @distinct_b();
  br i1 %cmp, label %cond.end, label %lor.lhs.false

lor.lhs.false:
  %mul = shl i32 %c, 1
  %add = add nsw i32 %mul, %a
  %cmp1 = icmp slt i32 %add, %b
  br i1 %cmp1, label %cond.false, label %cond.end
; NORMAL-LABEL: lor.lhs.false:
; AGGRESIVE-LABEL: lor.lhs.false:
; WAYAGGRESIVE-LABEL: lor.lhs.false:
; NORMAL: br i1
; AGGRESSIVE: br i1
; WAYAGGRESSIVE-NOT: br i1

cond.false:
  %0 = load i32, i32* %input, align 4
  br label %cond.end

cond.end:
  %cond = phi i32 [ %0, %cond.false ], [ 0, %lor.lhs.false ],[ 0, %pred_a ],[ 0, %pred_b ]
  ret i32 %cond
}
