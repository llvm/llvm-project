; Test that -riscv-sched-mispredict-penalty overrides the scheduler model's
; MispredictPenalty field, and that the override affects code generation
; decisions that depend on MispredictPenalty (specifically SelectOptimize,
; which weighs the cost of branch mispredictions against the benefit of
; out-of-order execution when deciding whether to convert select instructions
; to conditional branches).
;
; sifive-p550 is an out-of-order core with SelectOptimize enabled. For a
; select in an inner loop, SelectOptimize calculates:
;
;   BranchCost = PredictedPathCost + MispredictCost
;   MispredictCost = max(MispredictPenalty, CondCost) * MispredictRate / 100
;
; When MispredictCost is very large (high penalty), BranchCost >> SelectCost
; and the select is kept as-is. When MispredictCost is small (low penalty),
; BranchCost < SelectCost and the select is converted to a conditional branch.
;
; The -select-opti-loop-cycle-gain-threshold=1 flag lowers the minimum
; absolute gain requirement so that the gain between the two cases is visible.
; It does not affect which direction MispredictPenalty pushes the decision.
;
; RUN: opt -passes='require<profile-summary>,function(select-optimize)' \
; RUN:   -mtriple=riscv64 -mcpu=sifive-p550 \
; RUN:   -riscv-sched-mispredict-penalty=10000 \
; RUN:   -select-opti-loop-cycle-gain-threshold=1 \
; RUN:   -S < %s \
; RUN:   | FileCheck %s --check-prefix=HIGHPENALTY
; RUN: opt -passes='require<profile-summary>,function(select-optimize)' \
; RUN:   -mtriple=riscv64 -mcpu=sifive-p550 \
; RUN:   -riscv-sched-mispredict-penalty=0 \
; RUN:   -select-opti-loop-cycle-gain-threshold=1 \
; RUN:   -S < %s \
; RUN:   | FileCheck %s --check-prefix=LOWPENALTY

; With a very high mispredict penalty (10000 cycles), BranchCost is dominated
; by misprediction cost. SelectOptimize keeps the select instruction because
; the misprediction risk makes branches unprofitable.
; HIGHPENALTY-LABEL: @sum_filtered(
; HIGHPENALTY:         %cond = icmp slt i64 %v, 0
; HIGHPENALTY-NEXT:    %sel = select i1 %cond, i64 %v, i64 0

; With a zero mispredict penalty, branch mispredictions carry no cost.
; BranchCost (= PredictedPathCost only) is less than SelectCost (which
; must speculatively compute both paths). The select is converted to a
; conditional branch + phi.
; LOWPENALTY-LABEL: @sum_filtered(
; LOWPENALTY:          %cond = icmp slt i64 %v, 0
; LOWPENALTY-NEXT:     %cond.frozen = freeze i1 %cond
; LOWPENALTY-NEXT:     br i1 %cond.frozen, label %select.end, label %select.false

define i64 @sum_filtered(ptr %p, i64 %n) {
entry:
  %entry.cmp = icmp sgt i64 %n, 0
  br i1 %entry.cmp, label %loop.ph, label %exit

loop.ph:
  br label %loop

loop:
  %i = phi i64 [ 0, %loop.ph ], [ %i.next, %loop ]
  %acc = phi i64 [ 0, %loop.ph ], [ %acc.next, %loop ]
  %ptr = getelementptr i64, ptr %p, i64 %i
  %v = load i64, ptr %ptr
  %cond = icmp slt i64 %v, 0
  %sel = select i1 %cond, i64 %v, i64 0
  %acc.next = add i64 %acc, %sel
  %i.next = add i64 %i, 1
  %cmp = icmp ne i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  %result = phi i64 [ 0, %entry ], [ %acc.next, %loop ]
  ret i64 %result
}
