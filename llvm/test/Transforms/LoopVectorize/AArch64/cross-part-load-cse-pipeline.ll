; The redundancy analysis changes only the interleave count. This test
; demonstrates that the standard O3 pipeline can realize the motivating
; opportunity, without making downstream elimination part of the contract.
;
; RUN: opt -passes='default<O3>' -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -enable-interleave-cse -interleave-cse-min-pct=1 \
; RUN:     -S %s | FileCheck %s --check-prefix=ENABLED
; RUN: opt -passes='default<O3>' -force-vector-width=4 \
; RUN:     -force-target-max-vector-interleave=2 -small-loop-cost=0 \
; RUN:     -S %s | FileCheck %s --check-prefix=DISABLED

target triple = "aarch64-unknown-linux-gnu"

; With the analysis enabled, IC=2 exposes one redundant vector load to the O3
; pipeline. The vector body processes eight source iterations with three loads.
; With the analysis disabled, IC=1 processes four iterations with two loads.
define void @positive(ptr noalias %a, ptr noalias %c, i64 %n) {
; ENABLED-LABEL: define void @positive(
; ENABLED:       vector.body:
; ENABLED-COUNT-3: load <4 x i32>
; ENABLED-NOT:   load <4 x i32>
; ENABLED:       add nuw i64 {{.*}}, 8
;
; DISABLED-LABEL: define void @positive(
; DISABLED:       vector.body:
; DISABLED-COUNT-2: load <4 x i32>
; DISABLED-NOT:   load <4 x i32>
; DISABLED:       add nuw i64 {{.*}}, 4
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %a.iv = getelementptr inbounds i32, ptr %a, i64 %iv
  %l1 = load i32, ptr %a.iv, align 4
  %iv.plus.4 = add nuw nsw i64 %iv, 4
  %a.iv.plus.4 = getelementptr inbounds i32, ptr %a, i64 %iv.plus.4
  %l2 = load i32, ptr %a.iv.plus.4, align 4
  %sum = add i32 %l1, %l2
  %c.iv = getelementptr inbounds i32, ptr %c, i64 %iv
  store i32 %sum, ptr %c.iv, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
