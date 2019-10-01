;
; There should be just a single copy of loop when strictest mutiplier candidates
; formula (unscaled candidates == 0) is enforced:
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=0 -unswitch-siblings-toplevel-div=1 \
; RUN:     -passes='loop(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=0 -unswitch-siblings-toplevel-div=8 \
; RUN:     -passes='loop(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; With relaxed candidates multiplier (unscaled candidates == 8) we should allow
; some unswitches to happen until siblings multiplier starts kicking in:
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=8 -unswitch-siblings-toplevel-div=1 \
; RUN:     -passes='loop(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP5
;
; With relaxed candidates multiplier (unscaled candidates == 8) and with relaxed
; siblings multiplier for top-level loops (toplevel-div == 8) we should get
;    2^(num conds) == 2^5 == 32
; copies of the loop:
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=8 -unswitch-siblings-toplevel-div=8 \
; RUN:     -passes='loop(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP32
;
; Similarly get
;    2^(num conds) == 2^5 == 32
; copies of the loop when cost multiplier is disabled:
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=false \
; RUN:     -passes='loop(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP32
;
;
; Single loop, not unswitched
; LOOP1:     Loop at depth 1 containing:
; LOOP1-NOT: Loop at depth 1 containing:

; 5 loops, unswitched 4 times
; LOOP5-COUNT-5: Loop at depth 1 containing:
; LOOP5-NOT:     Loop at depth 1 containing:

; 32 loops, fully unswitched
; LOOP32-COUNT-32: Loop at depth 1 containing:
; LOOP32-NOT:     Loop at depth 1 containing:

define void @loop_simple5(i32* %addr, i1 %c1, i1 %c2, i1 %c3, i1 %c4, i1 %c5) {
entry:
  br label %loop
loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop_latch]
  %iv.next = add i32 %iv, 1
  br i1 %c1, label %loop_next1, label %loop_next1_right
loop_next1_right:
  br label %loop_next1
loop_next1:
  br i1 %c2, label %loop_next2, label %loop_next2_right
loop_next2_right:
  br label %loop_next2
loop_next2:
  br i1 %c3, label %loop_next3, label %loop_next3_right
loop_next3_right:
  br label %loop_next3
loop_next3:
  br i1 %c4, label %loop_next4, label %loop_next4_right
loop_next4_right:
  br label %loop_next4
loop_next4:
  br i1 %c5, label %loop_latch, label %loop_latch_right
loop_latch_right:
  br label %loop_latch
loop_latch:
  store volatile i32 0, i32* %addr
  %test_loop = icmp slt i32 %iv, 50
  br i1 %test_loop, label %loop, label %exit
exit:
  ret void
}
