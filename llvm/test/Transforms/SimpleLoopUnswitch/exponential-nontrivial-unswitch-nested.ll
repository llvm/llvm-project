;
; There should be just a single copy of each loop when strictest mutiplier
; candidates formula (unscaled candidates == 0) is enforced:

; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=0 -unswitch-siblings-toplevel-div=1 \
; RUN:     -passes='loop(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=0 -unswitch-siblings-toplevel-div=16 \
; RUN:     -passes='loop(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=0 -unswitch-siblings-toplevel-div=1 \
; RUN:     -passes='loop-mssa(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=0 -unswitch-siblings-toplevel-div=16 \
; RUN:     -passes='loop-mssa(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; When we relax the candidates part of a multiplier formula
; (unscaled candidates == 4) we start getting  some unswitches,
; which leads to siblings multiplier kicking in.
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=4 -unswitch-siblings-toplevel-div=1 \
; RUN:     -passes='loop(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | \
; RUN:     sort -b -k 1 | FileCheck %s --check-prefixes=LOOP-UNSCALE4-DIV1
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=4 -unswitch-siblings-toplevel-div=1 \
; RUN:     -passes='loop-mssa(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | \
; RUN:     sort -b -k 1 | FileCheck %s --check-prefixes=LOOP-UNSCALE4-DIV1
;
; NB: sort -b is essential here and below, otherwise blanks might lead to different
; order depending on locale.
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=4 -unswitch-siblings-toplevel-div=2 \
; RUN:     -passes='loop(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | \
; RUN:     sort -b -k 1 | FileCheck %s --check-prefixes=LOOP-UNSCALE4-DIV2
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=4 -unswitch-siblings-toplevel-div=2 \
; RUN:     -passes='loop-mssa(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | \
; RUN:     sort -b -k 1 | FileCheck %s --check-prefixes=LOOP-UNSCALE4-DIV2
;
; Get
;    2^(num conds) == 2^5 = 32
; loop nests when cost multiplier is disabled:
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=false \
; RUN:     -passes='loop(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | \
; RUN:	   sort -b -k 1 | FileCheck %s --check-prefixes=LOOP32
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=false \
; RUN:     -passes='loop-mssa(unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | \
; RUN:	   sort -b -k 1 | FileCheck %s --check-prefixes=LOOP32
;
; Single loop nest, not unswitched
; LOOP1:     Loop at depth 1 containing:
; LOOP1:     Loop at depth 2 containing:
; LOOP1:     Loop at depth 3 containing:
; LOOP1-NOT: Loop at depth {{[0-9]+}} containing:
;
; Half unswitched loop nests, with unscaled4 and div1 it gets less depth1 loops unswitched
; since they have more cost.
; LOOP-UNSCALE4-DIV1-COUNT-6: Loop at depth 1 containing:
; LOOP-UNSCALE4-DIV1-COUNT-19: Loop at depth 2 containing:
; LOOP-UNSCALE4-DIV1-COUNT-29: Loop at depth 3 containing:
; LOOP-UNSCALE4-DIV1-NOT:      Loop at depth {{[0-9]+}} containing:
;
; Half unswitched loop nests, with unscaled4 and div2 it gets more depth1 loops unswitched
; as div2 kicks in.
; LOOP-UNSCALE4-DIV2-COUNT-11: Loop at depth 1 containing:
; LOOP-UNSCALE4-DIV2-COUNT-22: Loop at depth 2 containing:
; LOOP-UNSCALE4-DIV2-COUNT-29: Loop at depth 3 containing:
; LOOP-UNSCALE4-DIV2-NOT:      Loop at depth {{[0-9]+}} containing:
;
; 32 loop nests, fully unswitched
; LOOP32-COUNT-32: Loop at depth 1 containing:
; LOOP32-COUNT-32: Loop at depth 2 containing:
; LOOP32-COUNT-32: Loop at depth 3 containing:
; LOOP32-NOT:      Loop at depth {{[0-9]+}} containing:

declare void @bar()

define void @loop_nested3_conds5(i32* %addr, i1 %c1i, i1 %c2i, i1 %c3i, i1 %c4i, i1 %c5i) {
entry:
  ; c1 ~ c5 are guaranteed to be never undef or poison.
  %c1 = freeze i1 %c1i
  %c2 = freeze i1 %c2i
  %c3 = freeze i1 %c3i
  %c4 = freeze i1 %c4i
  %c5 = freeze i1 %c5i
  %addr1 = getelementptr i32, i32* %addr, i64 0
  %addr2 = getelementptr i32, i32* %addr, i64 1
  %addr3 = getelementptr i32, i32* %addr, i64 2
  br label %outer
outer:
  %iv1 = phi i32 [0, %entry], [%iv1.next, %outer_latch]
  %iv1.next = add i32 %iv1, 1
  ;; skip nontrivial unswitch
  call void @bar()
  br label %middle
middle:
  %iv2 = phi i32 [0, %outer], [%iv2.next, %middle_latch]
  %iv2.next = add i32 %iv2, 1
  ;; skip nontrivial unswitch
  call void @bar()
  br label %loop
loop:
  %iv3 = phi i32 [0, %middle], [%iv3.next, %loop_latch]
  %iv3.next = add i32 %iv3, 1
  ;; skip nontrivial unswitch
  call void @bar()
  br i1 %c1, label %loop_next1_left, label %loop_next1_right
loop_next1_left:
  br label %loop_next1
loop_next1_right:
  br label %loop_next1

loop_next1:
  br i1 %c2, label %loop_next2_left, label %loop_next2_right
loop_next2_left:
  br label %loop_next2
loop_next2_right:
  br label %loop_next2

loop_next2:
  br i1 %c3, label %loop_next3_left, label %loop_next3_right
loop_next3_left:
  br label %loop_next3
loop_next3_right:
  br label %loop_next3

loop_next3:
  br i1 %c4, label %loop_next4_left, label %loop_next4_right
loop_next4_left:
  br label %loop_next4
loop_next4_right:
  br label %loop_next4

loop_next4:
  br i1 %c5, label %loop_latch_left, label %loop_latch_right
loop_latch_left:
  br label %loop_latch
loop_latch_right:
  br label %loop_latch

loop_latch:
  store volatile i32 0, i32* %addr1
  %test_loop = icmp slt i32 %iv3, 50
  br i1 %test_loop, label %loop, label %middle_latch
middle_latch:
  store volatile i32 0, i32* %addr2
  %test_middle = icmp slt i32 %iv2, 50
  br i1 %test_middle, label %middle, label %outer_latch
outer_latch:
  store volatile i32 0, i32* %addr3
  %test_outer = icmp slt i32 %iv1, 50
  br i1 %test_outer, label %outer, label %exit
exit:
  ret void
}
