;
; Here all the branches are exiting ones. Checking that we dont have
; exponential behavior with any kind of controlling heuristics here.
;
; There we should have just a single loop.
;
; RUN: opt < %s -enable-nontrivial-unswitch -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=0 -unswitch-siblings-toplevel-div=1 \
; RUN:     -passes='loop(unswitch),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; RUN: opt < %s -enable-nontrivial-unswitch -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=0 -unswitch-siblings-toplevel-div=8 \
; RUN:     -passes='loop(unswitch),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; RUN: opt < %s -enable-nontrivial-unswitch -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=8 -unswitch-siblings-toplevel-div=1 \
; RUN:     -passes='loop(unswitch),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; RUN: opt < %s -enable-nontrivial-unswitch -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=8 -unswitch-siblings-toplevel-div=8 \
; RUN:     -passes='loop(unswitch),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; RUN: opt < %s -enable-nontrivial-unswitch -enable-unswitch-cost-multiplier=false \
; RUN:     -passes='loop(unswitch),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
;
; Single loop, not unswitched
; LOOP1:     Loop at depth 1 containing:
; LOOP1-NOT: Loop at depth 1 containing:

declare void @bar()

define void @loop_simple5(i32* %addr, i1 %c1, i1 %c2, i1 %c3, i1 %c4, i1 %c5) {
entry:
  br label %loop
loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop_latch]
  %iv.next = add i32 %iv, 1
  ;; disabling trivial unswitch
  call void @bar()
  br i1 %c1, label %loop_next1, label %exit
loop_next1:
  br i1 %c2, label %loop_next2, label %exit
loop_next2:
  br i1 %c3, label %loop_next3, label %exit
loop_next3:
  br i1 %c4, label %loop_next4, label %exit
loop_next4:
  br i1 %c5, label %loop_latch, label %exit
loop_latch:
  store volatile i32 0, i32* %addr
  %test_loop = icmp slt i32 %iv, 50
  br i1 %test_loop, label %loop, label %exit
exit:
  ret void
}
