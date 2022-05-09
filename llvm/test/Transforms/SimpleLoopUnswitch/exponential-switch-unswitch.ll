;
; Here we have 5-way unswitchable switch with each successor also having an unswitchable
; exiting branch in it. If we start unswitching those branches we start duplicating the
; whole switch. This can easily lead to exponential behavior w/o proper control.
; On a real-life testcase there was 16-way switch and that took forever to compile w/o
; a cost control.
;
;
; When we use the stricted multiplier candidates formula (unscaled candidates == 0)
; we should be getting just a single loop.
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=0 -unswitch-siblings-toplevel-div=1 \
; RUN:     -passes='loop(simple-loop-unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=0 -unswitch-siblings-toplevel-div=16 \
; RUN:     -passes='loop(simple-loop-unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=0 -unswitch-siblings-toplevel-div=1 \
; RUN:     -passes='loop-mssa(simple-loop-unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=0 -unswitch-siblings-toplevel-div=16 \
; RUN:     -passes='loop-mssa(simple-loop-unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | FileCheck %s --check-prefixes=LOOP1
;
; With relaxed candidates multiplier (unscaled candidates == 8) we should allow
; some unswitches to happen until siblings multiplier starts kicking in:
;
; The tests below also run licm, because it is needed to hoist out
; loop-invariant freeze instructions, which otherwise may block further
; unswitching.
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=8 -unswitch-siblings-toplevel-div=1 \
; RUN:     -passes='loop-mssa(licm,simple-loop-unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | \
; RUN:     sort -b -k 1 | FileCheck %s --check-prefixes=LOOP-RELAX
;
; With relaxed candidates multiplier (unscaled candidates == 8) and with relaxed
; siblings multiplier for top-level loops (toplevel-div == 8) we should get
; considerably more copies of the loop (especially top-level ones).
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=true \
; RUN:     -unswitch-num-initial-unscaled-candidates=8 -unswitch-siblings-toplevel-div=8 \
; RUN:     -passes='loop-mssa(licm,simple-loop-unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | \
; RUN:     sort -b -k 1 | FileCheck %s --check-prefixes=LOOP-RELAX2
;
; We get hundreds of copies of the loop when cost multiplier is disabled:
;
; RUN: opt < %s -enable-unswitch-cost-multiplier=false \
; RUN:     -passes='loop-mssa(licm,simple-loop-unswitch<nontrivial>),print<loops>' -disable-output 2>&1 | \
; RUN:     sort -b -k 1 | FileCheck %s --check-prefixes=LOOP-MAX

; Single loop nest, not unswitched
; LOOP1:     Loop at depth 1 containing:
; LOOP1-NOT: Loop at depth 1 containing:
; LOOP1:     Loop at depth 2 containing:
; LOOP1-NOT: Loop at depth 2 containing:
;
; Somewhat relaxed restrictions on candidates:
; LOOP-RELAX-COUNT-5:     Loop at depth 1 containing:
; LOOP-RELAX-NOT: Loop at depth 1 containing:
; LOOP-RELAX-COUNT-32:     Loop at depth 2 containing:
; LOOP-RELAX-NOT: Loop at depth 2 containing:
;
; Even more relaxed restrictions on candidates and siblings.
; LOOP-RELAX2-COUNT-11:     Loop at depth 1 containing:
; LOOP-RELAX2-NOT: Loop at depth 1 containing:
; LOOP-RELAX2-COUNT-40:     Loop at depth 2 containing:
; LOOP-RELAX-NOT: Loop at depth 2 containing:
;
; Unswitched as much as it could (with multiplier disabled).
; LOOP-MAX-COUNT-56:     Loop at depth 1 containing:
; LOOP-MAX-NOT: Loop at depth 1 containing:
; LOOP-MAX-COUNT-111:     Loop at depth 2 containing:
; LOOP-MAX-NOT: Loop at depth 2 containing:

define i32 @loop_switch(i32* %addr, i32 %c1, i32 %c2) {
entry:
  %addr1 = getelementptr i32, i32* %addr, i64 0
  %addr2 = getelementptr i32, i32* %addr, i64 1
  %check0 = icmp eq i32 %c2, 0
  %check1 = icmp eq i32 %c2, 31
  %check2 = icmp eq i32 %c2, 32
  %check3 = icmp eq i32 %c2, 33
  %check4 = icmp eq i32 %c2, 34
  br label %outer_loop

outer_loop:
  %iv1 = phi i32 [0, %entry], [%iv1.next, %outer_latch]
  %iv1.next = add i32 %iv1, 1
  br label %inner_loop
inner_loop:
  %iv2 = phi i32 [0, %outer_loop], [%iv2.next, %inner_latch]
  %iv2.next = add i32 %iv2, 1
  switch i32 %c1, label %inner_latch [
    i32 0, label %case0
    i32 1, label %case1
    i32 2, label %case2
    i32 3, label %case3
    i32 4, label %case4
  ]

case4:
  br i1 %check4, label %exit, label %inner_latch
case3:
  br i1 %check3, label %exit, label %inner_latch
case2:
  br i1 %check2, label %exit, label %inner_latch
case1:
  br i1 %check1, label %exit, label %inner_latch
case0:
  br i1 %check0, label %exit, label %inner_latch

inner_latch:
  store volatile i32 0, i32* %addr1
  %test_inner = icmp slt i32 %iv2, 50
  br i1 %test_inner, label %inner_loop, label %outer_latch

outer_latch:
  store volatile i32 0, i32* %addr2
  %test_outer = icmp slt i32 %iv1, 50
  br i1 %test_outer, label %outer_loop, label %exit

exit:                                            ; preds = %bci_0
  ret i32 1
}
