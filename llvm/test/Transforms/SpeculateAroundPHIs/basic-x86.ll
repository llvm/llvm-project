; Test the basic functionality of speculating around PHI nodes based on reduced
; cost of the constant operands to the PHI nodes using the x86 cost model.
;
; REQUIRES: x86-registered-target
; RUN: opt -S -passes=spec-phis < %s | FileCheck %s

target triple = "x86_64-unknown-unknown"

define i32 @test_basic(i1 %flag, i32 %arg) {
; CHECK-LABEL: define i32 @test_basic(
entry:
  br i1 %flag, label %a, label %b
; CHECK:         br i1 %flag, label %a, label %b

a:
  br label %exit
; CHECK:       a:
; CHECK-NEXT:    %[[SUM_A:.*]] = add i32 %arg, 7
; CHECK-NEXT:    br label %exit

b:
  br label %exit
; CHECK:       b:
; CHECK-NEXT:    %[[SUM_B:.*]] = add i32 %arg, 11
; CHECK-NEXT:    br label %exit

exit:
  %p = phi i32 [ 7, %a ], [ 11, %b ]
  %sum = add i32 %arg, %p
  ret i32 %sum
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ %[[SUM_A]], %a ], [ %[[SUM_B]], %b ]
; CHECK-NEXT:    ret i32 %[[PHI]]
}

; Check that we handle commuted operands and get the constant onto the RHS.
define i32 @test_commuted(i1 %flag, i32 %arg) {
; CHECK-LABEL: define i32 @test_commuted(
entry:
  br i1 %flag, label %a, label %b
; CHECK:         br i1 %flag, label %a, label %b

a:
  br label %exit
; CHECK:       a:
; CHECK-NEXT:    %[[SUM_A:.*]] = add i32 %arg, 7
; CHECK-NEXT:    br label %exit

b:
  br label %exit
; CHECK:       b:
; CHECK-NEXT:    %[[SUM_B:.*]] = add i32 %arg, 11
; CHECK-NEXT:    br label %exit

exit:
  %p = phi i32 [ 7, %a ], [ 11, %b ]
  %sum = add i32 %p, %arg
  ret i32 %sum
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ %[[SUM_A]], %a ], [ %[[SUM_B]], %b ]
; CHECK-NEXT:    ret i32 %[[PHI]]
}

define i32 @test_split_crit_edge(i1 %flag, i32 %arg) {
; CHECK-LABEL: define i32 @test_split_crit_edge(
entry:
  br i1 %flag, label %exit, label %a
; CHECK:       entry:
; CHECK-NEXT:    br i1 %flag, label %[[ENTRY_SPLIT:.*]], label %a
;
; CHECK:       [[ENTRY_SPLIT]]:
; CHECK-NEXT:    %[[SUM_ENTRY_SPLIT:.*]] = add i32 %arg, 7
; CHECK-NEXT:    br label %exit

a:
  br label %exit
; CHECK:       a:
; CHECK-NEXT:    %[[SUM_A:.*]] = add i32 %arg, 11
; CHECK-NEXT:    br label %exit

exit:
  %p = phi i32 [ 7, %entry ], [ 11, %a ]
  %sum = add i32 %arg, %p
  ret i32 %sum
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ %[[SUM_ENTRY_SPLIT]], %[[ENTRY_SPLIT]] ], [ %[[SUM_A]], %a ]
; CHECK-NEXT:    ret i32 %[[PHI]]
}

define i32 @test_no_spec_dominating_inst(i1 %flag, i32* %ptr) {
; CHECK-LABEL: define i32 @test_no_spec_dominating_inst(
entry:
  %load = load i32, i32* %ptr
  br i1 %flag, label %a, label %b
; CHECK:         %[[LOAD:.*]] = load i32, i32* %ptr
; CHECK-NEXT:    br i1 %flag, label %a, label %b

a:
  br label %exit
; CHECK:       a:
; CHECK-NEXT:    %[[SUM_A:.*]] = add i32 %[[LOAD]], 7
; CHECK-NEXT:    br label %exit

b:
  br label %exit
; CHECK:       b:
; CHECK-NEXT:    %[[SUM_B:.*]] = add i32 %[[LOAD]], 11
; CHECK-NEXT:    br label %exit

exit:
  %p = phi i32 [ 7, %a ], [ 11, %b ]
  %sum = add i32 %load, %p
  ret i32 %sum
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ %[[SUM_A]], %a ], [ %[[SUM_B]], %b ]
; CHECK-NEXT:    ret i32 %[[PHI]]
}

; We have special logic handling PHI nodes, make sure it doesn't get confused
; by a dominating PHI.
define i32 @test_no_spec_dominating_phi(i1 %flag1, i1 %flag2, i32 %x, i32 %y) {
; CHECK-LABEL: define i32 @test_no_spec_dominating_phi(
entry:
  br i1 %flag1, label %x.block, label %y.block
; CHECK:       entry:
; CHECK-NEXT:    br i1 %flag1, label %x.block, label %y.block

x.block:
  br label %merge
; CHECK:       x.block:
; CHECK-NEXT:    br label %merge

y.block:
  br label %merge
; CHECK:       y.block:
; CHECK-NEXT:    br label %merge

merge:
  %xy.phi = phi i32 [ %x, %x.block ], [ %y, %y.block ]
  br i1 %flag2, label %a, label %b
; CHECK:       merge:
; CHECK-NEXT:    %[[XY_PHI:.*]] = phi i32 [ %x, %x.block ], [ %y, %y.block ]
; CHECK-NEXT:    br i1 %flag2, label %a, label %b

a:
  br label %exit
; CHECK:       a:
; CHECK-NEXT:    %[[SUM_A:.*]] = add i32 %[[XY_PHI]], 7
; CHECK-NEXT:    br label %exit

b:
  br label %exit
; CHECK:       b:
; CHECK-NEXT:    %[[SUM_B:.*]] = add i32 %[[XY_PHI]], 11
; CHECK-NEXT:    br label %exit

exit:
  %p = phi i32 [ 7, %a ], [ 11, %b ]
  %sum = add i32 %xy.phi, %p
  ret i32 %sum
; CHECK:       exit:
; CHECK-NEXT:    %[[SUM_PHI:.*]] = phi i32 [ %[[SUM_A]], %a ], [ %[[SUM_B]], %b ]
; CHECK-NEXT:    ret i32 %[[SUM_PHI]]
}

; Ensure that we will speculate some number of "free" instructions on the given
; architecture even though they are unrelated to the PHI itself.
define i32 @test_speculate_free_insts(i1 %flag, i64 %arg) {
; CHECK-LABEL: define i32 @test_speculate_free_insts(
entry:
  br i1 %flag, label %a, label %b
; CHECK:         br i1 %flag, label %a, label %b

a:
  br label %exit
; CHECK:       a:
; CHECK-NEXT:    %[[T1_A:.*]] = trunc i64 %arg to i48
; CHECK-NEXT:    %[[T2_A:.*]] = trunc i48 %[[T1_A]] to i32
; CHECK-NEXT:    %[[SUM_A:.*]] = add i32 %[[T2_A]], 7
; CHECK-NEXT:    br label %exit

b:
  br label %exit
; CHECK:       b:
; CHECK-NEXT:    %[[T1_B:.*]] = trunc i64 %arg to i48
; CHECK-NEXT:    %[[T2_B:.*]] = trunc i48 %[[T1_B]] to i32
; CHECK-NEXT:    %[[SUM_B:.*]] = add i32 %[[T2_B]], 11
; CHECK-NEXT:    br label %exit

exit:
  %p = phi i32 [ 7, %a ], [ 11, %b ]
  %t1 = trunc i64 %arg to i48
  %t2 = trunc i48 %t1 to i32
  %sum = add i32 %t2, %p
  ret i32 %sum
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ %[[SUM_A]], %a ], [ %[[SUM_B]], %b ]
; CHECK-NEXT:    ret i32 %[[PHI]]
}

define i32 @test_speculate_free_phis(i1 %flag, i32 %arg1, i32 %arg2) {
; CHECK-LABEL: define i32 @test_speculate_free_phis(
entry:
  br i1 %flag, label %a, label %b
; CHECK:         br i1 %flag, label %a, label %b

a:
  br label %exit
; CHECK:       a:
; CHECK-NEXT:    %[[SUM_A:.*]] = add i32 %arg1, 7
; CHECK-NEXT:    br label %exit

b:
  br label %exit
; CHECK:       b:
; CHECK-NEXT:    %[[SUM_B:.*]] = add i32 %arg2, 11
; CHECK-NEXT:    br label %exit

exit:
  %p1 = phi i32 [ 7, %a ], [ 11, %b ]
  %p2 = phi i32 [ %arg1, %a ], [ %arg2, %b ]
  %sum = add i32 %p2, %p1
  ret i32 %sum
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ %[[SUM_A]], %a ], [ %[[SUM_B]], %b ]
; We don't DCE the now unused PHI node...
; CHECK-NEXT:    %{{.*}} = phi i32 [ %arg1, %a ], [ %arg2, %b ]
; CHECK-NEXT:    ret i32 %[[PHI]]
}

; We shouldn't speculate multiple uses even if each individually looks
; profitable because of the total cost.
define i32 @test_no_spec_multi_uses(i1 %flag, i32 %arg1, i32 %arg2, i32 %arg3) {
; CHECK-LABEL: define i32 @test_no_spec_multi_uses(
entry:
  br i1 %flag, label %a, label %b
; CHECK:         br i1 %flag, label %a, label %b

a:
  br label %exit
; CHECK:       a:
; CHECK-NEXT:    br label %exit

b:
  br label %exit
; CHECK:       b:
; CHECK-NEXT:    br label %exit

exit:
  %p = phi i32 [ 7, %a ], [ 11, %b ]
  %add1 = add i32 %arg1, %p
  %add2 = add i32 %arg2, %p
  %add3 = add i32 %arg3, %p
  %sum1 = add i32 %add1, %add2
  %sum2 = add i32 %sum1, %add3
  ret i32 %sum2
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ 7, %a ], [ 11, %b ]
; CHECK-NEXT:    %[[ADD1:.*]] = add i32 %arg1, %[[PHI]]
; CHECK-NEXT:    %[[ADD2:.*]] = add i32 %arg2, %[[PHI]]
; CHECK-NEXT:    %[[ADD3:.*]] = add i32 %arg3, %[[PHI]]
; CHECK-NEXT:    %[[SUM1:.*]] = add i32 %[[ADD1]], %[[ADD2]]
; CHECK-NEXT:    %[[SUM2:.*]] = add i32 %[[SUM1]], %[[ADD3]]
; CHECK-NEXT:    ret i32 %[[SUM2]]
}

define i32 @test_multi_phis1(i1 %flag, i32 %arg) {
; CHECK-LABEL: define i32 @test_multi_phis1(
entry:
  br i1 %flag, label %a, label %b
; CHECK:         br i1 %flag, label %a, label %b

a:
  br label %exit
; CHECK:       a:
; CHECK-NEXT:    %[[SUM_A1:.*]] = add i32 %arg, 1
; CHECK-NEXT:    %[[SUM_A2:.*]] = add i32 %[[SUM_A1]], 3
; CHECK-NEXT:    %[[SUM_A3:.*]] = add i32 %[[SUM_A2]], 5
; CHECK-NEXT:    br label %exit

b:
  br label %exit
; CHECK:       b:
; CHECK-NEXT:    %[[SUM_B1:.*]] = add i32 %arg, 2
; CHECK-NEXT:    %[[SUM_B2:.*]] = add i32 %[[SUM_B1]], 4
; CHECK-NEXT:    %[[SUM_B3:.*]] = add i32 %[[SUM_B2]], 6
; CHECK-NEXT:    br label %exit

exit:
  %p1 = phi i32 [ 1, %a ], [ 2, %b ]
  %p2 = phi i32 [ 3, %a ], [ 4, %b ]
  %p3 = phi i32 [ 5, %a ], [ 6, %b ]
  %sum1 = add i32 %arg, %p1
  %sum2 = add i32 %sum1, %p2
  %sum3 = add i32 %sum2, %p3
  ret i32 %sum3
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ %[[SUM_A3]], %a ], [ %[[SUM_B3]], %b ]
; CHECK-NEXT:    ret i32 %[[PHI]]
}

; Check that the order of the PHIs doesn't impact the behavior.
define i32 @test_multi_phis2(i1 %flag, i32 %arg) {
; CHECK-LABEL: define i32 @test_multi_phis2(
entry:
  br i1 %flag, label %a, label %b
; CHECK:         br i1 %flag, label %a, label %b

a:
  br label %exit
; CHECK:       a:
; CHECK-NEXT:    %[[SUM_A1:.*]] = add i32 %arg, 1
; CHECK-NEXT:    %[[SUM_A2:.*]] = add i32 %[[SUM_A1]], 3
; CHECK-NEXT:    %[[SUM_A3:.*]] = add i32 %[[SUM_A2]], 5
; CHECK-NEXT:    br label %exit

b:
  br label %exit
; CHECK:       b:
; CHECK-NEXT:    %[[SUM_B1:.*]] = add i32 %arg, 2
; CHECK-NEXT:    %[[SUM_B2:.*]] = add i32 %[[SUM_B1]], 4
; CHECK-NEXT:    %[[SUM_B3:.*]] = add i32 %[[SUM_B2]], 6
; CHECK-NEXT:    br label %exit

exit:
  %p3 = phi i32 [ 5, %a ], [ 6, %b ]
  %p2 = phi i32 [ 3, %a ], [ 4, %b ]
  %p1 = phi i32 [ 1, %a ], [ 2, %b ]
  %sum1 = add i32 %arg, %p1
  %sum2 = add i32 %sum1, %p2
  %sum3 = add i32 %sum2, %p3
  ret i32 %sum3
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ %[[SUM_A3]], %a ], [ %[[SUM_B3]], %b ]
; CHECK-NEXT:    ret i32 %[[PHI]]
}

define i32 @test_no_spec_indirectbr(i1 %flag, i32 %arg) {
; CHECK-LABEL: define i32 @test_no_spec_indirectbr(
entry:
  br i1 %flag, label %a, label %b
; CHECK:       entry:
; CHECK-NEXT:    br i1 %flag, label %a, label %b

a:
  indirectbr i8* undef, [label %exit]
; CHECK:       a:
; CHECK-NEXT:    indirectbr i8* undef, [label %exit]

b:
  indirectbr i8* undef, [label %exit]
; CHECK:       b:
; CHECK-NEXT:    indirectbr i8* undef, [label %exit]

exit:
  %p = phi i32 [ 7, %a ], [ 11, %b ]
  %sum = add i32 %arg, %p
  ret i32 %sum
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ 7, %a ], [ 11, %b ]
; CHECK-NEXT:    %[[SUM:.*]] = add i32 %arg, %[[PHI]]
; CHECK-NEXT:    ret i32 %[[SUM]]
}

declare void @g()

declare i32 @__gxx_personality_v0(...)

; FIXME: We should be able to handle this case -- only the exceptional edge is
; impossible to split.
define i32 @test_no_spec_invoke_continue(i1 %flag, i32 %arg) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: define i32 @test_no_spec_invoke_continue(
entry:
  br i1 %flag, label %a, label %b
; CHECK:       entry:
; CHECK-NEXT:    br i1 %flag, label %a, label %b

a:
  invoke void @g()
          to label %exit unwind label %lpad
; CHECK:       a:
; CHECK-NEXT:    invoke void @g()
; CHECK-NEXT:            to label %exit unwind label %lpad

b:
  invoke void @g()
          to label %exit unwind label %lpad
; CHECK:       b:
; CHECK-NEXT:    invoke void @g()
; CHECK-NEXT:            to label %exit unwind label %lpad

exit:
  %p = phi i32 [ 7, %a ], [ 11, %b ]
  %sum = add i32 %arg, %p
  ret i32 %sum
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ 7, %a ], [ 11, %b ]
; CHECK-NEXT:    %[[SUM:.*]] = add i32 %arg, %[[PHI]]
; CHECK-NEXT:    ret i32 %[[SUM]]

lpad:
  %lp = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef
}

define i32 @test_no_spec_landingpad(i32 %arg, i32* %ptr) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: define i32 @test_no_spec_landingpad(
entry:
  invoke void @g()
          to label %invoke.cont unwind label %lpad
; CHECK:       entry:
; CHECK-NEXT:    invoke void @g()
; CHECK-NEXT:            to label %invoke.cont unwind label %lpad

invoke.cont:
  invoke void @g()
          to label %exit unwind label %lpad
; CHECK:       invoke.cont:
; CHECK-NEXT:    invoke void @g()
; CHECK-NEXT:            to label %exit unwind label %lpad

lpad:
  %p = phi i32 [ 7, %entry ], [ 11, %invoke.cont ]
  %lp = landingpad { i8*, i32 }
          cleanup
  %sum = add i32 %arg, %p
  store i32 %sum, i32* %ptr
  resume { i8*, i32 } undef
; CHECK:       lpad:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ 7, %entry ], [ 11, %invoke.cont ]

exit:
  ret i32 0
}

declare i32 @__CxxFrameHandler3(...)

define i32 @test_no_spec_cleanuppad(i32 %arg, i32* %ptr) personality i32 (...)* @__CxxFrameHandler3 {
; CHECK-LABEL: define i32 @test_no_spec_cleanuppad(
entry:
  invoke void @g()
          to label %invoke.cont unwind label %lpad
; CHECK:       entry:
; CHECK-NEXT:    invoke void @g()
; CHECK-NEXT:            to label %invoke.cont unwind label %lpad

invoke.cont:
  invoke void @g()
          to label %exit unwind label %lpad
; CHECK:       invoke.cont:
; CHECK-NEXT:    invoke void @g()
; CHECK-NEXT:            to label %exit unwind label %lpad

lpad:
  %p = phi i32 [ 7, %entry ], [ 11, %invoke.cont ]
  %cp = cleanuppad within none []
  %sum = add i32 %arg, %p
  store i32 %sum, i32* %ptr
  cleanupret from %cp unwind to caller
; CHECK:       lpad:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ 7, %entry ], [ 11, %invoke.cont ]

exit:
  ret i32 0
}

; Check that we don't fall over when confronted with seemingly reasonable code
; for us to handle but in an unreachable region and with non-PHI use-def
; cycles.
define i32 @test_unreachable_non_phi_cycles(i1 %flag, i32 %arg) {
; CHECK-LABEL: define i32 @test_unreachable_non_phi_cycles(
entry:
  ret i32 42
; CHECK:       entry:
; CHECK-NEXT:    ret i32 42

a:
  br label %exit
; CHECK:       a:
; CHECK-NEXT:    br label %exit

b:
  br label %exit
; CHECK:       b:
; CHECK-NEXT:    br label %exit

exit:
  %p = phi i32 [ 7, %a ], [ 11, %b ]
  %zext = zext i32 %sum to i64
  %trunc = trunc i64 %zext to i32
  %sum = add i32 %trunc, %p
  br i1 %flag, label %a, label %b
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i32 [ 7, %a ], [ 11, %b ]
; CHECK-NEXT:    %[[ZEXT:.*]] = zext i32 %[[SUM:.*]] to i64
; CHECK-NEXT:    %[[TRUNC:.*]] = trunc i64 %[[ZEXT]] to i32
; CHECK-NEXT:    %[[SUM]] = add i32 %[[TRUNC]], %[[PHI]]
; CHECK-NEXT:    br i1 %flag, label %a, label %b
}

; Check that we don't speculate in the face of an expensive immediate. There
; are two reasons this should never speculate. First, even a local analysis
; should fail because it makes some paths (%a) potentially more expensive due
; to multiple uses of the immediate. Additionally, when we go to speculate the
; instructions, their cost will also be too high.
; FIXME: The goal is really to test the first property, but there doesn't
; happen to be any way to use free-to-speculate instructions here so that it
; would be the only interesting property.
define i64 @test_expensive_imm(i32 %flag, i64 %arg) {
; CHECK-LABEL: define i64 @test_expensive_imm(
entry:
  switch i32 %flag, label %a [
    i32 1, label %b
    i32 2, label %c
    i32 3, label %d
  ]
; CHECK:         switch i32 %flag, label %a [
; CHECK-NEXT:      i32 1, label %b
; CHECK-NEXT:      i32 2, label %c
; CHECK-NEXT:      i32 3, label %d
; CHECK-NEXT:    ]

a:
  br label %exit
; CHECK:       a:
; CHECK-NEXT:    br label %exit

b:
  br label %exit
; CHECK:       b:
; CHECK-NEXT:    br label %exit

c:
  br label %exit
; CHECK:       c:
; CHECK-NEXT:    br label %exit

d:
  br label %exit
; CHECK:       d:
; CHECK-NEXT:    br label %exit

exit:
  %p = phi i64 [ 4294967296, %a ], [ 1, %b ], [ 1, %c ], [ 1, %d ]
  %sum1 = add i64 %arg, %p
  %sum2 = add i64 %sum1, %p
  ret i64 %sum2
; CHECK:       exit:
; CHECK-NEXT:    %[[PHI:.*]] = phi i64 [ {{[0-9]+}}, %a ], [ 1, %b ], [ 1, %c ], [ 1, %d ]
; CHECK-NEXT:    %[[SUM1:.*]] = add i64 %arg, %[[PHI]]
; CHECK-NEXT:    %[[SUM2:.*]] = add i64 %[[SUM1]], %[[PHI]]
; CHECK-NEXT:    ret i64 %[[SUM2]]
}

define i32 @test_no_spec_non_postdominating_uses(i1 %flag1, i1 %flag2, i32 %arg) {
; CHECK-LABEL: define i32 @test_no_spec_non_postdominating_uses(
entry:
  br i1 %flag1, label %a, label %b
; CHECK:         br i1 %flag1, label %a, label %b

a:
  br label %merge
; CHECK:       a:
; CHECK-NEXT:    %[[SUM_A:.*]] = add i32 %arg, 7
; CHECK-NEXT:    br label %merge

b:
  br label %merge
; CHECK:       b:
; CHECK-NEXT:    %[[SUM_B:.*]] = add i32 %arg, 11
; CHECK-NEXT:    br label %merge

merge:
  %p1 = phi i32 [ 7, %a ], [ 11, %b ]
  %p2 = phi i32 [ 13, %a ], [ 42, %b ]
  %sum1 = add i32 %arg, %p1
  br i1 %flag2, label %exit1, label %exit2
; CHECK:       merge:
; CHECK-NEXT:    %[[PHI1:.*]] = phi i32 [ %[[SUM_A]], %a ], [ %[[SUM_B]], %b ]
; CHECK-NEXT:    %[[PHI2:.*]] = phi i32 [ 13, %a ], [ 42, %b ]
; CHECK-NEXT:    br i1 %flag2, label %exit1, label %exit2

exit1:
  ret i32 %sum1
; CHECK:       exit1:
; CHECK-NEXT:    ret i32 %[[PHI1]]

exit2:
  %sum2 = add i32 %arg, %p2
  ret i32 %sum2
; CHECK:       exit2:
; CHECK-NEXT:    %[[SUM2:.*]] = add i32 %arg, %[[PHI2]]
; CHECK-NEXT:    ret i32 %[[SUM2]]
}
