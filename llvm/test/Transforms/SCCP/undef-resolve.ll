; RUN: opt -sccp -S < %s | FileCheck %s


; PR6940
define double @test1() {
  %t = sitofp i32 undef to double
  ret double %t
; CHECK-LABEL: @test1(
; CHECK: ret double 0.0
}


; rdar://7832370
; Check that lots of stuff doesn't get turned into undef.
define i32 @test2() nounwind readnone ssp {
; CHECK-LABEL: @test2(
init:
  br label %control.outer.outer

control.outer.loopexit.us-lcssa:                  ; preds = %control
  br label %control.outer.loopexit

control.outer.loopexit:                           ; preds = %control.outer.loopexit.us-lcssa.us, %control.outer.loopexit.us-lcssa
  br label %control.outer.outer.backedge

control.outer.outer:                              ; preds = %control.outer.outer.backedge, %init
  %switchCond.0.ph.ph = phi i32 [ 2, %init ], [ 3, %control.outer.outer.backedge ] ; <i32> [#uses=2]
  %i.0.ph.ph = phi i32 [ undef, %init ], [ %i.0.ph.ph.be, %control.outer.outer.backedge ] ; <i32> [#uses=1]
  %tmp4 = icmp eq i32 %i.0.ph.ph, 0               ; <i1> [#uses=1]
  br i1 %tmp4, label %control.outer.outer.split.us, label %control.outer.outer.control.outer.outer.split_crit_edge

control.outer.outer.control.outer.outer.split_crit_edge: ; preds = %control.outer.outer
  br label %control.outer

control.outer.outer.split.us:                     ; preds = %control.outer.outer
  br label %control.outer.us

control.outer.us:                                 ; preds = %bb3.us, %control.outer.outer.split.us
  %A.0.ph.us = phi i32 [ %switchCond.0.us, %bb3.us ], [ 4, %control.outer.outer.split.us ] ; <i32> [#uses=2]
  %switchCond.0.ph.us = phi i32 [ %A.0.ph.us, %bb3.us ], [ %switchCond.0.ph.ph, %control.outer.outer.split.us ] ; <i32> [#uses=1]
  br label %control.us

bb3.us:                                           ; preds = %control.us
  br label %control.outer.us

bb0.us:                                           ; preds = %control.us
  br label %control.us

; CHECK: control.us:                                       ; preds = %bb0.us, %control.outer.us
; CHECK-NEXT:  %switchCond.0.us = phi i32
; CHECK-NEXT:  switch i32 %switchCond.0.us
control.us:                                       ; preds = %bb0.us, %control.outer.us
  %switchCond.0.us = phi i32 [ %A.0.ph.us, %bb0.us ], [ %switchCond.0.ph.us, %control.outer.us ] ; <i32> [#uses=2]
  switch i32 %switchCond.0.us, label %control.outer.loopexit.us-lcssa.us [
    i32 0, label %bb0.us
    i32 1, label %bb1.us-lcssa.us
    i32 3, label %bb3.us
    i32 4, label %bb4.us-lcssa.us
  ]

control.outer.loopexit.us-lcssa.us:               ; preds = %control.us
  br label %control.outer.loopexit

bb1.us-lcssa.us:                                  ; preds = %control.us
  br label %bb1

bb4.us-lcssa.us:                                  ; preds = %control.us
  br label %bb4

control.outer:                                    ; preds = %bb3, %control.outer.outer.control.outer.outer.split_crit_edge
  %A.0.ph = phi i32 [ %nextId17, %bb3 ], [ 4, %control.outer.outer.control.outer.outer.split_crit_edge ] ; <i32> [#uses=1]
  %switchCond.0.ph = phi i32 [ 0, %bb3 ], [ %switchCond.0.ph.ph, %control.outer.outer.control.outer.outer.split_crit_edge ] ; <i32> [#uses=1]
  br label %control

control:                                          ; preds = %bb0, %control.outer
  %switchCond.0 = phi i32 [ %A.0.ph, %bb0 ], [ %switchCond.0.ph, %control.outer ] ; <i32> [#uses=2]
  switch i32 %switchCond.0, label %control.outer.loopexit.us-lcssa [
    i32 0, label %bb0
    i32 1, label %bb1.us-lcssa
    i32 3, label %bb3
    i32 4, label %bb4.us-lcssa
  ]

bb4.us-lcssa:                                     ; preds = %control
  br label %bb4

bb4:                                              ; preds = %bb4.us-lcssa, %bb4.us-lcssa.us
  br label %control.outer.outer.backedge

control.outer.outer.backedge:                     ; preds = %bb4, %control.outer.loopexit
  %i.0.ph.ph.be = phi i32 [ 1, %bb4 ], [ 0, %control.outer.loopexit ] ; <i32> [#uses=1]
  br label %control.outer.outer

bb3:                                              ; preds = %control
  %nextId17 = add i32 %switchCond.0, -2           ; <i32> [#uses=1]
  br label %control.outer

bb0:                                              ; preds = %control
  br label %control

bb1.us-lcssa:                                     ; preds = %control
  br label %bb1

bb1:                                              ; preds = %bb1.us-lcssa, %bb1.us-lcssa.us
  ret i32 0
}

; Make sure SCCP honors the xor "idiom"
; rdar://9956541
define i32 @test3() {
  %t = xor i32 undef, undef
  ret i32 %t
; CHECK-LABEL: @test3(
; CHECK: ret i32 0
}

; Be conservative with FP ops
define double @test4(double %x) {
  %t = fadd double %x, undef
  ret double %t
; CHECK-LABEL: @test4(
; CHECK: fadd double %x, undef
}

; Make sure casts produce a possible value
define i32 @test5() {
  %t = sext i8 undef to i32
  ret i32 %t
; CHECK-LABEL: @test5(
; CHECK: ret i32 0
}

; Make sure ashr produces a possible value
define i32 @test6() {
  %t = ashr i32 undef, 31
  ret i32 %t
; CHECK-LABEL: @test6(
; CHECK: ret i32 -1
}

; Make sure lshr produces a possible value
define i32 @test7() {
  %t = lshr i32 undef, 31
  ret i32 %t
; CHECK-LABEL: @test7(
; CHECK: ret i32 0
}

; icmp eq with undef simplifies to undef
define i1 @test8() {
  %t = icmp eq i32 undef, -1
  ret i1 %t
; CHECK-LABEL: @test8(
; CHECK: ret i1 undef
}

; Make sure we don't conclude that relational comparisons simplify to undef
define i1 @test9() {
  %t = icmp ugt i32 undef, -1
  ret i1 %t
; CHECK-LABEL: @test9(
; CHECK: icmp ugt
}

; Make sure we handle extractvalue
define i64 @test10() { 
entry:
  %e = extractvalue { i64, i64 } undef, 1
  ret i64 %e
; CHECK-LABEL: @test10(
; CHECK: ret i64 undef
}

@GV = external global i32

define i32 @test11(i1 %tobool) {
entry:
  %shr4 = ashr i32 undef, zext (i1 icmp eq (i32* bitcast (i32 (i1)* @test11 to i32*), i32* @GV) to i32)
  ret i32 %shr4
; CHECK-LABEL: @test11(
; CHECK: ret i32 -1
}
