; RUN: llc -mtriple=arm-eabi -arm-atomic-cfg-tidy=0 %s -o - | FileCheck -check-prefix=ARM %s
; RUN: llc -mtriple=thumb-eabi -arm-atomic-cfg-tidy=0 %s -o - | FileCheck -check-prefix=THUMB %s
; RUN: llc -mtriple=thumb-eabi -arm-atomic-cfg-tidy=0 -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck -check-prefix=T2 %s
; RUN: llc -mtriple=thumbv8-eabi -arm-atomic-cfg-tidy=0 %s -o - | FileCheck -check-prefix=V8 %s

; FIXME: The -mtriple=thumb test doesn't change if -disable-peephole is specified.

%struct.Foo = type { i8* }

; ARM-LABEL:   foo:
; THUMB-LABEL: foo:
; T2-LABEL:    foo:
define %struct.Foo* @foo(%struct.Foo* %this, i32 %acc) nounwind readonly align 2 {
entry:
  %scevgep = getelementptr %struct.Foo, %struct.Foo* %this, i32 1
  br label %tailrecurse

tailrecurse:                                      ; preds = %sw.bb, %entry
  %lsr.iv2 = phi %struct.Foo* [ %scevgep3, %sw.bb ], [ %scevgep, %entry ]
  %lsr.iv = phi i32 [ %lsr.iv.next, %sw.bb ], [ 1, %entry ]
  %acc.tr = phi i32 [ %or, %sw.bb ], [ %acc, %entry ]
  %lsr.iv24 = bitcast %struct.Foo* %lsr.iv2 to i8**
  %scevgep5 = getelementptr i8*, i8** %lsr.iv24, i32 -1
  %tmp2 = load i8*, i8** %scevgep5
  %0 = ptrtoint i8* %tmp2 to i32

; ARM:      ands {{r[0-9]+}}, {{r[0-9]+}}, #3
; ARM-NEXT: beq

; THUMB:      movs r[[R0:[0-9]+]], #3
; THUMB-NEXT: ands r[[R0]], r
; THUMB-NEXT: cmp r[[R0]], #0
; THUMB-NEXT: beq

; T2:      ands {{r[0-9]+}}, {{r[0-9]+}}, #3
; T2-NEXT: beq

  %and = and i32 %0, 3
  %tst = icmp eq i32 %and, 0
  br i1 %tst, label %sw.bb, label %tailrecurse.switch

tailrecurse.switch:                               ; preds = %tailrecurse
; V8-LABEL: %tailrecurse.switch
; V8: cmp
; V8-NEXT: beq
; V8-NEXT: %tailrecurse.switch
; V8: cmp
; V8-NEXT: beq
; V8-NEXT: %tailrecurse.switch
; V8: cmp
; V8-NEXT: beq
; V8-NEXT: %sw.epilog
; V8-NEXT: bx lr
  switch i32 %and, label %sw.epilog [
    i32 1, label %sw.bb
    i32 3, label %sw.bb6
    i32 2, label %sw.bb8
  ], !prof !1

sw.bb:                                            ; preds = %tailrecurse.switch, %tailrecurse
  %shl = shl i32 %acc.tr, 1
  %or = or i32 %and, %shl
  %lsr.iv.next = add i32 %lsr.iv, 1
  %scevgep3 = getelementptr %struct.Foo, %struct.Foo* %lsr.iv2, i32 1
  br label %tailrecurse

sw.bb6:                                           ; preds = %tailrecurse.switch
  ret %struct.Foo* %lsr.iv2

sw.bb8:                                           ; preds = %tailrecurse.switch
  %tmp1 = add i32 %acc.tr, %lsr.iv
  %add.ptr11 = getelementptr inbounds %struct.Foo, %struct.Foo* %this, i32 %tmp1
  ret %struct.Foo* %add.ptr11

sw.epilog:                                        ; preds = %tailrecurse.switch
  ret %struct.Foo* undef
}

; Another test that exercises the AND/TST peephole optimization and also
; generates a predicated ANDS instruction. Check that the predicate is printed
; after the "S" modifier on the instruction.

%struct.S = type { i8* (i8*)*, [1 x i8] }

; ARM-LABEL: bar:
; THUMB-LABEL: bar:
; T2-LABEL: bar:
; V8-LABEL: bar:
define internal zeroext i8 @bar(%struct.S* %x, %struct.S* nocapture %y) nounwind readonly {
entry:
  %0 = getelementptr inbounds %struct.S, %struct.S* %x, i32 0, i32 1, i32 0
  %1 = load i8, i8* %0, align 1
  %2 = zext i8 %1 to i32
; ARM: ands
; THUMB: ands
; T2: ands
; V8: ands
; V8-NEXT: beq
  %3 = and i32 %2, 112
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %return, label %bb

bb:                                               ; preds = %entry
; V8-NEXT: %bb
  %5 = getelementptr inbounds %struct.S, %struct.S* %y, i32 0, i32 1, i32 0
  %6 = load i8, i8* %5, align 1
  %7 = zext i8 %6 to i32
; ARM: andsne
; THUMB: ands
; T2: andsne
; V8: ands
; V8-NEXT: beq
  %8 = and i32 %7, 112
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %return, label %bb2

bb2:                                              ; preds = %bb
; V8-NEXT: %bb2
; V8-NEXT: cmp
; V8-NEXT: it	ne
; V8-NEXT: cmpne
; V8-NEXT: bne
  %10 = icmp eq i32 %3, 16
  %11 = icmp eq i32 %8, 16
  %or.cond = or i1 %10, %11
  br i1 %or.cond, label %bb4, label %return

bb4:                                              ; preds = %bb2
  %12 = ptrtoint %struct.S* %x to i32
  %phitmp = trunc i32 %12 to i8
  ret i8 %phitmp

return:                                           ; preds = %bb2, %bb, %entry
  ret i8 1
}


; We were looking through multiple COPY instructions to find an AND we might
; fold into a TST, but in doing so we changed the register being tested allowing
; folding of unrelated tests (in this case, a TST against r1 was eliminated in
; favour of an AND of r0).

define i32 @test_tst_assessment(i32 %a, i32 %b) {
; ARM-LABEL: test_tst_assessment:
; ARM:       @ %bb.0:
; ARM-NEXT:    and r0, r0, #1
; ARM-NEXT:    tst r1, #1
; ARM-NEXT:    subne r0, r0, #1
; ARM-NEXT:    mov pc, lr
;
; THUMB-LABEL: test_tst_assessment:
; THUMB:       @ %bb.0:
; THUMB-NEXT:    movs r2, r0
; THUMB-NEXT:    movs r0, #1
; THUMB-NEXT:    ands r0, r2
; THUMB-NEXT:    subs r2, r0, #1
; THUMB-NEXT:    lsls r1, r1, #31
; THUMB-NEXT:    beq .LBB2_2
; THUMB-NEXT:  @ %bb.1:
; THUMB-NEXT:    movs r0, r2
; THUMB-NEXT:  .LBB2_2:
; THUMB-NEXT:    bx lr
;
; T2-LABEL: test_tst_assessment:
; T2:       @ %bb.0:
; T2-NEXT:    lsls r1, r1, #31
; T2-NEXT:    and r0, r0, #1
; T2-NEXT:    it ne
; T2-NEXT:    subne r0, #1
; T2-NEXT:    bx lr
;
; V8-LABEL: test_tst_assessment:
; V8:       @ %bb.0:
; V8-NEXT:    and r0, r0, #1
; V8-NEXT:    lsls r1, r1, #31
; V8-NEXT:    it ne
; V8-NEXT:    subne r0, #1
; V8-NEXT:    bx lr
  %and1 = and i32 %a, 1
  %sub = sub i32 %and1, 1
  %and2 = and i32 %b, 1
  %cmp = icmp eq i32 %and2, 0
  %sel = select i1 %cmp, i32 %and1, i32 %sub
  ret i32 %sel
}

!1 = !{!"branch_weights", i32 1, i32 1, i32 3, i32 2 }
