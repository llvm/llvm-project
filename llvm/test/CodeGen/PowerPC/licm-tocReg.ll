; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; The instructions ADDIStocHA/LDtocL are used to calculate the address of
; globals. The ones that are in bb.3.if.end could not be hoisted by Machine
; LICM due to BCTRL_LDinto_toc in bb2.if.then.  This call causes the compiler
; to insert a save TOC to stack before the call and load into X2 to restore TOC
; after. By communicating to Machine LICM that X2 is guaranteed to have the
; same value before and after BCTRL_LDinto_toc, these instructions can be
; hoisted out of bb.3.if.end to outside of the loop.

; Pre Machine LICM MIR
;
;body:
;  bb.0.entry:
;    successors: %bb.2.if.then(0x40000000), %bb.3.if.end(0x40000000)
;    liveins: %x3
;
;    %4 = COPY %x3
;    %5 = ADDIStocHA %x2, @ga
;    %6 = LDtocL @ga, killed %5 :: (load 8 from got)
;    %7 = LWZ 0, %6 :: (volatile dereferenceable load 4 from @ga)
;    %8 = ADDIStocHA %x2, @gb
;    %9 = LDtocL @gb, killed %8 :: (load 8 from got)
;    %10 = LWZ 0, killed %9 :: (volatile dereferenceable load 4 from @gb)
;    %0 = LWZ 0, %6 :: (volatile dereferenceable load 4 from @ga)
;    %11 = CMPW killed %7, killed %10
;    BCC 44, killed %11, %bb.2.if.then
;    B %bb.3.if.end
;
;  bb.2.if.then:
;    %1 = PHI %0, %bb.0.entry, %3, %bb.3.if.end
;    ADJCALLSTACKDOWN 32, 0, implicit-def dead %r1, implicit %r1
;    %20 = COPY %x2
;    STD %20, 24, %x1 :: (store 8 into stack + 24)
;    %21 = EXTSW_32_64 %1
;    %x3 = COPY %21
;    %x12 = COPY %4
;    MTCTR8 %4, implicit-def %ctr8
;    BCTRL8_LDinto_toc 24, %x1, csr_svr464_altivec, implicit-def dead %lr8, implicit-def dead %x2, implicit %ctr8, implicit %rm, implicit %x3, implicit %x12, implicit %x2, implicit-def %r1, implicit-def %x3
;    ADJCALLSTACKUP 32, 0, implicit-def dead %r1, implicit %r1
;    %22 = COPY %x3
;    %x3 = COPY %22
;    BLR8 implicit %lr8, implicit %rm, implicit %x3
;
;  bb.3.if.end:
;    successors: %bb.2.if.then(0x04000000), %bb.3.if.end(0x7c000000)
;
;    %2 = PHI %0, %bb.0.entry, %3, %bb.3.if.end
;    %12 = ADDI %2, 1
;    %13 = ADDIStocHA %x2, @ga
;    %14 = LDtocL @ga, killed %13 :: (load 8 from got)
;    STW killed %12, 0, %14 :: (volatile store 4 into @ga)
;    %15 = LWZ 0, %14 :: (volatile dereferenceable load 4 from @ga)
;    %16 = ADDIStocHA %x2, @gb
;    %17 = LDtocL @gb, killed %16 :: (load 8 from got)
;    %18 = LWZ 0, killed %17 :: (volatile dereferenceable load 4 from @gb)
;    %3 = LWZ 0, %14 :: (volatile dereferenceable load 4 from @ga)
;    %19 = CMPW killed %15, killed %18
;    BCC 44, killed %19, %bb.2.if.then
;    B %bb.3.if.end

@ga = external global i32, align 4
@gb = external global i32, align 4

define signext i32 @test(i32 (i32)* nocapture %FP) local_unnamed_addr #0 {
; CHECK-LABEL: test:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    addis 6, 2, .LC0@toc@ha
; CHECK-NEXT:    addis 4, 2, .LC1@toc@ha
; CHECK-NEXT:    ld 5, .LC1@toc@l(4)
; CHECK-NEXT:    ld 6, .LC0@toc@l(6)
; CHECK-NEXT:    lwz 4, 0(5)
; CHECK-NEXT:    lwz 7, 0(6)
; CHECK-NEXT:    cmpw 4, 7
; CHECK-NEXT:    lwz 7, 0(5)
; CHECK-NEXT:    mr 4, 3
; CHECK-NEXT:    bgt 0, .LBB0_2
; CHECK-NOT:    addis {{[0-9]+}}, 2, .LC0@toc@ha
; CHECK-NOT:    addis {{[0-9]+}}, 2, .LC1@toc@ha
; CHECK-NEXT:    .p2align 5
; CHECK-NEXT:  .LBB0_1: # %if.end
; CHECK-NOT:    addis {{[0-9]+}}, 2, .LC0@toc@ha
; CHECK-NOT:    addis {{[0-9]+}}, 2, .LC1@toc@ha
; CHECK:    blr
entry:
  %0 = load volatile i32, i32* @ga, align 4
  %1 = load volatile i32, i32* @gb, align 4
  %cmp1 = icmp sgt i32 %0, %1
  %2 = load volatile i32, i32* @ga, align 4
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %if.end, %entry
  %.lcssa = phi i32 [ %2, %entry ], [ %6, %if.end ]
  %call = tail call signext i32 %FP(i32 signext %.lcssa) #1
  ret i32 %call

if.end:                                           ; preds = %entry, %if.end
  %3 = phi i32 [ %6, %if.end ], [ %2, %entry ]
  %inc = add nsw i32 %3, 1
  store volatile i32 %inc, i32* @ga, align 4
  %4 = load volatile i32, i32* @ga, align 4
  %5 = load volatile i32, i32* @gb, align 4
  %cmp = icmp sgt i32 %4, %5
  %6 = load volatile i32, i32* @ga, align 4
  br i1 %cmp, label %if.then, label %if.end
}
