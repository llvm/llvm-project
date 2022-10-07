; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck -check-prefixes=CHECK,CHECKLX %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff < %s | FileCheck -check-prefixes=CHECK,CHECKAIX %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc-ibm-aix-xcoff < %s | FileCheck -check-prefixes=CHECK,CHECKAIX32 %s

; The instructions ADDIStocHA8/LDtocL are used to calculate the address of
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
;    %5 = ADDIStocHA8 %x2, @ga
;    %6 = LDtocL @ga, killed %5 :: (load (s64) from got)
;    %7 = LWZ 0, %6 :: (volatile dereferenceable load (s32) from @ga)
;    %8 = ADDIStocHA8 %x2, @gb
;    %9 = LDtocL @gb, killed %8 :: (load (s64) from got)
;    %10 = LWZ 0, killed %9 :: (volatile dereferenceable load (s32) from @gb)
;    %0 = LWZ 0, %6 :: (volatile dereferenceable load (s32) from @ga)
;    %11 = CMPW killed %7, killed %10
;    BCC 44, killed %11, %bb.2.if.then
;    B %bb.3.if.end
;
;  bb.2.if.then:
;    %1 = PHI %0, %bb.0.entry, %3, %bb.3.if.end
;    ADJCALLSTACKDOWN 32, 0, implicit-def dead %r1, implicit %r1
;    %20 = COPY %x2
;    STD %20, 24, %x1 :: (store (s64) into stack + 24)
;    %21 = EXTSW_32_64 %1
;    %x3 = COPY %21
;    %x12 = COPY %4
;    MTCTR8 %4, implicit-def %ctr8
;    BCTRL8_LDinto_toc 24, %x1, csr_ppc64_altivec, implicit-def dead %lr8, implicit-def dead %x2, implicit %ctr8, implicit %rm, implicit %x3, implicit %x12, implicit %x2, implicit-def %r1, implicit-def %x3
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
;    %13 = ADDIStocHA8 %x2, @ga
;    %14 = LDtocL @ga, killed %13 :: (load (s64) from got)
;    STW killed %12, 0, %14 :: (volatile store (s32) into @ga)
;    %15 = LWZ 0, %14 :: (volatile dereferenceable load (s32) from @ga)
;    %16 = ADDIStocHA8 %x2, @gb
;    %17 = LDtocL @gb, killed %16 :: (load (s64) from got)
;    %18 = LWZ 0, killed %17 :: (volatile dereferenceable load (s32) from @gb)
;    %3 = LWZ 0, %14 :: (volatile dereferenceable load (s32) from @ga)
;    %19 = CMPW killed %15, killed %18
;    BCC 44, killed %19, %bb.2.if.then
;    B %bb.3.if.end

@ga = external global i32, align 4
@gb = external global i32, align 4
define signext i32 @test(ptr nocapture %FP) local_unnamed_addr #0 {
; CHECK-LABEL: test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mflr 0
; CHECKLX:        addis 4, 2, .LC0@toc@ha
; CHECKLX-NEXT:   addis 5, 2, .LC1@toc@ha
; CHECKLX-NEXT:   mr 12, 3
; CHECKLX-NEXT:   ld 4, .LC0@toc@l(4)
; CHECKLX-NEXT:   ld 5, .LC1@toc@l(5)
; CHECKLX-NEXT:   lwz 6, 0(4)
; CHECKLX-NEXT:   lwz 7, 0(5)
; CHECKLX-NEXT:   cmpw 6, 7
; CHECKLX-NEXT:   lwz 6, 0(4)
; CHECKLX-NEXT:   bgt 0, .LBB0_2
; CHECKLX-NOT:    addis {{[0-9]+}}, 2, .LC0@toc@ha
; CHECKLX-NOT:    addis {{[0-9]+}}, 2, .LC1@toc@ha
; CHECKLX-NEXT:   .p2align 5
; CHECKLX-NEXT: .LBB0_1: # %if.end
; CHECKLX-NOT:    addis {{[0-9]+}}, 2, .LC0@toc@ha
; CHECKLX-NOT:    addis {{[0-9]+}}, 2, .LC1@toc@ha
; CHECKAIX:        ld 5, L..C0(2)
; CHECKAIX-NEXT:   ld 6, L..C1(2)
; CHECKAIX-NEXT: L..BB0_1: # %if.end
; CHECKAIX-NOT:    ld {{[0-9]+}}, L..C0(2)
; CHECKAIX-NOT:    ld {{[0-9]+}}, L..C1(2)
; CHECKAIX32:        lwz 5, L..C0(2)
; CHECKAIX32-NEXT:   lwz 6, L..C1(2)
; CHECKAIX32-NEXT: L..BB0_1: # %if.end
; CHECKAIX32-NOT:    lwz 5, L..C0(2)
; CHECKAIX32-NOT:    lwz 6, L..C1(2)
; CHECK:    blr
entry:
  %0 = load volatile i32, ptr @ga, align 4
  %1 = load volatile i32, ptr @gb, align 4
  %cmp1 = icmp sgt i32 %0, %1
  %2 = load volatile i32, ptr @ga, align 4
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %if.end, %entry
  %.lcssa = phi i32 [ %2, %entry ], [ %6, %if.end ]
  %call = tail call signext i32 %FP(i32 signext %.lcssa) #1
  ret i32 %call

if.end:                                           ; preds = %entry, %if.end
  %3 = phi i32 [ %6, %if.end ], [ %2, %entry ]
  %inc = add nsw i32 %3, 1
  store volatile i32 %inc, ptr @ga, align 4
  %4 = load volatile i32, ptr @ga, align 4
  %5 = load volatile i32, ptr @gb, align 4
  %cmp = icmp sgt i32 %4, %5
  %6 = load volatile i32, ptr @ga, align 4
  br i1 %cmp, label %if.then, label %if.end
}
