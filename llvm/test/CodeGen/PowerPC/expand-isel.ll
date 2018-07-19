target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"
; RUN: llc -ppc-gpr-icmps=all -verify-machineinstrs -O2 -ppc-asm-full-reg-names -mcpu=pwr7 -ppc-gen-isel=false < %s | FileCheck %s --implicit-check-not isel

define signext i32 @testExpandISELToIfElse(i32 signext %i, i32 signext %j) {
entry:
  %cmp = icmp sgt i32 %i, 0
  %add = add nsw i32 %i, 1
  %cond = select i1 %cmp, i32 %add, i32 %j
  ret i32 %cond

; CHECK-LABEL: @testExpandISELToIfElse
; CHECK: addi r5, r3, 1
; CHECK-NEXT: cmpwi cr0, r3, 0
; CHECK-NEXT: bc 12, gt, [[TRUE:.LBB[0-9]+]]
; CHECK: ori r3, r4, 0
; CHECK-NEXT: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK-NEXT:  [[TRUE]]
; CHECK-NEXT: addi r3, r5, 0
; CHECK-NEXT: [[SUCCESSOR]]
; CHECK-NEXT: extsw r3, r3
; CHECK-NEXT: blr
}


define signext i32 @testExpandISELToIf(i32 signext %i, i32 signext %j) {
entry:
  %cmp = icmp sgt i32 %i, 0
  %cond = select i1 %cmp, i32 %j, i32 %i
  ret i32 %cond

; CHECK-LABEL: @testExpandISELToIf
; CHECK: cmpwi	 r3, 0
; CHECK-NEXT: bc 12, gt, [[TRUE:.LBB[0-9]+]]
; CHECK-NEXT: blr
; CHECK-NEXT:  [[TRUE]]
; CHECK-NEXT: addi r3, r4, 0
; CHECK-NEXT: blr
}

define signext i32 @testExpandISELToElse(i32 signext %i, i32 signext %j) {
entry:
  %cmp = icmp sgt i32 %i, 0
  %cond = select i1 %cmp, i32 %i, i32 %j
  ret i32 %cond

; CHECK-LABEL: @testExpandISELToElse
; CHECK: cmpwi	 r3, 0
; CHECK-NEXT: bclr 12, gt, 0
; CHECK: ori r3, r4, 0
; CHECK-NEXT: blr
}


define signext i32 @testExpandISELToNull(i32 signext %i, i32 signext %j) {
entry:
  %cmp = icmp sgt i32 %i, 0
  %cond = select i1 %cmp, i32 %i, i32 %i
  ret i32 %cond

; CHECK-LABEL: @testExpandISELToNull
; CHECK-NOT: b {{.LBB[0-9]+}}
; CHECK-NOT: bc
; CHECK: blr
}

define signext i32 @testExpandISELsTo2ORIs2ADDIs
  (i32 signext %a, i32 signext %b, i32 signext %d,
   i32 signext %f, i32 signext %g) {
entry:

  %cmp = icmp sgt i32 %g, 0
  %a.b = select i1 %cmp, i32 %g, i32 %b
  %d.f = select i1 %cmp, i32 %d, i32 %f
  %add = add nsw i32 %a.b, %d.f
  ret i32 %add

; CHECK-LABEL: @testExpandISELsTo2ORIs2ADDIs
; CHECK: cmpwi r7, 0
; CHECK-NEXT: bc 12, gt, [[TRUE:.LBB[0-9]+]]
; CHECK: ori r3, r4, 0
; CHECK-NEXT: ori r12, r6, 0
; CHECK-NEXT: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK-NEXT:  [[TRUE]]
; CHECK-NEXT: addi r3, r7, 0
; CHECK-NEXT: addi r12, r5, 0
; CHECK-NEXT: [[SUCCESSOR]]
; CHECK-NEXT: add r3, r3, r12
; CHECK-NEXT: extsw r3, r3
; CHECK-NEXT: blr
}

define signext i32 @testExpandISELsTo2ORIs1ADDI
  (i32 signext %a, i32 signext %b, i32 signext %d,
   i32 signext %f, i32 signext %g) {
entry:
  %cmp = icmp sgt i32 %g, 0
  %a.b = select i1 %cmp, i32 %a, i32 %b
  %d.f = select i1 %cmp, i32 %d, i32 %f
  %add = add nsw i32 %a.b, %d.f
  ret i32 %add

; CHECK-LABEL: @testExpandISELsTo2ORIs1ADDI
; CHECK: cmpwi cr0, r7, 0
; CHECK-NEXT: bc 12, gt, [[TRUE:.LBB[0-9]+]]
; CHECK: ori r3, r4, 0
; CHECK-NEXT: ori r12, r6, 0
; CHECK-NEXT: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK-NEXT: [[TRUE]]
; CHECK-NEXT: addi r12, r5, 0
; CHECK-NEXT:  [[SUCCESSOR]]
; CHECK-NEXT: add r3, r3, r12
; CHECK-NEXT: extsw r3, r3
; CHECK-NEXT: blr
}

define signext i32 @testExpandISELsTo1ORI1ADDI
  (i32 signext %a, i32 signext %b, i32 signext %d,
   i32 signext %f, i32 signext %g) {
entry:

  %cmp = icmp sgt i32 %g, 0
  %a.b = select i1 %cmp, i32 %a, i32 %b
  %d.f = select i1 %cmp, i32 %d, i32 %f
  %add1 = add nsw i32 %a.b, %d.f
  %add2 = add nsw i32 %a, %add1
  ret i32 %add2

; CHECK-LABEL: @testExpandISELsTo1ORI1ADDI
; CHECK: cmpwi cr0, r7, 0
; CHECK-NEXT: bc 12, gt, [[TRUE:.LBB[0-9]+]]
; CHECK: ori r5, r6, 0
; CHECK-NEXT: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK-NEXT: [[TRUE]]
; CHECK-NEXT: addi r4, r3, 0
; CHECK-NEXT:  [[SUCCESSOR]]
; CHECK-NEXT: add r4, r4, r5
; CHECK-NEXT: add r3, r3, r4
; CHECK-NEXT: extsw r3, r3
; CHECK-NEXT: blr
}

define signext i32 @testExpandISELsTo0ORI2ADDIs
  (i32 signext %a, i32 signext %b, i32 signext %d,
   i32 signext %f, i32 signext %g) {
entry:

  %cmp = icmp sgt i32 %g, 0
  %a.b = select i1 %cmp, i32 %a, i32 %b
  %d.f = select i1 %cmp, i32 %d, i32 %f
  %add1 = add nsw i32 %a.b, %d.f
  %add2 = add nsw i32 %a, %add1
  %sub1 = sub nsw i32 %add2, %d
  ret i32 %sub1

; CHECK-LABEL: @testExpandISELsTo0ORI2ADDIs
; CHECK: cmpwi cr0, r7, 0
; CHECK-NEXT: bc 12, gt, [[TRUE:.LBB[0-9]+]]
; CHECK-NEXT: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK-NEXT:  [[TRUE]]
; CHECK-NEXT: addi r4, r3, 0
; CHECK-NEXT: addi r6, r5, 0
; CHECK-NEXT:  [[SUCCESSOR]]
; CHECK-NEXT: add r4, r4, r6
; CHECK-NEXT: add r3, r3, r4
; CHECK-NEXT: subf r3, r5, r3
; CHECK-NEXT: extsw r3, r3
; CHECK-NEXT: blr
}


@b = common local_unnamed_addr global i32 0, align 4
@a = common local_unnamed_addr global i32 0, align 4
; Function Attrs: norecurse nounwind readonly
define signext i32 @testComplexISEL() #0 {
entry:
  %0 = load i32, i32* @b, align 4, !tbaa !1
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end, label %cleanup

if.end:
  %1 = load i32, i32* @a, align 4, !tbaa !1
  %conv = sext i32 %1 to i64
  %2 = inttoptr i64 %conv to i32 (...)*
  %cmp = icmp eq i32 (...)* %2, bitcast (i32 ()* @testComplexISEL to i32 (...)*)
  %conv3 = zext i1 %cmp to i32
  br label %cleanup

cleanup:
  %retval.0 = phi i32 [ %conv3, %if.end ], [ 1, %entry ]
  ret i32 %retval.0

; CHECK-LABEL: @testComplexISEL
; CHECK: cmplwi r3, 0
; CHECK: li r3, 1
; CHECK: beq cr0, [[TGT:.LBB[0-9_]+]]
; CHECK: clrldi r3, r3, 32
; CHECK: blr
; CHECK: [[TGT]]
; CHECK: xor [[XOR:r[0-9]+]]
; CHECK: cntlzd [[CZ:r[0-9]+]], [[XOR]]
; CHECK: rldicl [[SH:r[0-9]+]], [[CZ]], 58, 63
}

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
