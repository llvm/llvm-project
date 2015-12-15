; RUN: llc %s -o - -enable-shrink-wrap=true | FileCheck %s --check-prefix=CHECK --check-prefix=ENABLE
; RUN: llc %s -o - -enable-shrink-wrap=false | FileCheck %s --check-prefix=CHECK --check-prefix=DISABLE
; We cannot merge this test with the main test for shrink-wrapping, because
; the code path we want to exerce is not taken with ios lowering.
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n8:16:32-S64"
target triple = "armv7--linux-gnueabi"

@skip = internal unnamed_addr constant [2 x i8] c"\01\01", align 1

; Check that we do not restore the before having used the saved CSRs.
; This happened because of a bad use of the post-dominance property.
; The exit block of the loop happens to also lead to defs/uses of CSRs.
; It also post-dominates the loop body and we use to generate invalid
; restore sequence. I.e., we restored too early.
;
; CHECK-LABEL: wrongUseOfPostDominate:
;
; The prologue is the first thing happening in the function
; without shrink-wrapping.
; DISABLE: push
;
; CHECK: cmp r1, #0
;
; With shrink-wrapping, we branch to a pre-header, where the prologue
; is located.
; ENABLE-NEXT: blt [[LOOP_PREHEADER:[.a-zA-Z0-9_]+]]
; Without shrink-wrapping, we go straight into the loop.
; DISABLE-NEXT: blt [[LOOP_HEADER:[.a-zA-Z0-9_]+]]
;
; CHECK: @ %if.end29
; DISABLE-NEXT: pop
; ENABLE-NEXT: bx lr
;
; ENABLE: [[LOOP_PREHEADER]]
; ENABLE: push
; We must not find a pop here, otherwise that means we are in the loop
; and are restoring before using the saved CSRs.
; ENABLE-NOT: pop
; ENALBE-NEXT: [[LOOP_HEADER:[.a-zA-Z0-9_]+]]: @ %while.cond2.outer
;
; DISABLE: [[LOOP_HEADER]]: @ %while.cond2.outer
;
; ENABLE-NOT: pop
;
; CHECK: @ %while.cond2
; CHECK: add
; CHECK-NEXT: cmp r{{[0-1]+}}, #1
; Set the return value.
; CHECK-NEXT: moveq r0,
; CHECK-NEXT: popeq
;
; Use the back edge to check we get the label of the loop right.
; This is to make sure we check the right loop pattern.
; CHECK:  @ %while.body24.land.rhs14_crit_edge
; CHECK: cmp r{{[0-9]+}}, #192
; CHECK-NEXT bhs [[LOOP_HEADER]]
define fastcc i8* @wrongUseOfPostDominate(i8* readonly %s, i32 %off, i8* readnone %lim) {
entry:
  %cmp = icmp sgt i32 %off, -1
  br i1 %cmp, label %while.cond.preheader, label %while.cond2.outer

while.cond.preheader:                             ; preds = %entry
  %tobool4 = icmp ne i32 %off, 0
  %cmp15 = icmp ult i8* %s, %lim
  %sel66 = and i1 %tobool4, %cmp15
  br i1 %sel66, label %while.body, label %if.end29

while.body:                                       ; preds = %while.body, %while.cond.preheader
  %s.addr.08 = phi i8* [ %add.ptr, %while.body ], [ %s, %while.cond.preheader ]
  %off.addr.07 = phi i32 [ %dec, %while.body ], [ %off, %while.cond.preheader ]
  %dec = add nsw i32 %off.addr.07, -1
  %tmp = load i8, i8* %s.addr.08, align 1, !tbaa !2
  %idxprom = zext i8 %tmp to i32
  %arrayidx = getelementptr inbounds [2 x i8], [2 x i8]* @skip, i32 0, i32 %idxprom
  %tmp1 = load i8, i8* %arrayidx, align 1, !tbaa !2
  %conv = zext i8 %tmp1 to i32
  %add.ptr = getelementptr inbounds i8, i8* %s.addr.08, i32 %conv
  %tobool = icmp ne i32 %off.addr.07, 1
  %cmp1 = icmp ult i8* %add.ptr, %lim
  %sel6 = and i1 %tobool, %cmp1
  br i1 %sel6, label %while.body, label %if.end29

while.cond2.outer:                                ; preds = %while.body24.land.rhs14_crit_edge, %while.body24, %land.rhs14.preheader, %if.then7, %entry
  %off.addr.1.ph = phi i32 [ %off, %entry ], [ %inc, %land.rhs14.preheader ], [ %inc, %if.then7 ], [ %inc, %while.body24.land.rhs14_crit_edge ], [ %inc, %while.body24 ]
  %s.addr.1.ph = phi i8* [ %s, %entry ], [ %incdec.ptr, %land.rhs14.preheader ], [ %incdec.ptr, %if.then7 ], [ %lsr.iv, %while.body24.land.rhs14_crit_edge ], [ %lsr.iv, %while.body24 ]
  br label %while.cond2

while.cond2:                                      ; preds = %while.body4, %while.cond2.outer
  %off.addr.1 = phi i32 [ %inc, %while.body4 ], [ %off.addr.1.ph, %while.cond2.outer ]
  %inc = add nsw i32 %off.addr.1, 1
  %tobool3 = icmp eq i32 %off.addr.1, 0
  br i1 %tobool3, label %if.end29, label %while.body4

while.body4:                                      ; preds = %while.cond2
  %tmp2 = icmp ugt i8* %s.addr.1.ph, %lim
  br i1 %tmp2, label %if.then7, label %while.cond2

if.then7:                                         ; preds = %while.body4
  %incdec.ptr = getelementptr inbounds i8, i8* %s.addr.1.ph, i32 -1
  %tmp3 = load i8, i8* %incdec.ptr, align 1, !tbaa !2
  %conv1525 = zext i8 %tmp3 to i32
  %tobool9 = icmp slt i8 %tmp3, 0
  %cmp129 = icmp ugt i8* %incdec.ptr, %lim
  %or.cond13 = and i1 %tobool9, %cmp129
  br i1 %or.cond13, label %land.rhs14.preheader, label %while.cond2.outer

land.rhs14.preheader:                             ; preds = %if.then7
  %cmp1624 = icmp slt i8 %tmp3, 0
  %cmp2026 = icmp ult i32 %conv1525, 192
  %or.cond27 = and i1 %cmp1624, %cmp2026
  br i1 %or.cond27, label %while.body24.preheader, label %while.cond2.outer

while.body24.preheader:                           ; preds = %land.rhs14.preheader
  %scevgep = getelementptr i8, i8* %s.addr.1.ph, i32 -2
  br label %while.body24

while.body24:                                     ; preds = %while.body24.land.rhs14_crit_edge, %while.body24.preheader
  %lsr.iv = phi i8* [ %scevgep, %while.body24.preheader ], [ %scevgep34, %while.body24.land.rhs14_crit_edge ]
  %cmp12 = icmp ugt i8* %lsr.iv, %lim
  br i1 %cmp12, label %while.body24.land.rhs14_crit_edge, label %while.cond2.outer

while.body24.land.rhs14_crit_edge:                ; preds = %while.body24
  %.pre = load i8, i8* %lsr.iv, align 1, !tbaa !2
  %cmp16 = icmp slt i8 %.pre, 0
  %conv15 = zext i8 %.pre to i32
  %cmp20 = icmp ult i32 %conv15, 192
  %or.cond = and i1 %cmp16, %cmp20
  %scevgep34 = getelementptr i8, i8* %lsr.iv, i32 -1
  br i1 %or.cond, label %while.body24, label %while.cond2.outer

if.end29:                                         ; preds = %while.cond2, %while.body, %while.cond.preheader
  %s.addr.3 = phi i8* [ %s, %while.cond.preheader ], [ %add.ptr, %while.body ], [ %s.addr.1.ph, %while.cond2 ]
  ret i8* %s.addr.3
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{!3, !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
