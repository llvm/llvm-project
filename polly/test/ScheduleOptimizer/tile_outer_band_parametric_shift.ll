; RUN: opt %loadNPMPolly '-passes=polly-custom<simplify-0;optree;delicm;simplify-1;opt-isl>' -plugin-arg=Polly,-polly-print-opt-isl -disable-output < %s | FileCheck %s
;
; Do not tile a band that contains other loops if its schedule depends on
; parameters. The two statements of this loop nest, reduced from dradfg in
; MultiSource/Applications/oggenc, are shifted against each other by the
; parameter ip. Tiling the outer band would make the tile boundaries of each
; statement depend on ip modulo the tile size, and the AST generator would have
; to distinguish all these cases.
;
;    for (int i = 0; i < ido; i++)
;      for (int k = 0; k < 2; k++) {
;        c1[ip + i + (k + 1) * ido] = 0;
;        c1[i + (k + 1) * ido] = 0;
;      }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define fastcc void @dradfg(i32 %ido, i32 %ip, ptr %c1) {
entry.split:
  br label %for.body169

for.body169:
  %indvars.iv1447.in = phi i32 [ %indvars.iv1447, %for.cond174.for.inc217_crit_edge ], [ %ip, %entry.split ]
  %indvars.iv1440 = phi i32 [ %indvars.iv.next1441, %for.cond174.for.inc217_crit_edge ], [ 0, %entry.split ]
  %i.21259 = phi i32 [ %add218, %for.cond174.for.inc217_crit_edge ], [ 0, %entry.split ]
  %indvars.iv1447 = add i32 %indvars.iv1447.in, 1
  %0 = sext i32 %indvars.iv1440 to i64
  %1 = sext i32 %indvars.iv1447.in to i64
  br label %for.body177

for.body177:
  %indvars.iv1449 = phi i64 [ %1, %for.body169 ], [ %indvars.iv.next1450, %for.body177 ]
  %indvars.iv1442 = phi i64 [ %0, %for.body169 ], [ %indvars.iv.next1443, %for.body177 ]
  %k.31255 = phi i32 [ 0, %for.body169 ], [ %inc215, %for.body177 ]
  %2 = zext i32 %ido to i64
  %indvars.iv.next1450 = add i64 %indvars.iv1449, %2
  %indvars.iv.next1443 = add i64 %indvars.iv1442, %2
  %3 = getelementptr [4 x i8], ptr %c1, i64 %indvars.iv.next1450
  store float 0.000000e+00, ptr %3, align 4
  %arrayidx213 = getelementptr [4 x i8], ptr %c1, i64 %indvars.iv.next1443
  store float 0.000000e+00, ptr %arrayidx213, align 4
  %inc215 = add i32 %k.31255, 1
  %exitcond1455.not = icmp eq i32 %k.31255, 1
  br i1 %exitcond1455.not, label %for.cond174.for.inc217_crit_edge, label %for.body177

for.cond174.for.inc217_crit_edge:
  %add218 = add i32 %i.21259, 1
  %indvars.iv.next1441 = add i32 %indvars.iv1440, 1
  %cmp167 = icmp slt i32 %add218, %ido
  br i1 %cmp167, label %for.body169, label %for.cond166.for.inc220_crit_edge

for.cond166.for.inc220_crit_edge:
  ret void
}

; CHECK:      Calculated schedule:
; CHECK-NEXT: domain:
; CHECK-NEXT: child:
; CHECK-NEXT:   schedule: "[ido, ip] -> [{ Stmt_for_body177[i0, i1] -> [(ip + i0)]; Stmt_for_body177_b[i0, i1] -> [(i0)] }, { Stmt_for_body177[i0, i1] -> [(i1)]; Stmt_for_body177_b[i0, i1] -> [(i1)] }]"
; CHECK-NEXT:   permutable: 1
; CHECK-NOT:  1st level tiling
