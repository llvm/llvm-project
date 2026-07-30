; REQUIRES: asserts
; RUN: opt -passes='loop-mssa(loop-rotate,licm)' %s -disable-output -debug-only=licm 2>&1 | FileCheck %s -check-prefix=LICM
; RUN: opt -passes='loop-mssa(loop-rotate,licm)' %s -S  | FileCheck %s

; LICM-NOT: LICM sinking instruction:   %.pre = load i8, ptr %arrayidx.phi.trans.insert

; CHECK-LABEL: @fn1
; CHECK-LABEL: entry:
; CHECK:    br i1 true, label %[[END:.*]], label %[[PH:.*]]
; CHECK: [[PH]]:
; CHECK:    br label %[[CRIT:.*]]
; CHECK: [[CRIT]]:
; CHECK:    load i8
; CHECK:    store i8
; CHECK:    br i1 true, label %[[ENDCRIT:.*]], label %[[CRIT]]
; CHECK: [[ENDCRIT]]:
; CHECK-NOT: load i8
; CHECK:    br label %[[END]]

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

define void @fn1() {
entry:
  %g = alloca [9 x i8], align 1
  br label %for.body

for.body:                                         ; preds = %for.body.for.body_crit_edge, %entry
  %0 = phi i64 [ 0, %entry ], [ %phitmp, %for.body.for.body_crit_edge ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body.for.body_crit_edge ]
  %arrayidx = getelementptr inbounds [9 x i8], ptr %g, i64 0, i64 %indvars.iv
  store i8 2, ptr %arrayidx, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br i1 undef, label %for.end18, label %for.body.for.body_crit_edge

for.body.for.body_crit_edge:                      ; preds = %for.body
  %arrayidx.phi.trans.insert = getelementptr inbounds [9 x i8], ptr %g, i64 0, i64 %indvars.iv.next
  %.pre = load i8, ptr %arrayidx.phi.trans.insert, align 1
  %phitmp = zext i8 %.pre to i64
  br label %for.body

for.end18:                                        ; preds = %for.body
  store i64 %0, ptr undef, align 8
  ret void
}

