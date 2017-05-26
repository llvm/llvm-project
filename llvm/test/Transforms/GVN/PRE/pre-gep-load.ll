; RUN: opt < %s -basicaa -gvn -enable-load-pre -S | FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn -enable-load-pre -S | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

define double @foo(i32 %stat, i32 %i, double** %p) {
; CHECK-LABEL: @foo(
entry:
  switch i32 %stat, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb
    i32 2, label %sw.bb2
  ]

sw.bb:                                            ; preds = %entry, %entry
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds double*, double** %p, i64 0
  %0 = load double*, double** %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds double, double* %0, i64 %idxprom
  %1 = load double, double* %arrayidx1, align 8
  %sub = fsub double %1, 1.000000e+00
  %cmp = fcmp olt double %sub, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %sw.bb
  br label %return

if.end:                                           ; preds = %sw.bb
  br label %sw.bb2

sw.bb2:                                           ; preds = %if.end, %entry
  %idxprom3 = sext i32 %i to i64
  %arrayidx4 = getelementptr inbounds double*, double** %p, i64 0
  %2 = load double*, double** %arrayidx4, align 8
  %arrayidx5 = getelementptr inbounds double, double* %2, i64 %idxprom3
  %3 = load double, double* %arrayidx5, align 8
; CHECK: sw.bb2:
; CHECK-NOT: sext
; CHECK-NEXT: phi double [
; CHECK-NOT: load
  %sub6 = fsub double 3.000000e+00, %3
  br label %return

sw.default:                                       ; preds = %entry
  br label %return

return:                                           ; preds = %sw.default, %sw.bb2, %if.then
  %retval.0 = phi double [ 0.000000e+00, %sw.default ], [ %sub6, %sw.bb2 ], [ %sub, %if.then ]
  ret double %retval.0
}

; The load causes the GEP's operands to be PREd earlier than normal. The
; resulting sext ends up in pre.dest and in the GVN system before that BB is
; actually processed. Make sure we can deal with the situation.

define void @test_shortcut_safe(i1 %tst, i32 %p1, i32* %a) {
; CHECK-LABEL: define void @test_shortcut_safe
; CHECK: [[SEXT1:%.*]] = sext i32 %p1 to i64
; CHECK: [[PHI1:%.*]] = phi i64 [ [[SEXT1]], {{%.*}} ], [ [[PHI2:%.*]], {{%.*}} ]
; CHECK: [[SEXT2:%.*]] = sext i32 %p1 to i64
; CHECK: [[PHI2]] = phi i64 [ [[SEXT2]], {{.*}} ], [ [[PHI1]], {{%.*}} ]
; CHECK: getelementptr inbounds i32, i32* %a, i64 [[PHI2]]

  br i1 %tst, label %sext1, label %pre.dest

pre.dest:
  br label %sext.use

sext1:
  %idxprom = sext i32 %p1 to i64
  br label %sext.use

sext.use:
  %idxprom2 = sext i32 %p1 to i64
  %arrayidx3 = getelementptr inbounds i32, i32* %a, i64 %idxprom2
  %val = load i32, i32* %arrayidx3, align 4
  tail call void (i32) @g(i32 %val)
  br label %pre.dest
}

declare void @g(i32)
