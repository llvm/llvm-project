; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s

; CHECK-NOT: r{{[0-9]+}} = asr(r{{[0-9]+}},#{{[0-9]+}})
; CHECK-NOT: r{{[0-9]+}}:{{[0-9]+}} = mpyu(r{{[0-9]+}},r{{[0-9]+}})
; CHECK-NOT: r{{[0-9]+}} += mpyi(r{{[0-9]+}},r{{[0-9]+}})
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = mpy(r{{[0-9]+}},r{{[0-9]+}})

; ModuleID = '39544.c'
source_filename = "39544.c"
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dso_local void @mul_n(ptr nocapture %p, ptr nocapture readonly %a, i32 %k, i32 %n) local_unnamed_addr {
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %conv1 = sext i32 %k to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %arrayidx.phi = phi ptr [ %a, %for.body.lr.ph ], [ %arrayidx.inc, %for.body ]
  %arrayidx2.phi = phi ptr [ %p, %for.body.lr.ph ], [ %arrayidx2.inc, %for.body ]
  %i.08 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %0 = load i32, ptr %arrayidx.phi, align 4
  %conv = sext i32 %0 to i64
  %mul = mul nsw i64 %conv, %conv1
  store i64 %mul, ptr %arrayidx2.phi, align 8
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %n
  %arrayidx.inc = getelementptr i32, ptr %arrayidx.phi, i32 1
  %arrayidx2.inc = getelementptr i64, ptr %arrayidx2.phi, i32 1
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
