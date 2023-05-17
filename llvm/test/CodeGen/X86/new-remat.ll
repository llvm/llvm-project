; RUN: llc -verify-regalloc < %s | FileCheck %s
; Check all spills are rematerialized.
; CHECK-NOT: Spill

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = common global double 0.000000e+00, align 8
@a = common global i32 0, align 4

; Function Attrs: nounwind uwtable
define i32 @uniform_testdata(i32 %p1) {
entry:
  %cmp3 = icmp sgt i32 %p1, 0
  br i1 %cmp3, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %tmp = add i32 %p1, -1
  %xtraiter = and i32 %p1, 7
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod, label %for.body.preheader.split, label %for.body.prol.preheader

for.body.prol.preheader:                          ; preds = %for.body.preheader
  br label %for.body.prol

for.body.prol:                                    ; preds = %for.body.prol, %for.body.prol.preheader
  %i.04.prol = phi i32 [ %inc.prol, %for.body.prol ], [ 0, %for.body.prol.preheader ]
  %prol.iter = phi i32 [ %prol.iter.sub, %for.body.prol ], [ %xtraiter, %for.body.prol.preheader ]
  %tmp1 = load double, ptr @b, align 8
  %call.prol = tail call double @pow(double %tmp1, double 2.500000e-01)
  %inc.prol = add nuw nsw i32 %i.04.prol, 1
  %prol.iter.sub = add i32 %prol.iter, -1
  %prol.iter.cmp = icmp eq i32 %prol.iter.sub, 0
  br i1 %prol.iter.cmp, label %for.body.preheader.split.loopexit, label %for.body.prol

for.body.preheader.split.loopexit:                ; preds = %for.body.prol
  %inc.prol.lcssa = phi i32 [ %inc.prol, %for.body.prol ]
  br label %for.body.preheader.split

for.body.preheader.split:                         ; preds = %for.body.preheader.split.loopexit, %for.body.preheader
  %i.04.unr = phi i32 [ 0, %for.body.preheader ], [ %inc.prol.lcssa, %for.body.preheader.split.loopexit ]
  %tmp2 = icmp ult i32 %tmp, 7
  br i1 %tmp2, label %for.end.loopexit, label %for.body.preheader.split.split

for.body.preheader.split.split:                   ; preds = %for.body.preheader.split
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader.split.split
  %i.04 = phi i32 [ %i.04.unr, %for.body.preheader.split.split ], [ %inc.7, %for.body ]
  %tmp3 = load double, ptr @b, align 8
  %call = tail call double @pow(double %tmp3, double 2.500000e-01)
  %tmp4 = load double, ptr @b, align 8
  %call.1 = tail call double @pow(double %tmp4, double 2.500000e-01)
  %inc.7 = add nsw i32 %i.04, 8
  %exitcond.7 = icmp eq i32 %inc.7, %p1
  br i1 %exitcond.7, label %for.end.loopexit.unr-lcssa, label %for.body

for.end.loopexit.unr-lcssa:                       ; preds = %for.body
  br label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.end.loopexit.unr-lcssa, %for.body.preheader.split
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %tmp5 = load i32, ptr @a, align 4
  ret i32 %tmp5
}

; Function Attrs: nounwind
declare double @pow(double, double)
