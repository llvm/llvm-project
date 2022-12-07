; RUN: opt -mcpu=kryo -passes=loop-simplify,loop-data-prefetch -max-prefetch-iters-ahead=1000 -min-prefetch-stride=16 -S < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

%struct._Chv = type { i32, i32, i32, i32, i32, i32, i32*, i32*, double*, %struct._DV, %struct._Chv* }
%struct._DV = type { i32, i32, i32, double* }

declare double* @f_entries() local_unnamed_addr

define i32 @f(%struct._Chv* %chv, i32 %npivot, i32* %pivotsizes, i32* %sizes) local_unnamed_addr {
if.end:
  switch i32 undef, label %sw.default [
    i32 1, label %sw.epilog
    i32 3, label %sw.epilog
    i32 4, label %sw.epilog
    i32 5, label %sw.epilog
    i32 6, label %sw.epilog
    i32 7, label %sw.epilog
  ]

sw.default:                                       ; preds = %if.end
  unreachable

sw.epilog:                                        ; preds = %if.end, %if.end, %if.end, %if.end, %if.end, %if.end
  br label %if.end12

if.end12:                                         ; preds = %sw.epilog
  %nD13 = getelementptr inbounds %struct._Chv, %struct._Chv* %chv, i64 0, i32 1
  %0 = load i32, i32* %nD13, align 4
  %nU15 = getelementptr inbounds %struct._Chv, %struct._Chv* %chv, i64 0, i32 3
  %1 = load i32, i32* %nU15, align 4
  %add17 = add i32 %1, %0
  %call18 = call double* @f_entries()
  switch i32 undef, label %sw.epilog2454 [
    i32 3, label %sw.bb213
  ]

sw.bb213:                                         ; preds = %if.end12
  switch i32 undef, label %sw.epilog2454 [
    i32 0, label %sw.bb214
  ]

sw.bb214:                                         ; preds = %sw.bb213
  br label %if.then220

if.then220:                                       ; preds = %sw.bb214
  %type230 = getelementptr inbounds %struct._Chv, %struct._Chv* %chv, i64 0, i32 4
  %2 = load i32, i32* %type230, align 8
  br label %if.else319

if.else319:                                       ; preds = %if.then220
  switch i32 %2, label %sw.epilog2454 [
    i32 2, label %for.cond372.preheader
  ]

for.cond372.preheader:                            ; preds = %if.else319
  %cmp3734342 = icmp sgt i32 %npivot, 0
  br i1 %cmp3734342, label %for.body374.lr.ph, label %sw.epilog2454

for.body374.lr.ph:                                ; preds = %for.cond372.preheader
  br label %for.body374

for.body374:                                      ; preds = %for.body374.lr.ph
  %arrayidx376 = getelementptr inbounds i32, i32* %pivotsizes, i64 0
  %3 = load i32, i32* %arrayidx376, align 4
  %add377 = add i32 %3, 0
  br label %for.body381.lr.ph

for.body381.lr.ph:                                ; preds = %for.body374
  %cmp3874327 = icmp slt i32 %add377, %add17
  br i1 %cmp3874327, label %for.body381.us, label %for.body381.preheader

for.body381.preheader:                            ; preds = %for.body381.lr.ph
  ret i32 undef

for.body381.us:                                   ; preds = %for.cond386.for.end421_crit_edge.us, %for.body381.lr.ph
  %indvars.iv4991.us = phi i64 [ %indvars.iv.next4992.us, %for.cond386.for.end421_crit_edge.us ], [ undef, %for.body381.lr.ph ]
  %stride226.54337.us = phi i32 [ %dec423.us, %for.cond386.for.end421_crit_edge.us ], [ %add17, %for.body381.lr.ph ]
  %kstart225.54336.us = phi i32 [ %add422.us, %for.cond386.for.end421_crit_edge.us ], [ 0, %for.body381.lr.ph ]
  %4 = trunc i64 %indvars.iv4991.us to i32
  %5 = xor i32 %4, -1
  %add382.us = add i32 %add377, %5
  %sub383.us = add i32 %add382.us, %kstart225.54336.us
  br label %for.body388.us

; CHECK: for.body388.us:
; CHECK: call void @llvm.prefetch
for.body388.us:                                   ; preds = %for.inc418.us, %for.body381.us
  %kk224.3.in4330.us = phi i32 [ %sub383.us, %for.body381.us ], [ %kk224.3.us, %for.inc418.us ]
  %jj223.34329.us = phi i32 [ %add377, %for.body381.us ], [ %inc419.us, %for.inc418.us ]
  %kk224.3.us = add nsw i32 %kk224.3.in4330.us, 1
  %mul389.us = shl nsw i32 %kk224.3.us, 1
  %idxprom390.us = sext i32 %mul389.us to i64
  %arrayidx391.us = getelementptr inbounds double, double* %call18, i64 %idxprom390.us
  %6 = load double, double* %arrayidx391.us, align 8
  %call396.us = call double @Zabs(double %6)
  br label %for.inc418.us

for.inc418.us:                                    ; preds = %for.body388.us
  %inc419.us = add nsw i32 %jj223.34329.us, 1
  %cmp387.us = icmp slt i32 %inc419.us, %add17
  br i1 %cmp387.us, label %for.body388.us, label %for.cond386.for.end421_crit_edge.us

for.cond386.for.end421_crit_edge.us:              ; preds = %for.inc418.us
  %add422.us = add nsw i32 %kstart225.54336.us, %stride226.54337.us
  %dec423.us = add nsw i32 %stride226.54337.us, -1
  %indvars.iv.next4992.us = add nsw i64 %indvars.iv4991.us, 1
  br label %for.body381.us

sw.epilog2454:                                    ; preds = %for.cond372.preheader, %if.else319, %sw.bb213, %if.end12
  ret i32 0
}

declare double @llvm.fabs.f64(double)
declare dso_local double @Zabs(double) local_unnamed_addr
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg)

