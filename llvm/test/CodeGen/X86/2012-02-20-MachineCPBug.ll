; RUN: llc < %s -mtriple=i386-apple-macosx -mcpu=core2 -mattr=+sse | FileCheck %s
; PR11940: Do not optimize away movb %al, %ch

%struct.APInt = type { i64* }

declare noalias i8* @calloc(i32, i32) nounwind

define void @bug(%struct.APInt* noalias nocapture sret %agg.result, %struct.APInt* nocapture %this, i32 %rotateAmt) nounwind align 2 {
entry:
; CHECK: bug:
  %call = tail call i8* @calloc(i32 1, i32 32)
  %call.i = tail call i8* @calloc(i32 1, i32 32) nounwind
  %0 = bitcast i8* %call.i to i64*
  %rem.i = and i32 %rotateAmt, 63
  %div.i = lshr i32 %rotateAmt, 6
  %cmp.i = icmp eq i32 %rem.i, 0
  br i1 %cmp.i, label %for.cond.preheader.i, label %if.end.i

for.cond.preheader.i:                             ; preds = %entry
  %sub.i = sub i32 4, %div.i
  %cmp23.i = icmp eq i32 %div.i, 4
  br i1 %cmp23.i, label %for.body9.lr.ph.i, label %for.body.lr.ph.i

for.body.lr.ph.i:                                 ; preds = %for.cond.preheader.i
  %pVal.i = getelementptr inbounds %struct.APInt* %this, i32 0, i32 0
  %.pre5.i = load i64** %pVal.i, align 4
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %for.body.lr.ph.i
  %i.04.i = phi i32 [ 0, %for.body.lr.ph.i ], [ %inc.i, %for.body.i ]
  %add.i = add i32 %i.04.i, %div.i
  %arrayidx.i = getelementptr inbounds i64* %.pre5.i, i32 %add.i
  %1 = load i64* %arrayidx.i, align 4
  %arrayidx3.i = getelementptr inbounds i64* %0, i32 %i.04.i
  store i64 %1, i64* %arrayidx3.i, align 4
  %inc.i = add i32 %i.04.i, 1
  %cmp2.i = icmp ult i32 %inc.i, %sub.i
  br i1 %cmp2.i, label %for.body.i, label %if.end.i

if.end.i:                                         ; preds = %for.body.i, %entry
  %cmp81.i = icmp eq i32 %div.i, 3
  br i1 %cmp81.i, label %_ZNK5APInt4lshrEj.exit, label %for.body9.lr.ph.i

for.body9.lr.ph.i:                                ; preds = %if.end.i, %for.cond.preheader.i
  %sub58.i = sub i32 3, %div.i
  %pVal11.i = getelementptr inbounds %struct.APInt* %this, i32 0, i32 0
  %sh_prom.i = zext i32 %rem.i to i64
  %sub17.i = sub i32 64, %rem.i
  %sh_prom18.i = zext i32 %sub17.i to i64
  %.pre.i = load i64** %pVal11.i, align 4
  br label %for.body9.i

for.body9.i:                                      ; preds = %for.body9.i, %for.body9.lr.ph.i
; CHECK: %for.body9.i
; CHECK: movb
; CHECK: shrdl
  %i6.02.i = phi i32 [ 0, %for.body9.lr.ph.i ], [ %inc21.i, %for.body9.i ]
  %add10.i = add i32 %i6.02.i, %div.i
  %arrayidx12.i = getelementptr inbounds i64* %.pre.i, i32 %add10.i
  %2 = load i64* %arrayidx12.i, align 4
  %shr.i = lshr i64 %2, %sh_prom.i
  %add14.i = add i32 %add10.i, 1
  %arrayidx16.i = getelementptr inbounds i64* %.pre.i, i32 %add14.i
  %3 = load i64* %arrayidx16.i, align 4
  %shl.i = shl i64 %3, %sh_prom18.i
  %or.i = or i64 %shl.i, %shr.i
  %arrayidx19.i = getelementptr inbounds i64* %0, i32 %i6.02.i
  store i64 %or.i, i64* %arrayidx19.i, align 4
  %inc21.i = add i32 %i6.02.i, 1
  %cmp8.i = icmp ult i32 %inc21.i, %sub58.i
  br i1 %cmp8.i, label %for.body9.i, label %_ZNK5APInt4lshrEj.exit

_ZNK5APInt4lshrEj.exit:                           ; preds = %for.body9.i, %if.end.i
  %call.i1 = tail call i8* @calloc(i32 1, i32 32) nounwind
  %4 = getelementptr inbounds %struct.APInt* %agg.result, i32 0, i32 0
  store i64* %0, i64** %4, align 4
  ret void
}
