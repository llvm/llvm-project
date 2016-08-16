; RUN: opt %loadPolly -S -polly-codegen -polly-invariant-load-hoisting=true < %s
;
; Check that we generate valid code as we did non preload the base pointer
; origin of %tmp4 at some point.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@TOP = external global i64*, align 8
@BOT = external global i64*, align 8

define void @RemoveConstraintVCG() {
entry:
  br i1 undef, label %for.end.161, label %for.cond.2.preheader

for.cond.2.preheader:                             ; preds = %entry
  br i1 undef, label %for.end.128, label %for.body.4

for.body.4:                                       ; preds = %for.inc.126, %for.cond.2.preheader
  br i1 undef, label %for.cond.8.preheader, label %for.inc.126

for.cond.8.preheader:                             ; preds = %for.body.4
  br i1 undef, label %for.inc.126, label %for.body.11

for.body.11:                                      ; preds = %for.inc, %for.cond.8.preheader
  br i1 undef, label %land.lhs.true, label %for.inc

land.lhs.true:                                    ; preds = %for.body.11
  br i1 undef, label %if.then.20, label %for.inc

if.then.20:                                       ; preds = %land.lhs.true
  %tmp = load i64*, i64** @TOP, align 8
  %tmp1 = load i64, i64* %tmp, align 8
  %cmp25 = icmp eq i64 %tmp1, 1
  %cmp47 = icmp eq i64 %tmp1, 0
  br i1 false, label %if.end.117, label %lor.lhs.false.85

lor.lhs.false.85:                                 ; preds = %if.then.20
  %add94 = add i64 %tmp1, 1
  %tmp2 = load i64*, i64** @TOP, align 8
  %arrayidx95 = getelementptr inbounds i64, i64* %tmp2, i64 %add94
  %tmp3 = load i64, i64* %arrayidx95, align 8
  br i1 false, label %if.else.103, label %land.lhs.true.97

land.lhs.true.97:                                 ; preds = %lor.lhs.false.85
  %tmp4 = load i64*, i64** @BOT, align 8
  %arrayidx99 = getelementptr inbounds i64, i64* %tmp4, i64 %add94
  %tmp5 = load i64, i64* %arrayidx99, align 8
  %tobool100 = icmp eq i64 %tmp5, 0
  br i1 %tobool100, label %if.else.103, label %if.then.101

if.then.101:                                      ; preds = %land.lhs.true.97
  br label %if.end.117

if.else.103:                                      ; preds = %land.lhs.true.97, %lor.lhs.false.85
  %tmp6 = load i64*, i64** @TOP, align 8
  %arrayidx105 = getelementptr inbounds i64, i64* %tmp6, i64 %add94
  %tmp7 = load i64, i64* %arrayidx105, align 8
  br i1 false, label %lor.lhs.false.107, label %if.else.112

lor.lhs.false.107:                                ; preds = %if.else.103
  %tmp8 = load i64*, i64** @BOT, align 8
  %arrayidx109 = getelementptr inbounds i64, i64* %tmp8, i64 %add94
  br i1 false, label %if.end.117, label %if.else.112

if.else.112:                                      ; preds = %lor.lhs.false.107, %if.else.103
  br label %if.end.117

if.end.117:                                       ; preds = %if.else.112, %lor.lhs.false.107, %if.then.101, %if.then.20
  br i1 undef, label %if.then.119, label %for.inc

if.then.119:                                      ; preds = %if.end.117
  br label %for.inc

for.inc:                                          ; preds = %if.then.119, %if.end.117, %land.lhs.true, %for.body.11
  br i1 false, label %for.body.11, label %for.inc.126

for.inc.126:                                      ; preds = %for.inc, %for.cond.8.preheader, %for.body.4
  br i1 undef, label %for.end.128, label %for.body.4

for.end.128:                                      ; preds = %for.inc.126, %for.cond.2.preheader
  br i1 false, label %cond.false, label %cond.end

cond.false:                                       ; preds = %for.end.128
  unreachable

cond.end:                                         ; preds = %for.end.128
  unreachable

for.end.161:                                      ; preds = %entry
  ret void
}
