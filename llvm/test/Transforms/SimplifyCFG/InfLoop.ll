; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -disable-output
; END.

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-ios9.0.0"

%struct.anon = type { %struct.anon.0, i32, i32, %union.T1 }
%struct.anon.0 = type { i32, [256 x i32], [256 x i8] }
%union.T1 = type { %struct.F}
%struct.F = type { i32 }

@U = internal global %struct.anon zeroinitializer, align 4

define void @main() {
entry:
  %0 = load i32, i32* getelementptr inbounds (%struct.anon, %struct.anon* @U, i32 0, i32 2), align 4
  %cmp.i = icmp eq i32 %0, -1
  br i1 %cmp.i, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %1 = load i32, i32* getelementptr inbounds (%struct.anon, %struct.anon* @U, i32 0, i32 2), align 4
  %bf.load = load i32, i32* getelementptr inbounds (%struct.anon, %struct.anon* @U, i32 0, i32 3, i32 0, i32 0), align 4
  %cmp = icmp slt i32 %0, 0
  br i1 %cmp, label %if.end7, label %cond.false

cond.false:                                       ; preds = %if.end
  %add = and i32 %bf.load, 30
  %shl = add nuw nsw i32 %add, 2
  br label %if.end7

if.end7:                                          ; preds = %if.end, %cond.false
  %2 = icmp eq i32 %0, 1
  br i1 %2, label %if.then9, label %if.else10

if.then9:                                         ; preds = %if.end7
  br label %if.end29

if.else10:                                        ; preds = %if.end7
  %cmp11 = icmp ugt i32 %0, 13
  br i1 %cmp11, label %if.then12, label %if.else14

if.then12:                                        ; preds = %if.else10
  br label %if.end26

if.else14:                                        ; preds = %if.else10
  %tobool = icmp eq i1 %2, 0
  br i1 %tobool, label %lor.rhs, label %if.then18

lor.rhs:                                          ; preds = %if.else14
  %tobool.not.i = icmp eq i1 %2, 0
  br i1 %tobool.not.i, label %if.else21, label %if.end.i54

if.end.i54:                                       ; preds = %lor.rhs
  br label %for.cond.i

for.cond.i:                                       ; preds = %if.end6.i, %if.end.i54
  %ix.0.i = phi i32 [ 0, %if.end.i54 ], [ %inc.i55, %if.end6.i ]
  %ret.0.off0.i = phi i1 [ false, %if.end.i54 ], [ %.ret.0.off0.i, %if.end6.i ]
  %cmp2.i = icmp ult i32 %ix.0.i, 2
  br i1 %cmp2.i, label %for.body.i, label %TmpSimpleNeedExt.exit

for.body.i:                                       ; preds = %for.cond.i
  %arrayidx.i = getelementptr inbounds %struct.anon, %struct.anon* @U, i32 0, i32 0, i32 2, i32 %ix.0.i
  %elt = load i8, i8* %arrayidx.i, align 1
  %cmp3.i = icmp sgt i8 %elt, 7
  br i1 %cmp3.i, label %if.else21, label %if.end6.i

if.end6.i:                                        ; preds = %for.body.i
  %cmp10.i = icmp ugt i8 %elt, 59
  %.ret.0.off0.i = or i1 %ret.0.off0.i, %cmp10.i
  %inc.i55 = add i32 %ix.0.i, 1
  br label %for.cond.i

TmpSimpleNeedExt.exit:                            ; preds = %for.body.i
  br i1 %ret.0.off0.i, label %if.then18, label %if.else21

if.then18:                                        ; preds = %if.else14, %TmpSimpleNeedExt.exit
  br label %if.end26

if.else21:                                        ; preds = %for.body.i, %lor.rhs, %TmpSimpleNeedExt.exit
  br label %if.end26

if.end26:                                         ; preds = %if.then18, %if.else21, %if.then12
  %cmp.i51 = icmp slt i32 %0, 7
  br i1 %cmp.i51, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %if.end26
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %if.end26
  br label %if.end29

if.then2.i:                                       ; preds = %if.end.i
  br label %if.end29

if.end29:                                         ; preds = %if.end.i, %if.then2.i, %if.then9
  ret void
}
