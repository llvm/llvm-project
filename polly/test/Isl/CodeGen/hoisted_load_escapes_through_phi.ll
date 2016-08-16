; RUN: opt %loadPolly -S -polly-codegen \
; RUN: -polly-invariant-load-hoisting < %s | FileCheck %s
;
; Check that we generate valid code even if the load of cont_STACKPOINTER is
; hoisted in one SCoP and used (through the phi node %tmp2).
;
; CHECK: polly.start
; CHECK: polly.start
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct1 = type { i32, %union1, %struct.2*, i32, i32 }
%union1 = type { %struct.2* }
%struct.2 = type { %struct.2*, i8* }

@cont_STACKPOINTER = external global i32, align 4
@cont_STACK = external global [1000 x i32], align 16

define fastcc void @subs_InternIdcEq() {
entry:
  br label %if.else.i.i

if.else.i.i:                                      ; preds = %entry
  %tmp = load %struct1*, %struct1** undef, align 8
  br label %while.body.i99.i.i

while.body.i99.i.i:                               ; preds = %while.body.i99.i.i, %if.else.i.i
  br i1 false, label %while.body.i99.i.i, label %while.end.i103.i.i

while.end.i103.i.i:                               ; preds = %while.body.i99.i.i
  %tmp1 = load i32, i32* @cont_STACKPOINTER, align 4
  %dec.i.i102.i.i = add nsw i32 %tmp1, -1
  br i1 false, label %cont_BackTrack.exit107.i.i, label %if.then.i106.i.i

if.then.i106.i.i:                                 ; preds = %while.end.i103.i.i
  br label %cont_BackTrack.exit107.i.i

cont_BackTrack.exit107.i.i:                       ; preds = %if.then.i106.i.i, %while.end.i103.i.i
  %tmp2 = phi i32 [ %dec.i.i102.i.i, %if.then.i106.i.i ], [ 0, %while.end.i103.i.i ]
  %symbol.i.i.i = getelementptr inbounds %struct1, %struct1* %tmp, i64 0, i32 0
  br i1 undef, label %land.lhs.true23.i.i, label %for.inc.i.i

land.lhs.true23.i.i:                              ; preds = %cont_BackTrack.exit107.i.i
  %idxprom.i.i57.i.i = sext i32 %tmp2 to i64
  %arrayidx.i.i58.i.i = getelementptr inbounds [1000 x i32], [1000 x i32]* @cont_STACK, i64 0, i64 %idxprom.i.i57.i.i
  store i32 undef, i32* %arrayidx.i.i58.i.i, align 4
  br i1 false, label %if.then.i45.i.i, label %fol_Atom.exit47.i.i

if.then.i45.i.i:                                  ; preds = %land.lhs.true23.i.i
  br label %fol_Atom.exit47.i.i

fol_Atom.exit47.i.i:                              ; preds = %if.then.i45.i.i, %land.lhs.true23.i.i
  unreachable

for.inc.i.i:                                      ; preds = %cont_BackTrack.exit107.i.i
  br label %for.end.i.i

for.end.i.i:                                      ; preds = %for.inc.i.i
  ret void
}
