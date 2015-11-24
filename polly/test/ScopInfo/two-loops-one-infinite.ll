; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; Verify we detect and create the SCoP correctly
;
; CHECK:      Statements {
; CHECK-NEXT:   Stmt_while_body_us
; CHECK-NEXT:     Domain :=
; CHECK-NEXT:       [a13] -> { Stmt_while_body_us[] };
; CHECK-NEXT:     Schedule :=
; CHECK-NEXT:       [a13] -> { Stmt_while_body_us[] -> [] };
; CHECK-NEXT:     MustWriteAccess := [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:       [a13] -> { Stmt_while_body_us[] -> MemRef_uuu[] };
; CHECK-NEXT: }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n8:16:32-S64"

define void @foo(i32* noalias nocapture readonly %xxx, i32* noalias nocapture readonly %yyy, i8*** nocapture readonly %zzz, i32 %conv6) {
while.body.us.preheader:
 %a2 = load i8**, i8*** %zzz, align 4  
 %sub = add nsw i32 %conv6, -1 
  br label %while.body.us

while.body.us:                                    ; preds = %while.body.us.preheader, %if.then.us
  %uuu = phi i32 [ %www, %if.then.us ], [ 0, %while.body.us.preheader ]
  %a13 = load i32, i32* %yyy, align 8
  %vvv = icmp sgt i32 %a13, 0
  br i1 %vvv, label %while.body.13.us58.preheader, label %if.then.us

while.body.13.us58.preheader:                     ; preds = %while.body.us
  br label %while.body.13.us58

if.then.us:                                       ; preds = %while.body.us
  %add.us = add nuw nsw i32 %uuu, 1
  tail call void @goo(i8** %a2, i32 %uuu, i8** %a2, i32 %add.us, i32 %sub, i32 %a13) #3
  %www = add nuw nsw i32 %uuu, %conv6
  %a14 = load i32, i32* %xxx, align 4
  %cmp.us = icmp slt i32 %www, %a14
  br i1 %cmp.us, label %while.body.us, label %while.end.21.loopexit145

while.body.13.us58:
    br label %while.body.13.us58

while.end.21.loopexit145:
  ret void
}

declare void @goo(i8**, i32, i8**, i32, i32, i32) #1

