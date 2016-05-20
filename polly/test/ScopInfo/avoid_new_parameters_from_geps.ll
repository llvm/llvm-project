; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; Check that we do no introduce a parameter here that is actually not needed.
;
; CHECK:      Region: %for.body58---%land.lhs.true
; CHECK-NEXT:     Max Loop Depth:  0
; CHECK-NEXT:     Invariant Accesses: {
; CHECK-NEXT:     }
; CHECK-NEXT:     Context:
; CHECK-NEXT:     {  :  }
; CHECK-NEXT:     Assumed Context:
; CHECK-NEXT:     {  :  }
; CHECK-NEXT:     Invalid Context:
; CHECK-NEXT:     {  : 1 = 0 }
; CHECK-NEXT:     Arrays {
; CHECK-NEXT:         i32* MemRef_team2_0_in; // Element size 8
; CHECK-NEXT:     }
; CHECK-NEXT:     Arrays (Bounds as pw_affs) {
; CHECK-NEXT:         i32* MemRef_team2_0_in; // Element size 8
; CHECK-NEXT:     }
; CHECK-NEXT:     Alias Groups (0):
; CHECK-NEXT:         n/a
; CHECK-NEXT:     Statements {
; CHECK-NEXT:     	Stmt_if_then60
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 { Stmt_if_then60[] };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 { Stmt_if_then60[] -> [] };
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_if_then60[] -> MemRef_team2_0_in[] };
; CHECK-NEXT:     }
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@sched = external global [18 x [15 x [3 x i32]]], align 16

; Function Attrs: nounwind uwtable
define void @common() #0 {
entry:
  br label %for.body36

for.body36:                                       ; preds = %entry
  br label %for.cond56.preheader

for.cond56.preheader:                             ; preds = %for.inc158, %for.body36
  %indvars.iv78 = phi i64 [ 0, %for.inc158 ], [ 1, %for.body36 ]
  br label %for.body58

for.body58:                                       ; preds = %for.cond56.preheader
  %cmp59 = icmp eq i32 1, 1
  br i1 %cmp59, label %if.then60, label %if.else71

if.then60:                                        ; preds = %for.body58
  %arrayidx70 = getelementptr inbounds [18 x [15 x [3 x i32]]], [18 x [15 x [3 x i32]]]* @sched, i64 0, i64 1, i64 %indvars.iv78, i64 1
  br label %land.lhs.true

if.else71:                                        ; preds = %for.body58
  br label %land.lhs.true

land.lhs.true:                                    ; preds = %if.else71, %if.then60
  %team2.0.in = phi i32* [ %arrayidx70, %if.then60 ], [ undef, %if.else71 ]
  br i1 undef, label %for.inc158, label %if.then86

if.then86:                                        ; preds = %land.lhs.true
  unreachable

for.inc158:                                       ; preds = %land.lhs.true
  br label %for.cond56.preheader
}
