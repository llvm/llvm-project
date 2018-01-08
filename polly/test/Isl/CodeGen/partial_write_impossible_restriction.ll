; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-codegen -S < %s | FileCheck %s
;
; The isl scheduler isolates %cond.false into two instances.
; A partial write access in one of the instances was never executed,
; which caused problems when querying for its index expression, which
; is not available in that case.
;
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define void @partial_write_impossible_restriction() {
entry:
  br i1 undef, label %invoke.cont258, label %cond.true.i.i.i.i1007

cond.true.i.i.i.i1007:
  br label %invoke.cont258

invoke.cont258:
  %.pn = phi i32* [ null, %cond.true.i.i.i.i1007 ], [ null, %entry ]
  br label %invoke.cont274

invoke.cont274:                                   ; preds = %invoke.cont258
  %tmp4 = load i32*, i32** undef
  %tmp5 = load i32, i32* undef
  %tmp6 = zext i32 %tmp5 to i64
  %tmp7 = sext i32 %tmp5 to i64
  br label %for.body344

for.body344:                                      ; preds = %cond.end, %invoke.cont274
  %indvars.iv1602 = phi i64 [ 0, %invoke.cont274 ], [ %indvars.iv.next1603, %cond.end ]
  %indvars.iv.next1603 = add nuw nsw i64 %indvars.iv1602, 1
  %cmp347 = icmp eq i64 %indvars.iv.next1603, %tmp6
  br i1 %cmp347, label %cond.end, label %cond.false

cond.false:                                       ; preds = %for.body344
  %add.ptr.i1128 = getelementptr inbounds i32, i32* %tmp4, i64 %indvars.iv.next1603
  %cond.in.sroa.speculate.load.cond.false = load i32, i32* %add.ptr.i1128
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %for.body344
  %cond.in.sroa.speculated = phi i32 [ %cond.in.sroa.speculate.load.cond.false, %cond.false ], [ undef, %for.body344 ]
  %add.ptr.i1132 = getelementptr inbounds i32, i32* %.pn, i64 %indvars.iv1602
  store i32 undef, i32* %add.ptr.i1132
  %cmp342 = icmp slt i64 %indvars.iv.next1603, %tmp7
  br i1 %cmp342, label %for.body344, label %if.then.i.i1141.loopexit

if.then.i.i1141.loopexit:                         ; preds = %cond.end
  ret void
}


; CHECK-LABEL: polly.stmt.cond.false:
; CHECK:         %polly.access..pn2 = getelementptr i32, i32* %.pn, i64 %polly.indvar
; CHECK:         store i32 %cond.in.sroa.speculate.load.cond.false_p_scalar_, i32* %polly.access..pn2, !alias.scope !0, !noalias !2
; CHECK:         br label %polly.merge

; CHECK-LABEL: polly.stmt.cond.false11:
; CHECK:         %polly.access..pn14 = getelementptr i32, i32* %.pn, i64 0
; CHECK:         store i32 %cond.in.sroa.speculate.load.cond.false_p_scalar_13, i32* %polly.access..pn14, !alias.scope !0, !noalias !2
; CHECK:         br label %polly.stmt.cond.end15
