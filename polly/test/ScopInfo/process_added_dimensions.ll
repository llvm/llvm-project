; RUN: opt %loadPolly -polly-detect -polly-scops -analyze < %s | FileCheck %s

; This test case produces the following memory access which we try hoist:
; { Stmt_for_cond40_preheader_5[i0] -> MemRef_call[0, 0, 2240] }. However, it
; accesses an array of size "i32 MemRef_call[*][6][64]".  That is why we
; should turn the whole SCoP into an invalid SCoP using corresponding bounds
; checks. Otherwise, we derive the incorrect access.

; CHECK: Valid Region for Scop: for.cond40.preheader.4 => for.end76
; CHECK-NOT:     Region: %for.cond40.preheader.4---%for.end76

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare noalias i8* @malloc()

define void @main() {
entry:
  %call = tail call noalias i8* @malloc()
  %0 = bitcast i8* %call to [6 x [6 x [64 x i32]]]*
  %arrayidx51.5.phi.trans.insert = getelementptr inbounds i8, i8* %call, i64 8960
  %1 = bitcast i8* %arrayidx51.5.phi.trans.insert to i32*
  br label %for.cond40.preheader.4

for.end76:                                        ; preds = %for.inc71.5
  ret void

for.cond40.preheader.4:                           ; preds = %for.inc71.5, %entry
  %t.0131 = phi i32 [ 0, %entry ], [ %inc75, %for.inc71.5 ]
  %indvars.iv.next135 = add nuw nsw i64 0, 1
  %2 = trunc i64 %indvars.iv.next135 to i32
  %indvars.iv.next = add nuw nsw i64 0, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 64
  %exitcond137 = icmp eq i32 %2, 6
  %indvars.iv.next135.1 = add nuw nsw i64 1, 1
  %indvars.iv.next.1 = add nuw nsw i64 0, 1
  %exitcond.1 = icmp eq i64 %indvars.iv.next.1, 64
  %lftr.wideiv.1 = trunc i64 %indvars.iv.next135.1 to i32
  %exitcond137.1 = icmp eq i32 %lftr.wideiv.1, 6
  %indvars.iv.next135.2 = add nuw nsw i64 2, 1
  %indvars.iv.next.2 = add nuw nsw i64 0, 1
  %exitcond.2 = icmp eq i64 %indvars.iv.next.2, 64
  %lftr.wideiv.2 = trunc i64 %indvars.iv.next135.2 to i32
  %exitcond137.2 = icmp eq i32 %lftr.wideiv.2, 6
  %indvars.iv.next135.3 = add nuw nsw i64 3, 1
  %indvars.iv.next.3 = add nuw nsw i64 0, 1
  %exitcond.3 = icmp eq i64 %indvars.iv.next.3, 64
  %lftr.wideiv.3 = trunc i64 %indvars.iv.next135.3 to i32
  %exitcond137.3 = icmp eq i32 %lftr.wideiv.3, 6
  %indvars.iv.next135.4 = add nuw nsw i64 4, 1
  %indvars.iv.next.4 = add nuw nsw i64 0, 1
  %exitcond.4 = icmp eq i64 %indvars.iv.next.4, 64
  %lftr.wideiv.4 = trunc i64 %indvars.iv.next135.4 to i32
  %exitcond137.4 = icmp eq i32 %lftr.wideiv.4, 6
  %arrayidx23.5 = getelementptr inbounds [6 x [6 x [64 x i32]]], [6 x [6 x [64 x i32]]]* %0, i64 0, i64 5, i64 5, i64 0
  store i32 36, i32* %arrayidx23.5, align 4
  %indvars.iv.next.5 = add nuw nsw i64 0, 1
  %exitcond.5 = icmp eq i64 %indvars.iv.next.5, 64
  %indvars.iv.next143 = add nuw nsw i64 1, 1
  %exitcond145 = icmp eq i64 %indvars.iv.next143, 64
  %indvars.iv.next149 = add nuw nsw i64 0, 1
  %lftr.wideiv150 = trunc i64 %indvars.iv.next149 to i32
  %exitcond151 = icmp eq i32 %lftr.wideiv150, 6
  %indvars.iv.next143.1 = add nuw nsw i64 1, 1
  %exitcond145.1 = icmp eq i64 %indvars.iv.next143.1, 64
  %indvars.iv.next149.1 = add nuw nsw i64 1, 1
  %lftr.wideiv150.1 = trunc i64 %indvars.iv.next149.1 to i32
  %exitcond151.1 = icmp eq i32 %lftr.wideiv150.1, 6
  %indvars.iv.next143.2 = add nuw nsw i64 1, 1
  %exitcond145.2 = icmp eq i64 %indvars.iv.next143.2, 64
  %indvars.iv.next149.2 = add nuw nsw i64 2, 1
  %lftr.wideiv150.2 = trunc i64 %indvars.iv.next149.2 to i32
  %exitcond151.2 = icmp eq i32 %lftr.wideiv150.2, 6
  %indvars.iv.next143.3 = add nuw nsw i64 1, 1
  %exitcond145.3 = icmp eq i64 %indvars.iv.next143.3, 64
  %indvars.iv.next149.3 = add nuw nsw i64 3, 1
  %lftr.wideiv150.3 = trunc i64 %indvars.iv.next149.3 to i32
  %exitcond151.3 = icmp eq i32 %lftr.wideiv150.3, 6
  br label %for.body44.4

for.body44.4:                                     ; preds = %for.body44.4, %for.cond40.preheader.4
  %indvars.iv142.4 = phi i64 [ 1, %for.cond40.preheader.4 ], [ %indvars.iv.next143.4, %for.body44.4 ]
  %indvars.iv.next143.4 = add nuw nsw i64 %indvars.iv142.4, 1
  %exitcond145.4 = icmp eq i64 %indvars.iv.next143.4, 64
  br i1 %exitcond145.4, label %for.cond40.preheader.5, label %for.body44.4

for.cond40.preheader.5:                           ; preds = %for.body44.4
  %indvars.iv.next149.4 = add nuw nsw i64 4, 1
  %lftr.wideiv150.4 = trunc i64 %indvars.iv.next149.4 to i32
  %exitcond151.4 = icmp eq i32 %lftr.wideiv150.4, 6
  %.pre160 = load i32, i32* %1, align 4
  br label %for.body44.5

for.body44.5:                                     ; preds = %for.body44.5, %for.cond40.preheader.5
  %indvars.iv142.5 = phi i64 [ 1, %for.cond40.preheader.5 ], [ %indvars.iv.next143.5, %for.body44.5 ]
  %indvars.iv.next143.5 = add nuw nsw i64 %indvars.iv142.5, 1
  %exitcond145.5 = icmp eq i64 %indvars.iv.next143.5, 64
  br i1 %exitcond145.5, label %for.inc71.5, label %for.body44.5

for.inc71.5:                                      ; preds = %for.body44.5
  %inc75 = add nuw nsw i32 %t.0131, 1
  %exitcond155 = icmp eq i32 %inc75, 2
  br i1 %exitcond155, label %for.end76, label %for.cond40.preheader.4
}
