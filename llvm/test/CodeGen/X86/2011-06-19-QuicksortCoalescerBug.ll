; RUN: llc < %s -verify-coalescing
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define void @Quicksort(ptr %a, i32 %l, i32 %r) nounwind ssp {
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %do.cond, %entry
  %l.tr = phi i32 [ %l, %entry ], [ %i.1, %do.cond ]
  %r.tr = phi i32 [ %r, %entry ], [ %l.tr, %do.cond ]
  %idxprom12 = sext i32 %r.tr to i64
  %arrayidx14 = getelementptr inbounds i32, ptr %a, i64 %idxprom12
  br label %do.body

do.body:                                          ; preds = %do.cond, %tailrecurse
  %i.0 = phi i32 [ %l.tr, %tailrecurse ], [ %i.1, %do.cond ]
  %add7 = add nsw i32 %i.0, 1
  %cmp = icmp sgt i32 %add7, %r.tr
  br i1 %cmp, label %do.cond, label %if.then

if.then:                                          ; preds = %do.body
  store i32 %add7, ptr %arrayidx14, align 4
  %add16 = add i32 %i.0, 2
  br label %do.cond

do.cond:                                          ; preds = %do.body, %if.then
  %i.1 = phi i32 [ %add16, %if.then ], [ %add7, %do.body ]
  %cmp19 = icmp sgt i32 %i.1, %r.tr
  br i1 %cmp19, label %tailrecurse, label %do.body
}
