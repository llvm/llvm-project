; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; CHECK:      if.then260:
; CHECK-NEXT:   %p.4 = getelementptr inbounds i8, i8* null, i64 1
; CHECK-NEXT:   %sub.ptr.lhs.cast263 = ptrtoint i8* %p.4 to i64
; CHECK-NEXT:   %sub.ptr.sub265 = sub i64 %sub.ptr.lhs.cast263, 0
; CHECK-NEXT:   %div = udiv i64 0, %sub.ptr.sub265
; CHECK-NEXT:   %cmp268 = icmp ult i64 0, %div
; CHECK-NEXT:   br i1 %cmp268, label %cond.true270, label %while.cond.region_exiting
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @XS_MIME__QuotedPrint_encode_qp() {
entry:
  %Perl_sv_len = alloca i64, align 8
  br label %if.end

if.end:                                           ; preds = %entry
  br label %while.cond

while.cond:                                       ; preds = %cond.true270, %if.then260, %if.end
  %p.0 = phi i8* [ null, %if.end ], [ %p.4, %if.then260 ], [ %p.4, %cond.true270 ]
  br i1 undef, label %if.then260, label %while.body210

while.body210:                                    ; preds = %while.cond
  ret void

if.then260:                                       ; preds = %while.cond
  %p.4 = getelementptr inbounds i8, i8* null, i64 1
  %sub.ptr.lhs.cast263 = ptrtoint i8* %p.4 to i64
  %sub.ptr.sub265 = sub i64 %sub.ptr.lhs.cast263, 0
  %div = udiv i64 0, %sub.ptr.sub265
  %cmp268 = icmp ult i64 0, %div
  br i1 %cmp268, label %cond.true270, label %while.cond

cond.true270:                                     ; preds = %if.then260
  br label %while.cond
}
