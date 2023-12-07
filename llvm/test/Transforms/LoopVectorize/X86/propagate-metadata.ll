; RUN: opt -S -mtriple="x86_64-unknown-linux-gnu" -passes=loop-vectorize < %s | FileCheck %s

; Don't crash on unknown metadata
; CHECK-LABEL: @no_propagate_range_metadata(
; CHECK: load <16 x i8>
; CHECK: store <16 x i8>
define void @no_propagate_range_metadata(ptr readonly %first.coerce, ptr readnone %last.coerce, ptr nocapture %result) {
for.body.preheader:
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %result.addr.05 = phi ptr [ %incdec.ptr, %for.body ], [ %result, %for.body.preheader ]
  %first.sroa.0.04 = phi ptr [ %incdec.ptr.i.i.i, %for.body ], [ %first.coerce, %for.body.preheader ]
  %0 = load i8, ptr %first.sroa.0.04, align 1, !range !0
  store i8 %0, ptr %result.addr.05, align 1
  %incdec.ptr.i.i.i = getelementptr inbounds i8, ptr %first.sroa.0.04, i64 1
  %incdec.ptr = getelementptr inbounds i8, ptr %result.addr.05, i64 1
  %lnot.i = icmp eq ptr %incdec.ptr.i.i.i, %last.coerce
  br i1 %lnot.i, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  ret void
}

!0 = !{i8 0, i8 2}
