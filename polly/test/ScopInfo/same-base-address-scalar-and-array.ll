; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; Verify we introduce two ScopArrayInfo objects (or virtual arrays) for the %out variable
; as it is used as a memory base pointer (%0) but also as a scalar (%out.addr.0.lcssa).
;
; CHECK:         Arrays {
; CHECK-NEXT:        ptr MemRef_out_addr_0_lcssa; // Element size 8
; CHECK-NEXT:        float MemRef_out[*]; // Element size 4
; CHECK-NEXT:    }
;
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind ssp uwtable
define void @ff_celp_lp_synthesis_filterf(ptr %out) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br i1 false, label %for.end.97, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry.split
  %arrayidx13 = getelementptr inbounds float, ptr %out, i64 -3
  %0 = load float, ptr %arrayidx13, align 4
  br label %for.end.97

for.end.97:                                       ; preds = %for.cond.for.end.97_crit_edge, %entry.split
  %out.addr.0.lcssa = phi ptr [ undef, %for.body.lr.ph ], [ %out, %entry.split ]
  ret void
}
