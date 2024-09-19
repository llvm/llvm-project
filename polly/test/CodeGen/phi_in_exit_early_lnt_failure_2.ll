; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s
;
; This caused an lnt crash at some point, just verify it will run through and
; produce the PHI node in the exit we are looking for.
;
; CHECK:       %eps1.addr.0.s2a = alloca double
; CHECK-NOT:   %eps1.addr.0.ph.s2a = alloca double
;
; CHECK-LABEL: polly.merge_new_and_old:
; CHECK:          %eps1.addr.0.ph.merge = phi double [ %eps1.addr.0.ph.final_reload, %polly.exiting ], [ %eps1.addr.0.ph, %if.end.47.region_exiting ]
;
; CHECK-LABEL: polly.start:
; CHECK-NEXT:    store double %eps1, ptr %eps1.s2a
;
; CHECK-LABEL: polly.exiting:
; CHECK-NEXT:     %eps1.addr.0.ph.final_reload = load double, ptr %eps1.addr.0.s2a
;
define void @dbisect(ptr %c, ptr %b, double %eps1, ptr %eps2) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  store double 0.000000e+00, ptr %b, align 8
  br i1 false, label %for.inc, label %for.end

if.end:                                           ; preds = %if.then, %for.body
  %0 = load double, ptr %c, align 8
  br label %for.inc

for.inc:                                          ; preds = %if.then.36, %if.end
  br i1 false, label %if.end, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.inc
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  %cmp45 = fcmp ugt double %eps1, 0.000000e+00
  br i1 %cmp45, label %if.end.47, label %if.then.46

if.then.46:                                       ; preds = %for.end
  %1 = load double, ptr %eps2, align 8
  br label %if.end.47

if.end.47:                                        ; preds = %if.then.46, %for.end
  %eps1.addr.0 = phi double [ %1, %if.then.46 ], [ %eps1, %for.end ]
  ret void
}
