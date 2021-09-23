; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; This caused an lnt crash at some point, just verify it will run through and
; produce the PHI node in the exit we are looking for.
;
; CHECK-LABEL: polly.merge_new_and_old:
; CHECK-NEXT:    %eps1.addr.0.ph.merge = phi double [ %eps1.addr.0.ph.final_reload, %polly.exiting ], [ %eps1.addr.0.ph, %if.end.47.region_exiting ]
; CHECK-NEXT:      br label %if.end.47
;
; CHECK-LABEL: if.end.47:
; CHECK-NEXT:        %eps1.addr.0 = phi double [ %eps1.addr.0.ph.merge, %polly.merge_new_and_old ]
;
define void @dbisect(double* %c, double* %b, double %eps1, double* %eps2) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  store double 0.000000e+00, double* %b, align 8
  %arrayidx9 = getelementptr inbounds double, double* %c, i64 0
  %0 = load double, double* %arrayidx9, align 8
  br i1 false, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry.split
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  br i1 false, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br i1 false, label %if.then.36, label %for.inc

if.then.36:                                       ; preds = %if.end
  br label %for.inc

for.inc:                                          ; preds = %if.then.36, %if.end
  br i1 false, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.inc
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  store double undef, double* %eps2, align 8
  %cmp45 = fcmp ugt double %eps1, 0.000000e+00
  br i1 %cmp45, label %if.end.47, label %if.then.46

if.then.46:                                       ; preds = %for.end
  br label %if.end.47

if.end.47:                                        ; preds = %if.then.46, %for.end
  %eps1.addr.0 = phi double [ undef, %if.then.46 ], [ %eps1, %for.end ]
  br i1 undef, label %if.then.55, label %for.cond.58.preheader

for.cond.58.preheader:                            ; preds = %if.end.47
  br i1 undef, label %for.end.68, label %for.body.61.lr.ph

for.body.61.lr.ph:                                ; preds = %for.cond.58.preheader
  br label %for.body.61

if.then.55:                                       ; preds = %if.end.47
  unreachable

for.body.61:                                      ; preds = %for.body.61, %for.body.61.lr.ph
  br i1 undef, label %for.body.61, label %for.cond.58.for.end.68_crit_edge

for.cond.58.for.end.68_crit_edge:                 ; preds = %for.body.61
  br label %for.end.68

for.end.68:                                       ; preds = %for.cond.58.for.end.68_crit_edge, %for.cond.58.preheader
  br i1 undef, label %for.end.137, label %for.cond.73.preheader.lr.ph

for.cond.73.preheader.lr.ph:                      ; preds = %for.end.68
  br label %for.cond.73.preheader

for.cond.73.preheader:                            ; preds = %while.end, %for.cond.73.preheader.lr.ph
  br i1 undef, label %for.end.87.loopexit, label %for.body.76.lr.ph

for.body.76.lr.ph:                                ; preds = %for.cond.73.preheader
  br label %for.body.76

for.body.76:                                      ; preds = %for.inc.85, %for.body.76.lr.ph
  br i1 undef, label %if.then.81, label %for.inc.85

if.then.81:                                       ; preds = %for.body.76
  br label %for.end.87

for.inc.85:                                       ; preds = %for.body.76
  br i1 undef, label %for.body.76, label %for.cond.73.for.end.87.loopexit_crit_edge

for.cond.73.for.end.87.loopexit_crit_edge:        ; preds = %for.inc.85
  br label %for.end.87.loopexit

for.end.87.loopexit:                              ; preds = %for.cond.73.for.end.87.loopexit_crit_edge, %for.cond.73.preheader
  br label %for.end.87

for.end.87:                                       ; preds = %for.end.87.loopexit, %if.then.81
  br i1 undef, label %if.then.92, label %if.end.95

if.then.92:                                       ; preds = %for.end.87
  br label %if.end.95

if.end.95:                                        ; preds = %if.then.92, %for.end.87
  br i1 undef, label %while.body.lr.ph, label %while.end

while.body.lr.ph:                                 ; preds = %if.end.95
  br label %while.body

while.body:                                       ; preds = %if.end.128, %while.body.lr.ph
  br i1 undef, label %if.then.109, label %if.end.128

if.then.109:                                      ; preds = %while.body
  br i1 undef, label %if.then.112, label %if.else

if.then.112:                                      ; preds = %if.then.109
  br label %if.end.128

if.else:                                          ; preds = %if.then.109
  br i1 undef, label %if.then.122, label %if.end.128

if.then.122:                                      ; preds = %if.else
  br label %if.end.128

if.end.128:                                       ; preds = %if.then.122, %if.else, %if.then.112, %while.body
  br i1 undef, label %while.body, label %while.cond.while.end_crit_edge

while.cond.while.end_crit_edge:                   ; preds = %if.end.128
  br label %while.end

while.end:                                        ; preds = %while.cond.while.end_crit_edge, %if.end.95
  br i1 undef, label %for.cond.73.preheader, label %for.cond.69.for.end.137_crit_edge

for.cond.69.for.end.137_crit_edge:                ; preds = %while.end
  br label %for.end.137

for.end.137:                                      ; preds = %for.cond.69.for.end.137_crit_edge, %for.end.68
  ret void
}
