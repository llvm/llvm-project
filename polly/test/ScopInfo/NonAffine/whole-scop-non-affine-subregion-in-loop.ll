; RUN: opt %loadPolly -polly-print-scops -disable-output < %s
;
; Regression test that triggered a memory leak at some point (24947).
;
define void @snrm2() {
entry:
  br label %for.body.56.lr.ph

for.body.56.lr.ph:                                ; preds = %entry
  br label %for.body.56

for.body.56:                                      ; preds = %for.inc.106, %for.body.56.lr.ph
  br label %if.end.73

if.end.73:                                        ; preds = %for.body.56
  %cmp82 = fcmp ogt float undef, undef
  br i1 %cmp82, label %if.then.84, label %if.end.100

if.then.84:                                       ; preds = %if.end.73
  br label %for.inc.106

if.end.100:                                       ; preds = %if.end.73
  br label %for.inc.106

for.inc.106:                                      ; preds = %if.end.100, %if.then.84
  br i1 undef, label %for.body.56, label %for.end.110

for.end.110:                                      ; preds = %for.inc.106
  ret void
}
