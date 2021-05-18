; Legacy pass manager
; RUN: opt %loadPolly -enable-new-pm=0 -O3 -polly -polly-position=early             -polly-dump-before-file=%t-legacy-before-early.ll --disable-output < %s && FileCheck --input-file=%t-legacy-before-early.ll --check-prefix=EARLY  %s
; RUN: opt %loadPolly -enable-new-pm=0 -O3 -polly -polly-position=before-vectorizer -polly-dump-before-file=%t-legacy-before-late.ll  --disable-output < %s && FileCheck --input-file=%t-legacy-before-late.ll  --check-prefix=LATE   %s
; RUN: opt %loadPolly -enable-new-pm=0 -O3 -polly -polly-position=early             -polly-dump-after-file=%t-legacy-after-early.ll   --disable-output < %s && FileCheck --input-file=%t-legacy-after-early.ll  --check-prefix=EARLY --check-prefix=AFTEREARLY %s
; RUN: opt %loadPolly -enable-new-pm=0 -O3 -polly -polly-position=before-vectorizer -polly-dump-after-file=%t-legacy-after-late.ll    --disable-output < %s && FileCheck --input-file=%t-legacy-after-late.ll   --check-prefix=LATE  --check-prefix=AFTERLATE  %s
;-
; New pass manager
; RUN: opt %loadPolly -enable-new-pm=1 -O3 -polly -polly-position=early             -polly-dump-before-file=%t-npm-before-early.ll    --disable-output < %s && FileCheck --input-file=%t-npm-before-early.ll    --check-prefix=EARLY %s
; RUN: opt %loadPolly -enable-new-pm=1 -O3 -polly -polly-position=early             -polly-dump-after-file=%t-npm-after-early.ll      --disable-output < %s && FileCheck --input-file=%t-npm-after-early.ll     --check-prefix=EARLY --check-prefix=AFTEREARLY %s
;
; Check the module dumping before Polly at specific positions in the
; pass pipeline.
;
; void callee(int n, double A[], int i) {
;   for (int j = 0; j < n; j += 1)
;     A[i+j] = 42.0;
; }
;
; void caller(int n, double A[]) {
;   for (int i = 0; i < n; i += 1)
;     callee(n, A, i);
; }


define internal void @callee(i32 %n, double* noalias nonnull %A, i32 %i) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      %idx = add i32 %i, %j
      %arrayidx = getelementptr inbounds double, double* %A, i32 %idx
      store double 42.0, double* %arrayidx
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


define void @caller(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %i = phi i32 [0, %entry], [%j.inc, %inc]
  %i.cmp = icmp slt i32 %i, %n
  br i1 %i.cmp, label %body, label %exit

    body:
      call void @callee(i32 %n, double* %A, i32 %i)
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %i, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; EARLY-LABEL: @callee(
; AFTEREARLY:  polly.split_new_and_old:
; EARLY:         store double 4.200000e+01, double* %arrayidx
; EARLY-LABEL: @caller(
; EARLY:         call void @callee(

; LATE-LABEL: @caller(
; AFTERLATE:  polly.split_new_and_old:
; LATE:          store double 4.200000e+01, double* %arrayidx
