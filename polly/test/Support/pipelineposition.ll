; Legacy pass manager
; RUN: opt %loadPolly -O3 -enable-new-pm=0 -polly -polly-position=early                    -disable-output -debug-only=polly-scops < %s 2>&1 | FileCheck %s --check-prefix=NOINLINE
; RUN: opt %loadPolly -O3 -enable-new-pm=0 -polly -polly-position=early -polly-run-inliner -disable-output -debug-only=polly-scops < %s 2>&1 | FileCheck %s --check-prefix=INLINED1
; RUN: opt %loadPolly -O3 -enable-new-pm=0 -polly -polly-position=after-loopopt            -disable-output -debug-only=polly-scops < %s 2>&1 | FileCheck %s --check-prefix=INLINED2
; RUN: opt %loadPolly -O3 -enable-new-pm=0 -polly -polly-position=before-vectorizer        -disable-output -debug-only=polly-scops < %s 2>&1 | FileCheck %s --check-prefix=INLINED3
;
; New pass manager
; RUN: opt %loadPolly -O3 -enable-new-pm=1 -polly -polly-position=early                    -disable-output -debug-only=polly-scops < %s 2>&1 | FileCheck %s --check-prefix=NOINLINE
; RUN: opt %loadPolly -O3 -enable-new-pm=1 -polly -polly-position=early -polly-run-inliner -disable-output -debug-only=polly-scops < %s 2>&1 | FileCheck %s --check-prefix=INLINED1
; RUN: opt %loadPolly -O3 -enable-new-pm=1 -polly -polly-position=before-vectorizer        -disable-output -debug-only=polly-scops < %s 2>&1 | FileCheck %s --check-prefix=INLINED3
;
; REQUIRES: asserts
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


; NOINLINE-LABEL: Function: callee
; NOINLINE:       Schedule :=
; NOINLINE-NEXT:    [n, i] -> { Stmt_body[i0] -> [i0] };

; INLINED1-LABEL: Function: caller
; INLINED1:       Schedule :=
; INLINED1-NEXT:    [n] -> { Stmt_body_i[i0, i1] -> [i0, i1] };

; INLINED2-LABEL: Function: caller
; INLINED2:       Schedule :=
; INLINED2-NEXT:    [n] -> { Stmt_polly_loop_header_i_us_us[i0, i1] -> [i0, 1, i1] };

; INLINED3-LABEL: Function: caller
; INLINED3:       Schedule :=
; INLINED3-NEXT:    [n] -> { Stmt_body_i_us[i0, i1] -> [i0, i1] };
