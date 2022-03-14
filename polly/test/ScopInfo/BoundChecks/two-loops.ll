; RUN: opt %loadPolly -polly-print-scops -disable-output< %s | FileCheck %s
; RUN: opt %loadPolly -polly-print-ast -disable-output < %s | FileCheck %s --check-prefix=AST
;
; This only works after the post-dominator tree has fixed.
; XFAIL: *
;
;    void exception() __attribute__((noreturn));
;
;    void foo(long n, float A[100]) {
;      for (long j = 0; j < n; j++) {
;        for (long i = j; i < n; i++) {
;          if (i < 0)
;            exception();
;
;          if (i >= 100)
;            exception();
;
;          A[i] += i;
;        }
;      }
;    }
;
; CHECK: Assumed Context:
; CHECK:  [n] -> {  : n >= 101 }

; AST: if (1 && 0 == n >= 101)
; AST:     for (int c0 = 0; c0 < n; c0 += 1)
; AST:       for (int c1 = 0; c1 < n - c0; c1 += 1)
; AST:         Stmt_if_end_7(c0, c1);
;
; AST-NOT: for
; AST-NOT: Stmt
;
; AST: else
; AST:     {  /* original code */ }
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64 %n, float* %A) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc.8, %entry
  %j.0 = phi i64 [ 0, %entry ], [ %inc9, %for.inc.8 ]
  %cmp = icmp slt i64 %j.0, %n
  br i1 %cmp, label %for.body, label %for.end.10

for.body:                                         ; preds = %for.cond
  br label %for.cond.1

for.cond.1:                                       ; preds = %for.inc, %for.body
  %i.0 = phi i64 [ %j.0, %for.body ], [ %inc, %for.inc ]
  %cmp2 = icmp slt i64 %i.0, %n
  br i1 %cmp2, label %for.body.3, label %for.end

for.body.3:                                       ; preds = %for.cond.1
  br i1 false, label %if.then, label %if.end

if.then:                                          ; preds = %for.body.3
  call void (...) @exception() #2
  unreachable

if.end:                                           ; preds = %for.body.3
  %cmp5 = icmp sgt i64 %i.0, 99
  br i1 %cmp5, label %if.then.6, label %if.end.7

if.then.6:                                        ; preds = %if.end
  call void (...) @exception() #2
  unreachable

if.end.7:                                         ; preds = %if.end
  %conv = sitofp i64 %i.0 to float
  %arrayidx = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp = load float, float* %arrayidx, align 4
  %add = fadd float %tmp, %conv
  store float %add, float* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end.7
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond.1

for.end:                                          ; preds = %for.cond.1
  br label %for.inc.8

for.inc.8:                                        ; preds = %for.end
  %inc9 = add nuw nsw i64 %j.0, 1
  br label %for.cond

for.end.10:                                       ; preds = %for.cond
  ret void
}

declare void @exception(...) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noreturn nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 246853)"}
