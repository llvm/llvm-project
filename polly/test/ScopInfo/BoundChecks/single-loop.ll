; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=print<polly-ast>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=AST
;
; This only works after the post-dominator tree has been fixed.
;
; XFAIL: *
;
;    void exception() __attribute__((noreturn));
;
;    void foo(long n, float A[100]) {
;      for (long i = 0; i < n; i++) {
;        if (i < 0)
;          exception();
;
;        if (i >= 100)
;          exception();
;
;        A[i] += i;
;      }
;    }

; We should detect this kernel as a SCoP and derive run-time conditions such
; that the bound-checked blocks are not part of the optimized SCoP.

; CHECK: Invalid Context:
; CHECK:  [n] -> {  : n >= 101 }

; AST: if (1 && 0 == n >= 101)
; AST:     for (int c0 = 0; c0 < n; c0 += 1)
; AST:       Stmt_if_end_4(c0);
;
; AST-NOT: for
; AST-NOT: Stmt
;
; AST: else
; AST:     {  /* original code */ }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @foo(i64 %n, ptr %A) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i64 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br i1 false, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  call void (...) @exception() #2
  unreachable

if.end:                                           ; preds = %for.body
  %cmp2 = icmp sgt i64 %i.0, 99
  br i1 %cmp2, label %if.then.3, label %if.end.4

if.then.3:                                        ; preds = %if.end
  call void (...) @exception() #2
  unreachable

if.end.4:                                         ; preds = %if.end
  %conv = sitofp i64 %i.0 to float
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %i.0
  %tmp = load float, ptr %arrayidx, align 4
  %add = fadd float %tmp, %conv
  store float %add, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end.4
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: noreturn
declare void @exception(...) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noreturn nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 246853)"}
