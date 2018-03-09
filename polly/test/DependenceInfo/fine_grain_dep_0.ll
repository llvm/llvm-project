; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-dependences -polly-dependences-analysis-type=value-based -polly-dependences-analysis-level=reference-wise -analyze < %s | FileCheck %s --check-prefix=REF
; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-dependences -polly-dependences-analysis-type=value-based -polly-dependences-analysis-level=access-wise -analyze < %s | FileCheck %s --check-prefix=ACC
; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-function-dependences -polly-dependences-analysis-type=value-based -polly-dependences-analysis-level=access-wise -analyze < %s | FileCheck %s --check-prefix=ACC
;
; REF:      RAW dependences:
; REF-NEXT:     [N] -> { Stmt_for_body[i0] -> Stmt_for_body[6 + i0] : 0 <= i0 <= -13 + N; Stmt_for_body[i0] -> Stmt_for_body[4 + i0] : 0 <= i0 <= -11 + N; [Stmt_for_body[i0] -> MemRef_a[]] -> [Stmt_for_body[4 + i0] -> MemRef_a[]] : 0 <= i0 <= -11 + N; [Stmt_for_body[i0] -> MemRef_b[]] -> [Stmt_for_body[6 + i0] -> MemRef_b[]] : 0 <= i0 <= -13 + N }
; REF-NEXT: WAR dependences:
; REF-NEXT:     {  }
; REF-NEXT: WAW dependences:
; REF-NEXT:     {  }
; REF-NEXT: Reduction dependences:
; REF-NEXT:     {  }

; ACC:      RAW dependences:
; ACC-NEXT:   [N] -> { Stmt_for_body[i0] -> Stmt_for_body[6 + i0] : 0 <= i0 <= -13 + N; Stmt_for_body[i0] -> Stmt_for_body[4 + i0] : 0 <= i0 <= -11 + N; [Stmt_for_body[i0] -> Stmt_for_body_Write1[]] -> [Stmt_for_body[4 + i0] -> Stmt_for_body_Read0[]] : 0 <= i0 <= -11 + N; [Stmt_for_body[i0] -> Stmt_for_body_Write3[]] -> [Stmt_for_body[6 + i0] -> Stmt_for_body_Read2[]] : 0 <= i0 <= -13 + N }

; ACC-NEXT: WAR dependences:
; ACC-NEXT:   [N] -> {  }
; ACC-NEXT: WAW dependences:
; ACC-NEXT:   [N] -> {  }
; ACC-NEXT: Reduction dependences:
; ACC-NEXT:   [N] -> {  }

; void test(char a[], char b[], long N) {
;   for (long i = 6; i < N; ++i) {
;     a[i] = a[i - 4] + i;
;     b[i] = b[i - 6] + i;
;   }
; }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @test(i8* %a, i8* %b, i64 %N) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 6, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i64 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %sub = sub nsw i64 %i.0, 4
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %sub
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %0 to i64
  %add = add nsw i64 %conv, %i.0
  %conv1 = trunc i64 %add to i8
  %arrayidx2 = getelementptr inbounds i8, i8* %a, i64 %i.0
  store i8 %conv1, i8* %arrayidx2, align 1
  %sub3 = sub nsw i64 %i.0, 6
  %arrayidx4 = getelementptr inbounds i8, i8* %b, i64 %sub3
  %1 = load i8, i8* %arrayidx4, align 1
  %conv5 = sext i8 %1 to i64
  %add6 = add nsw i64 %conv5, %i.0
  %conv7 = trunc i64 %add6 to i8
  %arrayidx8 = getelementptr inbounds i8, i8* %b, i64 %i.0
  store i8 %conv7, i8* %arrayidx8, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0 (http://llvm.org/git/clang.git 3d5d4c39659f11dfbe8e11c857cadf5c449b559b) (http://llvm.org/git/llvm.git 801561e2bba12f2aa0285feb1105e110df443761)"}
