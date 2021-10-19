; RUN: opt %loadPolly -polly-vectorizer=polly -polly-opt-isl -polly-codegen -S < %s | FileCheck %s
;
; Polly crashed during codegen with an assertion error while trying to generate
; a pointer bitcast from a pointer having an address space to one without
;
; CHECK-LABEL: entry:
; CHECK: load <4 x float>, <4 x float> addrspace(4)*
;
; ModuleID = '/tmp/lud.bc'
source_filename = "lud.c"
; This datalayout was for a 32-bit ARC processor with 512-bit vector extension
target datalayout = "e-m:e-p:32:32-p1:32:32-p3:32:32-p5:32:32-i64:32-f64:32-v64:32-v128:32-a:0:32-v256:32-v512:32-n8:16:32"
; Specify x86 because the ARC backend is still experimental and not built by default
target triple = "x86_64-unknown-unknown"

; Function Attrs: noinline nounwind
define void @LU_decomp_kij_opt(i32 %n, i32 %lda, float addrspace(4)* %A, float addrspace(4)* %scratch) #0 {
entry:
  %cmp34 = icmp sgt i32 %n, 0
  br i1 %cmp34, label %for.body.lr.ph, label %for.end34

for.body.lr.ph:                                   ; preds = %entry
  %0 = add nsw i32 %n, -1
  br label %for.body

for.body:                                         ; preds = %for.inc32, %for.body.lr.ph
  %k.035 = phi i32 [ 0, %for.body.lr.ph ], [ %add2, %for.inc32 ]
  %mul = mul nsw i32 %k.035, %lda
  %add = add nsw i32 %mul, %k.035
  %arrayidx = getelementptr inbounds float, float addrspace(4)* %A, i32 %add
  %1 = load float, float addrspace(4)* %arrayidx, align 4
  %conv1 = fdiv arcp float 1.000000e+00, %1
  %add2 = add nuw nsw i32 %k.035, 1
  %exitcond37 = icmp eq i32 %k.035, %0
  br i1 %exitcond37, label %for.end34, label %for.body6.lr.ph

for.body6.lr.ph:                                  ; preds = %for.body
  br label %for.body6

for.body6:                                        ; preds = %for.inc29, %for.body6.lr.ph
  %i.033 = phi i32 [ %add2, %for.body6.lr.ph ], [ %inc30, %for.inc29 ]
  %mul7 = mul nsw i32 %i.033, %lda
  %add8 = add nsw i32 %mul7, %k.035
  %arrayidx9 = getelementptr inbounds float, float addrspace(4)* %A, i32 %add8
  %2 = load float, float addrspace(4)* %arrayidx9, align 4
  %mul10 = fmul arcp contract float %conv1, %2
  store float %mul10, float addrspace(4)* %arrayidx9, align 4
  br label %for.body18

for.body18:                                       ; preds = %for.body18, %for.body6
  %j.031 = phi i32 [ %add2, %for.body6 ], [ %inc, %for.body18 ]
  %3 = load float, float addrspace(4)* %arrayidx9, align 4
  %add23 = add nsw i32 %j.031, %mul
  %arrayidx24 = getelementptr inbounds float, float addrspace(4)* %A, i32 %add23
  %4 = load float, float addrspace(4)* %arrayidx24, align 4
  %mul25 = fmul arcp contract float %3, %4
  %add27 = add nsw i32 %j.031, %mul7
  %arrayidx28 = getelementptr inbounds float, float addrspace(4)* %A, i32 %add27
  %5 = load float, float addrspace(4)* %arrayidx28, align 4
  %sub = fsub arcp contract float %5, %mul25
  store float %sub, float addrspace(4)* %arrayidx28, align 4
  %inc = add nuw nsw i32 %j.031, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.inc29, label %for.body18

for.inc29:                                        ; preds = %for.body18
  %inc30 = add nuw nsw i32 %i.033, 1
  %exitcond36 = icmp eq i32 %inc30, %n
  br i1 %exitcond36, label %for.inc32, label %for.body6

for.inc32:                                        ; preds = %for.inc29
  br label %for.body

for.end34:                                        ; preds = %for.body, %entry
  ret void
}

attributes #0 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"ArcIntrinsicCheck", i32 18224056}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{!"clang version 10.0.1 "}
