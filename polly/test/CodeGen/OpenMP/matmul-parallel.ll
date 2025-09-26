; RUN: opt %loadNPMPolly -polly-parallel '-passes=polly-opt-isl,print<polly-ast>' -disable-output -debug-only=polly-ast < %s 2>&1 | FileCheck --check-prefix=AST %s
; RUN: opt %loadNPMPolly -polly-parallel '-passes=polly-opt-isl,polly-codegen' -S < %s | FileCheck --check-prefix=CODEGEN %s
; REQUIRES: asserts

; Parallelization of detected matrix-multiplication.
; The outer loop should be parallelized.
; AST: // 1st level tiling - Tiles
; AST-NEXT: #pragma omp parallel for

; CODEGEN: polly.parallel.for

define i32 @foo(ptr nocapture readonly %A, ptr nocapture readonly %B, ptr nocapture %C) {
entry:
  br label %entry.split

entry.split:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv50 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next51, %for.cond.cleanup3 ]
  br label %for.cond5.preheader

for.cond.cleanup:
  ret i32 0

for.cond5.preheader:
  %indvars.iv47 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next48, %for.cond.cleanup7 ]
  %arrayidx10 = getelementptr inbounds [1536 x float], ptr %C, i64 %indvars.iv50, i64 %indvars.iv47
  br label %for.body8

for.cond.cleanup3:
  %indvars.iv.next51 = add nuw nsw i64 %indvars.iv50, 1
  %exitcond52 = icmp eq i64 %indvars.iv.next51, 1536
  br i1 %exitcond52, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup7:
  %indvars.iv.next48 = add nuw nsw i64 %indvars.iv47, 1
  %exitcond49 = icmp eq i64 %indvars.iv.next48, 1536
  br i1 %exitcond49, label %for.cond.cleanup3, label %for.cond5.preheader

for.body8:
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %0 = load float, ptr %arrayidx10, align 4
  %arrayidx14 = getelementptr inbounds [1536 x float], ptr %A, i64 %indvars.iv50, i64 %indvars.iv
  %1 = load float, ptr %arrayidx14, align 4
  %arrayidx18 = getelementptr inbounds [1536 x float], ptr %B, i64 %indvars.iv, i64 %indvars.iv47
  %2 = load float, ptr %arrayidx18, align 4
  %mul = fmul float %1, %2
  %add = fadd float %0, %mul
  store float %add, ptr %arrayidx10, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1536
  br i1 %exitcond, label %for.cond.cleanup7, label %for.body8
}
