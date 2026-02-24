; RUN: opt %loadNPMPolly -polly-parallel '-passes=polly-custom<opt-isl;ast>' -polly-print-ast -disable-output -debug-only=polly-ast < %s 2>&1 | FileCheck --check-prefix=AST %s
; RUN: opt %loadNPMPolly -polly-parallel '-passes=polly<no-default-opts;opt-isl>' -S < %s | FileCheck --check-prefix=CODEGEN %s
; REQUIRES: asserts

; Parallelization of detected matrix-multiplication.

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

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


; AST:      // 1st level tiling - Tiles
; AST-NEXT: #pragma minimal dependence distance: 1
; AST-NEXT: for (int c0 = 0; c0 <= 1; c0 += 1)
; AST-NEXT:   #pragma minimal dependence distance: 1
; AST-NEXT:   for (int c1 = 0; c1 <= 1; c1 += 1) {
; AST-NEXT:     #pragma omp parallel for
; AST-NEXT:     for (int c3 = 768 * c0; c3 <= 768 * c0 + 767; c3 += 1)
; AST-NEXT:       #pragma simd
; AST-NEXT:       for (int c4 = 1024 * c1; c4 <= min(1535, 1024 * c1 + 1023); c4 += 1)
; AST-NEXT:         CopyStmt_0(0, c3, c4);
; AST-NEXT:     #pragma minimal dependence distance: 1
; AST-NEXT:     for (int c2 = 0; c2 <= 31; c2 += 1) {
; AST-NEXT:       #pragma omp parallel for
; AST-NEXT:       for (int c6 = 48 * c2; c6 <= 48 * c2 + 47; c6 += 1)
; AST-NEXT:         #pragma simd
; AST-NEXT:         for (int c7 = 1024 * c1; c7 <= min(1535, 1024 * c1 + 1023); c7 += 1)
; AST-NEXT:           CopyStmt_1(c0, c1, c2, c6, c7);
; AST-NEXT:       // 1st level tiling - Points
; AST-NEXT:       // Register tiling - Tiles
; AST-NEXT:       #pragma omp parallel for
; AST-NEXT:       for (int c3 = 0; c3 <= 255; c3 += 1)
; AST-NEXT:         for (int c4 = 0; c4 <= 23; c4 += 1)
; AST-NEXT:           #pragma minimal dependence distance: 1
; AST-NEXT:           for (int c5 = 0; c5 <= min(1023, -1024 * c1 + 1535); c5 += 1) {
; AST-NEXT:             // Loop Vectorizer Disabled
; AST-NEXT:             // Register tiling - Points
; AST-NEXT:             {
; AST-NEXT:               Stmt_for_body8(48 * c2 + 2 * c4, 768 * c0 + 3 * c3, 1024 * c1 + c5);
; AST-NEXT:               Stmt_for_body8(48 * c2 + 2 * c4, 768 * c0 + 3 * c3 + 1, 1024 * c1 + c5);
; AST-NEXT:               Stmt_for_body8(48 * c2 + 2 * c4, 768 * c0 + 3 * c3 + 2, 1024 * c1 + c5);
; AST-NEXT:               Stmt_for_body8(48 * c2 + 2 * c4 + 1, 768 * c0 + 3 * c3, 1024 * c1 + c5);
; AST-NEXT:               Stmt_for_body8(48 * c2 + 2 * c4 + 1, 768 * c0 + 3 * c3 + 1, 1024 * c1 + c5);
; AST-NEXT:               Stmt_for_body8(48 * c2 + 2 * c4 + 1, 768 * c0 + 3 * c3 + 2, 1024 * c1 + c5);
; AST-NEXT:             }
; AST-NEXT:           }
; AST-NEXT:     }
; AST-NEXT:   }

; CODEGEN: subfunc
