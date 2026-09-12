; RUN: opt -disable-output -debug-pass-manager \
; RUN:   -passes='function(require<escape-analysis>,invalidate<escape-analysis>,require<escape-analysis>)' \
; RUN:   %s 2>&1 | FileCheck %s

; CHECK: Running pass: RequireAnalysisPass<llvm::EscapeAnalysis
; CHECK: Running analysis: EscapeAnalysis on f
; CHECK: Running pass: InvalidateAnalysisPass<llvm::EscapeAnalysis> on f
; CHECK: Invalidating analysis: EscapeAnalysis on f
; CHECK: Running pass: RequireAnalysisPass<llvm::EscapeAnalysis
; CHECK: Running analysis: EscapeAnalysis on f

; Verify that the re-run analysis also produces correct results by using
; print<escape-analysis> which exercises isEscaping() on the result.
; RUN: opt -passes='print<escape-analysis>' -disable-output %s 2>&1 | FileCheck %s --check-prefix=RESULT

; RESULT-LABEL: Printing analysis 'Escape Analysis' for function 'f':
; RESULT:         a escapes: no

define void @f() {
  %a = alloca i8, align 1
  ret void
}
