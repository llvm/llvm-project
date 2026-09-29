; Test that when a pass like correlated-propagation populates an analysis such
; as LVI with references back into the IR of a function that the inliner will
; delete, this doesn't crash or go awry despite the inliner clearing the analyses
; separately from when it deletes the function.
;
; RUN: opt -debug-pass-manager -S < %s 2>&1 \
; RUN:     -passes='cgscc(inline,function(correlated-propagation))' \
; RUN:     | FileCheck %s
;
; CHECK: Running pass: InlinerPass on (callee)
; CHECK: Running pass: CorrelatedValuePropagationPass on callee
; CHECK: Running analysis: LazyValueAnalysis
; CHECK: Running pass: InlinerPass on (caller)
; CHECK: Clearing all analysis results for: callee
; CHECK: Running pass: CorrelatedValuePropagationPass on caller
; CHECK: Running analysis: LazyValueAnalysis

define internal i32 @callee(i32 %x) {
; CHECK-NOT: @callee
entry:
  ret i32 %x
}

define i32 @caller(i32 %x) {
; CHECK-LABEL: define i32 @caller
entry:
  %call = call i32 @callee(i32 %x)
; CHECK-NOT: call
  ret i32 %call
; CHECK: ret i32 %x
}
