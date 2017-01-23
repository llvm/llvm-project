; Test that the inliner clears analyses which may hold references to function
; bodies when it decides to delete them after inlining the last caller.
; We check this by using correlated-propagation to populate LVI with basic
; block references that would dangle if we failed to clear the inlined function
; body.
;
; RUN: opt -debug-pass-manager -S < %s 2>&1 \
; RUN:     -passes='cgscc(inline,function(correlated-propagation))' \
; RUN:     | FileCheck %s
;
; CHECK-LABEL: Starting llvm::Module pass manager run.
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
