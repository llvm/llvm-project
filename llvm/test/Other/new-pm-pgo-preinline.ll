; RUN: opt -disable-verify -eagerly-invalidate-analyses=0 -debug-pass-manager -pgo-kind=pgo-instr-gen-pipeline -passes='default<O2>' -S %s 2>&1 | FileCheck %s --check-prefixes=CHECK-O2

; CHECK-O2: Running pass: ModuleInlinerWrapperPass
; CHECK-O2-NEXT: Running analysis: InlineAdvisorAnalysis
; CHECK-O2-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-O2-NEXT: Running analysis: LazyCallGraphAnalysis
; CHECK-O2-NEXT: Running analysis: FunctionAnalysisManagerCGSCCProxy on (foo)
; CHECK-O2-NEXT: Running analysis: OuterAnalysisManagerProxy
; CHECK-O2-NEXT: Running pass: InlinerPass on (foo)
; CHECK-O2-NEXT: Running pass: InlinerPass on (foo)
; CHECK-O2-NEXT: Running pass: SROAPass on foo
; CHECK-O2-NEXT: Running pass: EarlyCSEPass on foo
; CHECK-O2-NEXT: Running pass: SimplifyCFGPass on foo
; CHECK-O2-NEXT: Running pass: InstCombinePass on foo
; CHECK-O2-NEXT: Invalidating analysis: InlineAdvisorAnalysis
; CHECK-O2-NEXT: Running pass: GlobalDCEPass
; CHECK-O2-NEXT: Running pass: PGOInstrumentationGen

define void @foo() {
  ret void
}
