; RUN: opt -disable-verify -debug-pass-manager -pgo-kind=pgo-instr-gen-pipeline -passes='default<Os>' -S %s 2>&1 | FileCheck %s --check-prefixes=CHECK-Osz
; RUN: opt -disable-verify -debug-pass-manager -pgo-kind=pgo-instr-gen-pipeline -passes='default<Oz>' -S %s 2>&1 | FileCheck %s --check-prefixes=CHECK-Osz

; CHECK-Osz: Running pass: ModuleInlinerWrapperPass
; CHECK-Osz-NEXT: Running analysis: InlineAdvisorAnalysis
; CHECK-Osz-NEXT: Running analysis: InnerAnalysisManagerProxy
; CHECK-Osz-NEXT: Running analysis: LazyCallGraphAnalysis
; CHECK-Osz-NEXT: Running analysis: FunctionAnalysisManagerCGSCCProxy on (foo)
; CHECK-Osz-NEXT: Running analysis: OuterAnalysisManagerProxy
; CHECK-Osz-NEXT: Running pass: InlinerPass on (foo)
; CHECK-Osz-NEXT: Running pass: InlinerPass on (foo)
; CHECK-Osz-NEXT: Running pass: SROA on foo
; CHECK-Osz-NEXT: Running pass: EarlyCSEPass on foo
; CHECK-Osz-NEXT: Running pass: SimplifyCFGPass on foo
; CHECK-Osz-NEXT: Running pass: InstCombinePass on foo
; CHECK-Osz-NEXT: Invalidating analysis: InlineAdvisorAnalysis
; CHECK-Osz-NEXT: Running pass: GlobalDCEPass
; CHECK-Osz-NEXT: Running pass: PGOInstrumentationGen

define void @foo() {
  ret void
}
