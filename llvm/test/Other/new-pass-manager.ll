; This test is essentially doing very basic things with the opt tool and the
; new pass manager pipeline. It will be used to flesh out the feature
; completeness of the opt tool when the new pass manager is engaged. The tests
; may not be useful once it becomes the default or may get spread out into other
; files, but for now this is just going to step the new process through its
; paces.

; RUN: opt -disable-output -disable-verify -debug-pass-manager \
; RUN:     -passes=no-op-module %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MODULE-PASS
; CHECK-MODULE-PASS: Starting pass manager
; CHECK-MODULE-PASS-NEXT: Running pass: NoOpModulePass
; CHECK-MODULE-PASS-NEXT: Finished pass manager

; RUN: opt -disable-output -disable-verify -debug-pass-manager \
; RUN:     -passes=no-op-cgscc %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-CGSCC-PASS
; RUN: opt -disable-output -disable-verify -debug-pass-manager \
; RUN:     -passes='cgscc(no-op-cgscc)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-CGSCC-PASS
; CHECK-CGSCC-PASS: Starting pass manager
; CHECK-CGSCC-PASS-NEXT: Running pass: ModuleToPostOrderCGSCCPassAdaptor
; CHECK-CGSCC-PASS-NEXT: Running analysis: CGSCCAnalysisManagerModuleProxy
; CHECK-CGSCC-PASS-NEXT: Running analysis: Lazy CallGraph Analysis
; CHECK-CGSCC-PASS-NEXT: Starting pass manager
; CHECK-CGSCC-PASS-NEXT: Running pass: NoOpCGSCCPass
; CHECK-CGSCC-PASS-NEXT: Finished pass manager
; CHECK-CGSCC-PASS-NEXT: Finished pass manager

; RUN: opt -disable-output -disable-verify -debug-pass-manager \
; RUN:     -passes=no-op-function %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-FUNCTION-PASS
; RUN: opt -disable-output -disable-verify -debug-pass-manager \
; RUN:     -passes='function(no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-FUNCTION-PASS
; CHECK-FUNCTION-PASS: Starting pass manager
; CHECK-FUNCTION-PASS-NEXT: Running pass: ModuleToFunctionPassAdaptor
; CHECK-FUNCTION-PASS-NEXT: Running analysis: FunctionAnalysisManagerModuleProxy
; CHECK-FUNCTION-PASS-NEXT: Starting pass manager
; CHECK-FUNCTION-PASS-NEXT: Running pass: NoOpFunctionPass
; CHECK-FUNCTION-PASS-NEXT: Finished pass manager
; CHECK-FUNCTION-PASS-NEXT: Finished pass manager

; RUN: opt -disable-output -debug-pass-manager -passes=print %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MODULE-PRINT
; CHECK-MODULE-PRINT: Starting pass manager
; CHECK-MODULE-PRINT: Running pass: VerifierPass
; CHECK-MODULE-PRINT: Running pass: PrintModulePass
; CHECK-MODULE-PRINT: ModuleID
; CHECK-MODULE-PRINT: define void @foo()
; CHECK-MODULE-PRINT: Running pass: VerifierPass
; CHECK-MODULE-PRINT: Finished pass manager

; RUN: opt -disable-output -debug-pass-manager -disable-verify -passes='print,verify' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MODULE-VERIFY
; CHECK-MODULE-VERIFY: Starting pass manager
; CHECK-MODULE-VERIFY: Running pass: PrintModulePass
; CHECK-MODULE-VERIFY: ModuleID
; CHECK-MODULE-VERIFY: define void @foo()
; CHECK-MODULE-VERIFY: Running pass: VerifierPass
; CHECK-MODULE-VERIFY: Finished pass manager

; RUN: opt -disable-output -debug-pass-manager -passes='function(print)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-FUNCTION-PRINT
; CHECK-FUNCTION-PRINT: Starting pass manager
; CHECK-FUNCTION-PRINT: Running pass: VerifierPass
; CHECK-FUNCTION-PRINT: Running pass: ModuleToFunctionPassAdaptor
; CHECK-FUNCTION-PRINT: Running analysis: FunctionAnalysisManagerModuleProxy
; CHECK-FUNCTION-PRINT: Starting pass manager
; CHECK-FUNCTION-PRINT: Running pass: PrintFunctionPass
; CHECK-FUNCTION-PRINT-NOT: ModuleID
; CHECK-FUNCTION-PRINT: define void @foo()
; CHECK-FUNCTION-PRINT: Finished pass manager
; CHECK-FUNCTION-PRINT: Running pass: VerifierPass
; CHECK-FUNCTION-PRINT: Finished pass manager

; RUN: opt -disable-output -debug-pass-manager -disable-verify -passes='function(print,verify)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-FUNCTION-VERIFY
; CHECK-FUNCTION-VERIFY: Starting pass manager
; CHECK-FUNCTION-VERIFY: Starting pass manager
; CHECK-FUNCTION-VERIFY: Running pass: PrintFunctionPass
; CHECK-FUNCTION-VERIFY-NOT: ModuleID
; CHECK-FUNCTION-VERIFY: define void @foo()
; CHECK-FUNCTION-VERIFY: Running pass: VerifierPass
; CHECK-FUNCTION-VERIFY: Finished pass manager
; CHECK-FUNCTION-VERIFY: Finished pass manager

; RUN: opt -S -o - -passes='no-op-module,no-op-module' %s \
; RUN:     | FileCheck %s --check-prefix=CHECK-NOOP
; CHECK-NOOP: define void @foo() {
; CHECK-NOOP:   ret void
; CHECK-NOOP: }

; Round trip through bitcode.
; RUN: opt -f -o - -passes='no-op-module,no-op-module' %s \
; RUN:     | llvm-dis \
; RUN:     | FileCheck %s --check-prefix=CHECK-NOOP

; RUN: opt -disable-output -debug-pass-manager -verify-each -passes='no-op-module,function(no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-VERIFY-EACH
; CHECK-VERIFY-EACH: Starting pass manager
; CHECK-VERIFY-EACH: Running pass: VerifierPass
; CHECK-VERIFY-EACH: Running pass: NoOpModulePass
; CHECK-VERIFY-EACH: Running pass: VerifierPass
; CHECK-VERIFY-EACH: Starting pass manager
; CHECK-VERIFY-EACH: Running pass: NoOpFunctionPass
; CHECK-VERIFY-EACH: Running pass: VerifierPass
; CHECK-VERIFY-EACH: Finished pass manager
; CHECK-VERIFY-EACH: Running pass: VerifierPass
; CHECK-VERIFY-EACH: Finished pass manager

; RUN: opt -disable-output -debug-pass-manager -disable-verify -passes='no-op-module,function(no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NO-VERIFY
; CHECK-NO-VERIFY: Starting pass manager
; CHECK-NO-VERIFY-NOT: VerifierPass
; CHECK-NO-VERIFY: Running pass: NoOpModulePass
; CHECK-NO-VERIFY-NOT: VerifierPass
; CHECK-NO-VERIFY: Starting pass manager
; CHECK-NO-VERIFY: Running pass: NoOpFunctionPass
; CHECK-NO-VERIFY-NOT: VerifierPass
; CHECK-NO-VERIFY: Finished pass manager
; CHECK-NO-VERIFY-NOT: VerifierPass
; CHECK-NO-VERIFY: Finished pass manager

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='require<no-op-module>,cgscc(require<no-op-cgscc>,function(require<no-op-function>))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-ANALYSES
; CHECK-ANALYSES: Starting pass manager
; CHECK-ANALYSES: Running pass: RequireAnalysisPass
; CHECK-ANALYSES: Running analysis: NoOpModuleAnalysis
; CHECK-ANALYSES: Starting pass manager
; CHECK-ANALYSES: Running pass: RequireAnalysisPass
; CHECK-ANALYSES: Running analysis: NoOpCGSCCAnalysis
; CHECK-ANALYSES: Starting pass manager
; CHECK-ANALYSES: Running pass: RequireAnalysisPass
; CHECK-ANALYSES: Running analysis: NoOpFunctionAnalysis

; Make sure no-op passes that preserve all analyses don't even try to do any
; analysis invalidation.
; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='require<no-op-module>,cgscc(require<no-op-cgscc>,function(require<no-op-function>))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NO-OP-INVALIDATION
; CHECK-NO-OP-INVALIDATION: Starting pass manager
; CHECK-NO-OP-INVALIDATION-NOT: Invalidating all non-preserved analyses

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='require<no-op-module>,require<no-op-module>,require<no-op-module>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DO-CACHE-MODULE-ANALYSIS-RESULTS
; CHECK-DO-CACHE-MODULE-ANALYSIS-RESULTS: Starting pass manager
; CHECK-DO-CACHE-MODULE-ANALYSIS-RESULTS: Running pass: RequireAnalysisPass
; CHECK-DO-CACHE-MODULE-ANALYSIS-RESULTS: Running analysis: NoOpModuleAnalysis
; CHECK-DO-CACHE-MODULE-ANALYSIS-RESULTS-NOT: Running analysis: NoOpModuleAnalysis

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='require<no-op-module>,invalidate<no-op-module>,require<no-op-module>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DO-INVALIDATE-MODULE-ANALYSIS-RESULTS
; CHECK-DO-INVALIDATE-MODULE-ANALYSIS-RESULTS: Starting pass manager
; CHECK-DO-INVALIDATE-MODULE-ANALYSIS-RESULTS: Running pass: RequireAnalysisPass
; CHECK-DO-INVALIDATE-MODULE-ANALYSIS-RESULTS: Running analysis: NoOpModuleAnalysis
; CHECK-DO-INVALIDATE-MODULE-ANALYSIS-RESULTS: Invalidating analysis: NoOpModuleAnalysis
; CHECK-DO-INVALIDATE-MODULE-ANALYSIS-RESULTS: Running analysis: NoOpModuleAnalysis

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='cgscc(require<no-op-cgscc>,require<no-op-cgscc>,require<no-op-cgscc>)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DO-CACHE-CGSCC-ANALYSIS-RESULTS
; CHECK-DO-CACHE-CGSCC-ANALYSIS-RESULTS: Starting pass manager
; CHECK-DO-CACHE-CGSCC-ANALYSIS-RESULTS: Running pass: RequireAnalysisPass
; CHECK-DO-CACHE-CGSCC-ANALYSIS-RESULTS: Running analysis: NoOpCGSCCAnalysis
; CHECK-DO-CACHE-CGSCC-ANALYSIS-RESULTS-NOT: Running analysis: NoOpCGSCCAnalysis

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='cgscc(require<no-op-cgscc>,invalidate<no-op-cgscc>,require<no-op-cgscc>)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DO-INVALIDATE-CGSCC-ANALYSIS-RESULTS
; CHECK-DO-INVALIDATE-CGSCC-ANALYSIS-RESULTS: Starting pass manager
; CHECK-DO-INVALIDATE-CGSCC-ANALYSIS-RESULTS: Running pass: RequireAnalysisPass
; CHECK-DO-INVALIDATE-CGSCC-ANALYSIS-RESULTS: Running analysis: NoOpCGSCCAnalysis
; CHECK-DO-INVALIDATE-CGSCC-ANALYSIS-RESULTS: Invalidating analysis: NoOpCGSCCAnalysis
; CHECK-DO-INVALIDATE-CGSCC-ANALYSIS-RESULTS: Running analysis: NoOpCGSCCAnalysis

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(require<no-op-function>,require<no-op-function>,require<no-op-function>)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DO-CACHE-FUNCTION-ANALYSIS-RESULTS
; CHECK-DO-CACHE-FUNCTION-ANALYSIS-RESULTS: Starting pass manager
; CHECK-DO-CACHE-FUNCTION-ANALYSIS-RESULTS: Running pass: RequireAnalysisPass
; CHECK-DO-CACHE-FUNCTION-ANALYSIS-RESULTS: Running analysis: NoOpFunctionAnalysis
; CHECK-DO-CACHE-FUNCTION-ANALYSIS-RESULTS-NOT: Running analysis: NoOpFunctionAnalysis

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(require<no-op-function>,invalidate<no-op-function>,require<no-op-function>)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-DO-INVALIDATE-FUNCTION-ANALYSIS-RESULTS
; CHECK-DO-INVALIDATE-FUNCTION-ANALYSIS-RESULTS: Starting pass manager
; CHECK-DO-INVALIDATE-FUNCTION-ANALYSIS-RESULTS: Running pass: RequireAnalysisPass
; CHECK-DO-INVALIDATE-FUNCTION-ANALYSIS-RESULTS: Running analysis: NoOpFunctionAnalysis
; CHECK-DO-INVALIDATE-FUNCTION-ANALYSIS-RESULTS: Invalidating analysis: NoOpFunctionAnalysis
; CHECK-DO-INVALIDATE-FUNCTION-ANALYSIS-RESULTS: Running analysis: NoOpFunctionAnalysis

; RUN: opt -disable-output -disable-verify -debug-pass-manager \
; RUN:     -passes='require<no-op-module>,module(require<no-op-module>,function(require<no-op-function>,invalidate<all>,require<no-op-function>),require<no-op-module>),require<no-op-module>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-INVALIDATE-ALL
; CHECK-INVALIDATE-ALL: Starting pass manager
; CHECK-INVALIDATE-ALL: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL: Running analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL: Starting pass manager
; CHECK-INVALIDATE-ALL: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-NOT: Running analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL: Starting pass manager
; CHECK-INVALIDATE-ALL: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL: Running analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL: Running pass: InvalidateAllAnalysesPass
; CHECK-INVALIDATE-ALL: Invalidating all non-preserved analyses
; CHECK-INVALIDATE-ALL: Invalidating analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL: Running analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL: Finished pass manager
; CHECK-INVALIDATE-ALL: Invalidating all non-preserved analyses
; CHECK-INVALIDATE-ALL-NOT: Running analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL: Invalidating all non-preserved analyses
; CHECK-INVALIDATE-ALL: Invalidating analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL: Running analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL: Finished pass manager
; CHECK-INVALIDATE-ALL: Invalidating all non-preserved analyses
; CHECK-INVALIDATE-ALL-NOT: Invalidating analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-NOT: Running analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL: Finished pass manager

; RUN: opt -disable-output -disable-verify -debug-pass-manager \
; RUN:     -passes='require<no-op-module>,module(require<no-op-module>,cgscc(require<no-op-cgscc>,function(require<no-op-function>,invalidate<all>,require<no-op-function>),require<no-op-cgscc>),require<no-op-module>),require<no-op-module>' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-INVALIDATE-ALL-CG
; CHECK-INVALIDATE-ALL-CG: Starting pass manager
; CHECK-INVALIDATE-ALL-CG: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG: Running analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL-CG: Starting pass manager
; CHECK-INVALIDATE-ALL-CG: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG-NOT: Running analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL-CG: Starting pass manager
; CHECK-INVALIDATE-ALL-CG: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG: Running analysis: NoOpCGSCCAnalysis
; CHECK-INVALIDATE-ALL-CG: Starting pass manager
; CHECK-INVALIDATE-ALL-CG: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG: Running analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL-CG: Running pass: InvalidateAllAnalysesPass
; CHECK-INVALIDATE-ALL-CG: Invalidating all non-preserved analyses
; CHECK-INVALIDATE-ALL-CG: Invalidating analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL-CG: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG: Running analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL-CG: Finished pass manager
; CHECK-INVALIDATE-ALL-CG: Invalidating all non-preserved analyses
; CHECK-INVALIDATE-ALL-CG-NOT: Running analysis: NoOpFunctionAnalysis
; CHECK-INVALIDATE-ALL-CG: Invalidating all non-preserved analyses
; CHECK-INVALIDATE-ALL-CG: Invalidating analysis: NoOpCGSCCAnalysis
; CHECK-INVALIDATE-ALL-CG: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG: Running analysis: NoOpCGSCCAnalysis
; CHECK-INVALIDATE-ALL-CG: Finished pass manager
; CHECK-INVALIDATE-ALL-CG: Invalidating all non-preserved analyses
; CHECK-INVALIDATE-ALL-CG-NOT: Invalidating analysis: NoOpCGSCCAnalysis
; CHECK-INVALIDATE-ALL-CG: Invalidating all non-preserved analyses
; CHECK-INVALIDATE-ALL-CG: Invalidating analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL-CG: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG: Running analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL-CG: Finished pass manager
; CHECK-INVALIDATE-ALL-CG: Invalidating all non-preserved analyses
; CHECK-INVALIDATE-ALL-CG-NOT: Invalidating analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL-CG: Running pass: RequireAnalysisPass
; CHECK-INVALIDATE-ALL-CG-NOT: Running analysis: NoOpModuleAnalysis
; CHECK-INVALIDATE-ALL-CG: Finished pass manager

; RUN: opt -disable-output -disable-verify -debug-pass-manager %s 2>&1 \
; RUN:     -passes='require<targetlibinfo>,invalidate<all>,require<targetlibinfo>' \
; RUN:     | FileCheck %s --check-prefix=CHECK-TLI
; CHECK-TLI: Starting pass manager
; CHECK-TLI: Running pass: RequireAnalysisPass
; CHECK-TLI: Running analysis: TargetLibraryAnalysis
; CHECK-TLI: Running pass: InvalidateAllAnalysesPass
; CHECK-TLI-NOT: Invalidating analysis: TargetLibraryAnalysis
; CHECK-TLI: Running pass: RequireAnalysisPass
; CHECK-TLI-NOT: Running analysis: TargetLibraryAnalysis
; CHECK-TLI: Finished pass manager

; RUN: opt -disable-output -disable-verify -debug-pass-manager %s 2>&1 \
; RUN:     -passes='require<targetir>,invalidate<all>,require<targetir>' \
; RUN:     | FileCheck %s --check-prefix=CHECK-TIRA
; CHECK-TIRA: Starting pass manager
; CHECK-TIRA: Running pass: RequireAnalysisPass
; CHECK-TIRA: Running analysis: TargetIRAnalysis
; CHECK-TIRA: Running pass: InvalidateAllAnalysesPass
; CHECK-TIRA-NOT: Invalidating analysis: TargetIRAnalysis
; CHECK-TIRA: Running pass: RequireAnalysisPass
; CHECK-TIRA-NOT: Running analysis: TargetIRAnalysis
; CHECK-TIRA: Finished pass manager

; RUN: opt -disable-output -disable-verify -debug-pass-manager %s 2>&1 \
; RUN:     -passes='require<domtree>' \
; RUN:     | FileCheck %s --check-prefix=CHECK-DT
; CHECK-DT: Starting pass manager
; CHECK-DT: Running pass: RequireAnalysisPass
; CHECK-DT: Running analysis: DominatorTreeAnalysis
; CHECK-DT: Finished pass manager

; RUN: opt -disable-output -disable-verify -debug-pass-manager %s 2>&1 \
; RUN:     -passes='require<aa>' \
; RUN:     | FileCheck %s --check-prefix=CHECK-AA
; CHECK-AA: Starting pass manager
; CHECK-AA: Running pass: RequireAnalysisPass
; CHECK-AA: Running analysis: AAManager
; CHECK-AA: Finished pass manager

; RUN: opt -disable-output -disable-verify -debug-pass-manager %s 2>&1 \
; RUN:     -passes='require<basic-aa>' \
; RUN:     | FileCheck %s --check-prefix=CHECK-BASIC-AA
; CHECK-BASIC-AA: Starting pass manager
; CHECK-BASIC-AA: Running pass: RequireAnalysisPass
; CHECK-BASIC-AA: Running analysis: BasicAA
; CHECK-BASIC-AA: Finished pass manager

define void @foo() {
  ret void
}

declare void @bar()
