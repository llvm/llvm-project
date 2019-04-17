; This is really testing that instcombine preserves analyses correctly, so we
; don't care much about the code other than it is something instcombine can
; transform.
;
; RUN: opt < %s -disable-output -debug-pass-manager 2>&1 -aa-pipeline=basic-aa,globals-aa \
; RUN:    -passes='require<globals-aa>,function(require<aa>,instcombine),function(require<aa>)' \
; RUN:    | FileCheck %s --check-prefix=AA
; AA: Running analysis: GlobalsAA
; AA: Running analysis: AAManager
; AA: Running analysis: BasicAA
; AA: Running pass: InstCombinePass on test
; AA-NOT: Invalidating analysis: GlobalsAA
; AA-NOT: Invalidating analysis: AAmanager
; AA-NOT: Invalidating analysis: BasicAA
; AA: Running pass: RequireAnalysisPass<{{.*}}AAManager
; AA-NOT: Running analysis: GlobalsAA
; AA-NOT: Running analysis: AAmanager
; AA-NOT: Running analysis: BasicAA
;
; RUN: opt < %s -disable-output -debug-pass-manager 2>&1 \
; RUN:    -passes='require<domtree>,instcombine,require<domtree>' \
; RUN:    | FileCheck %s --check-prefix=DT
; DT: Running analysis: DominatorTreeAnalysis
; DT: Running pass: InstCombinePass on test
; DT-NOT: Invalidating analysis: DominatorTreeAnalysis
; DT: Running pass: RequireAnalysisPass<{{.*}}DominatorTreeAnalysis
; DT-NOT: Running analysis: DominatorTreeAnalysis

define i32 @test(i32 %A) {
  %B = add i32 %A, 5
  %C = add i32 %B, -5
  ret i32 %C
}
