; RUN: opt -mtriple=wasm32-unknown-unknown -expand-variadics-override=lowering -disable-output -debug-pass-manager -passes='function(require<domtree>),expand-variadics,function(require<domtree>)' %s 2>&1 | FileCheck %s
; REQUIRES: webassembly-registered-target

; Rewriting @vararg changes the module even though it has no callers, so the
; analyses of @other should be invalidated.

; CHECK: Running pass: ExpandVariadicsPass on [module]{{$}}
; CHECK-NEXT: Clearing all analysis results for: {{$}}
; CHECK-NEXT: Invalidating analysis: DominatorTreeAnalysis on other{{$}}

define void @vararg(...) {
  ret void
}

define void @other() {
  ret void
}
