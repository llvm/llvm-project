; Basic test to check that DominatorTreeAnalysis is preserved by IPSCCP and
; the following analysis can re-use it. The test contains two trivial functions
; IPSCCP can simplify, so we can test the case where IPSCCP makes changes.

; RUN: opt -disable-verify -debug-pass-manager \
; RUN:     -passes='function(require<domtree>,require<postdomtree>),ipsccp,function(require<domtree>,require<postdomtree>)' -S  %s 2>&1 \
; RUN:     | FileCheck -check-prefixes='IR,NEW-PM' %s

; RUN: opt -passes='function(require<postdomtree>),ipsccp,function(verify<domtree>)' -S  %s | FileCheck -check-prefixes='IR' %s

; NEW-PM: Starting llvm::Module pass manager run.
; NEW-PM: Running analysis: DominatorTreeAnalysis on f1
; NEW-PM: Running analysis: PostDominatorTreeAnalysis on f1
; NEW-PM: Running analysis: DominatorTreeAnalysis on f2
; NEW-PM: Running analysis: PostDominatorTreeAnalysis on f2
; NEW-PM: Running pass: IPSCCPPass
; NEW-PM-DAG: Running analysis: AssumptionAnalysis on f1
; NEW-PM-DAG: Running analysis: AssumptionAnalysis on f2
; NEW-PM-NEXT: Invalidating all non-preserved analyses for:
; NEW-PM-NEXT: Invalidating all non-preserved analyses for: f1
; NEW-PM-NEXT: Invalidating all non-preserved analyses for: f2
; NEW-PM-NEXT: Running pass: ModuleToFunctionPassAdaptor
; NEW-PM-NOT: Running analysis:

; IR-LABEL: @f1
; IR-LABEL: entry:
; IR-NEXT: br label %bb2
; IR-LABEL: bb2:
; IR-NEXT: undef

; IR-LABEL: @f2
; IR-NOT: icmp
; IR:    br label %bbtrue
; IR-LABEL: bbtrue:
; IR-NEXT:   ret i32 0
define internal i32 @f1() readnone {
entry:
  br i1 false, label %bb1, label %bb2
bb1:
  ret i32 10
bb2:
  ret i32 10
}

define i32 @f2(i32 %n) {
  %i = call i32 @f1()
  %cmp = icmp eq i32 %i, 10
  br i1 %cmp, label %bbtrue, label %bbfalse

bbtrue:
  ret i32 0

bbfalse:
  %res = add i32 %n, %i
  ret i32 %res
}
