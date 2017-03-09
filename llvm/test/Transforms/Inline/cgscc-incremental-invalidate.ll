; Test for a subtle bug when computing analyses during inlining and mutating
; the SCC structure. Without care, this can fail to invalidate analyses.
;
; RUN: opt < %s -passes='cgscc(inline,function(verify<domtree>))' -debug-pass-manager -S 2>&1 | FileCheck %s

; First we check that the passes run in the way we expect. Otherwise this test
; may stop testing anything.
;
; CHECK-LABEL: Starting llvm::Module pass manager run.
; CHECK: Running pass: InlinerPass on (test1_f, test1_g, test1_h)
; CHECK: Running analysis: FunctionAnalysisManagerCGSCCProxy on (test1_f, test1_g, test1_h)
; CHECK: Running analysis: DominatorTreeAnalysis on test1_f
; CHECK: Running analysis: DominatorTreeAnalysis on test1_g
; CHECK: Invalidating all non-preserved analyses for: (test1_f, test1_g, test1_h)
; CHECK: Invalidating all non-preserved analyses for: test1_f
; CHECK: Invalidating analysis: DominatorTreeAnalysis on test1_f
; CHECK: Invalidating all non-preserved analyses for: test1_g
; CHECK: Invalidating analysis: DominatorTreeAnalysis on test1_g
; CHECK: Invalidating all non-preserved analyses for: test1_h
; CHECK-NOT: Invalidating anaylsis:
; CHECK: Running analysis: DominatorTreeAnalysis on test1_h
; CHECK: Invalidating all non-preserved analyses for: (test1_g, test1_h)
; CHECK: Invalidating all non-preserved analyses for: test1_h
; CHECK: Invalidating analysis: DominatorTreeAnalysis on test1_h

; An external function used to control branches.
declare i1 @flag()
; CHECK-LABEL: declare i1 @flag()

; The utility function with interesting control flow that gets inlined below to
; perturb the dominator tree.
define internal void @callee() {
entry:
  %ptr = alloca i8
  %flag = call i1 @flag()
  br i1 %flag, label %then, label %else

then:
  store volatile i8 42, i8* %ptr
  br label %return

else:
  store volatile i8 -42, i8* %ptr
  br label %return

return:
  ret void
}

; The 'test1_' prefixed functions work to carefully test that incrementally
; reducing an SCC in the inliner cannot accidentially leave stale function
; analysis results due to failing to invalidate them for all the functions.

; The inliner visits this last function. It can't actually break any cycles
; here, but because we visit this function we compute fresh analyses for it.
; These analyses are then invalidated when we inline callee disrupting the
; CFG, and it is important that they be freed.
define void @test1_h() {
; CHECK-LABEL: define void @test1_h()
entry:
  call void @test1_g()
; CHECK: call void @test1_g()

  ; Pull interesting CFG into this function.
  call void @callee()
; CHECK-NOT: call void @callee()

  ret void
; CHECK: ret void
}

; We visit this function second and here we inline the edge to 'test1_f'
; separating it into its own SCC. The current SCC is now just 'test1_g' and
; 'test1_h'.
define void @test1_g() {
; CHECK-LABEL: define void @test1_g()
entry:
  ; This edge gets inlined away.
  call void @test1_f()
; CHECK-NOT: call void @test1_f()
; CHECK: call void @test1_g()

  ; We force this edge to survive inlining.
  call void @test1_h() noinline
; CHECK: call void @test1_h()

  ; Pull interesting CFG into this function.
  call void @callee()
; CHECK-NOT: call void @callee()

  ret void
; CHECK: ret void
}

; We visit this function first in the inliner, and while we inline callee
; perturbing the CFG, we don't inline anything else and the SCC structure
; remains in tact.
define void @test1_f() {
; CHECK-LABEL: define void @test1_f()
entry:
  ; We force this edge to survive inlining.
  call void @test1_g() noinline
; CHECK: call void @test1_g()

  ; Pull interesting CFG into this function.
  call void @callee()
; CHECK-NOT: call void @callee()

  ret void
; CHECK: ret void
}
