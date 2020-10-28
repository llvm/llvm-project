; Test for a subtle bug when computing analyses during inlining and mutating
; the SCC structure. Without care, this can fail to invalidate analyses.
;
; RUN: opt < %s -passes='cgscc(inline,function(verify<domtree>))' -debug-pass-manager -S 2>&1 | FileCheck %s

; First we check that the passes run in the way we expect. Otherwise this test
; may stop testing anything.
;
; CHECK-LABEL: Starting llvm::Module pass manager run.
; CHECK: Running pass: InlinerPass on (test1_f, test1_g, test1_h)
; CHECK: Running analysis: DominatorTreeAnalysis on test1_f
; CHECK: Running analysis: DominatorTreeAnalysis on test1_g
; CHECK: Invalidating analysis: DominatorTreeAnalysis on test1_f
; CHECK: Invalidating analysis: LoopAnalysis on test1_f
; CHECK: Invalidating analysis: BranchProbabilityAnalysis on test1_f
; CHECK: Invalidating analysis: BlockFrequencyAnalysis on test1_f
; CHECK: Invalidating analysis: DominatorTreeAnalysis on test1_g
; CHECK: Invalidating analysis: LoopAnalysis on test1_g
; CHECK: Invalidating analysis: BranchProbabilityAnalysis on test1_g
; CHECK: Invalidating analysis: BlockFrequencyAnalysis on test1_g
; CHECK: Invalidating analysis: DominatorTreeAnalysis on test1_h
; CHECK: Invalidating analysis: LoopAnalysis on test1_h
; CHECK: Invalidating analysis: BranchProbabilityAnalysis on test1_h
; CHECK: Invalidating analysis: BlockFrequencyAnalysis on test1_h
; CHECK-NOT: Invalidating analysis:
; CHECK: Starting llvm::Function pass manager run.
; CHECK-NEXT: Running pass: DominatorTreeVerifierPass on test1_g
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on test1_g
; CHECK-NEXT: Finished llvm::Function pass manager run.
; CHECK-NOT: Invalidating analysis:
; CHECK: Starting llvm::Function pass manager run.
; CHECK-NEXT: Running pass: DominatorTreeVerifierPass on test1_h
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on test1_h
; CHECK-NEXT: Finished llvm::Function pass manager run.
; CHECK-NOT: Invalidating analysis:
; CHECK: Running pass: DominatorTreeVerifierPass on test1_f
; CHECK-NEXT: Running analysis: DominatorTreeAnalysis on test1_f

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

; The 'test2_' prefixed code works to carefully trigger forming an SCC with
; a dominator tree for one of the functions but not the other and without even
; a function analysis manager proxy for the SCC that things get merged into.
; Without proper handling when updating the call graph this will find a stale
; dominator tree.

@test2_global = external global i32, align 4

define void @test2_hoge(i1 (i32*)* %arg) {
; CHECK-LABEL: define void @test2_hoge(
bb:
  %tmp2 = call zeroext i1 %arg(i32* @test2_global)
; CHECK: call zeroext i1 %arg(
  br label %bb3

bb3:
  %tmp5 = call zeroext i1 %arg(i32* @test2_global)
; CHECK: call zeroext i1 %arg(
  br i1 %tmp5, label %bb3, label %bb6

bb6:
  ret void
}

define zeroext i1 @test2_widget(i32* %arg) {
; CHECK-LABEL: define zeroext i1 @test2_widget(
bb:
  %tmp1 = alloca i8, align 1
  %tmp2 = alloca i32, align 4
  call void @test2_quux()
; CHECK-NOT:     call
;
; CHECK:         call zeroext i1 @test2_widget(i32* @test2_global)
; CHECK-NEXT:    br label %[[NEW_BB:.*]]
;
; CHECK:       [[NEW_BB]]:
; CHECK-NEXT:    call zeroext i1 @test2_widget(i32* @test2_global)
;
; CHECK:       {{.*}}:

  call void @test2_hoge.1(i32* %arg)
; CHECK-NEXT:    call void @test2_hoge.1(

  %tmp4 = call zeroext i1 @test2_barney(i32* %tmp2)
  %tmp5 = zext i1 %tmp4 to i32
  store i32 %tmp5, i32* %tmp2, align 4
  %tmp6 = call zeroext i1 @test2_barney(i32* null)
  call void @test2_ham(i8* %tmp1)
; CHECK:         call void @test2_ham(

  call void @test2_quux()
; CHECK-NOT:     call
;
; CHECK:         call zeroext i1 @test2_widget(i32* @test2_global)
; CHECK-NEXT:    br label %[[NEW_BB:.*]]
;
; CHECK:       [[NEW_BB]]:
; CHECK-NEXT:    call zeroext i1 @test2_widget(i32* @test2_global)
;
; CHECK:       {{.*}}:
  ret i1 true
; CHECK-NEXT:    ret i1 true
}

define internal void @test2_quux() {
; CHECK-NOT: @test2_quux
bb:
  call void @test2_hoge(i1 (i32*)* @test2_widget)
  ret void
}

declare void @test2_hoge.1(i32*)

declare zeroext i1 @test2_barney(i32*)

declare void @test2_ham(i8*)
