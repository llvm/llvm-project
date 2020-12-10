; RUN: opt     -S -debug -enable-new-pm=0 %s 2>&1 | FileCheck %s --check-prefix=O0
; RUN: opt -O1 -S -debug -enable-new-pm=0 %s 2>&1 | FileCheck %s --check-prefix=O1
; RUN: opt -O2 -S -debug -enable-new-pm=0 %s 2>&1 | FileCheck %s --check-prefix=O1 --check-prefix=O2O3
; RUN: opt -O3 -S -debug -enable-new-pm=0 %s 2>&1 | FileCheck %s --check-prefix=O1 --check-prefix=O2O3
; RUN: opt -dce -gvn-hoist -loweratomic -S -debug -enable-new-pm=0 %s 2>&1 | FileCheck %s --check-prefix=MORE
; RUN: opt -indvars -licm -loop-deletion -loop-extract -loop-idiom -loop-instsimplify -loop-reduce -loop-reroll -loop-rotate -loop-unroll -loop-unswitch -enable-new-pm=0 -S -debug %s 2>&1 | FileCheck %s --check-prefix=LOOP
; RUN: opt -passes='default<O0>' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=%llvmcheckext-NPM-O0
; RUN: opt -passes='default<O1>' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=NPM-O1
; RUN: opt -passes='default<O2>' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=NPM-O1 --check-prefix=NPM-O2O3
; RUN: opt -passes='default<O3>' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=NPM-O1 --check-prefix=NPM-O2O3
; RUN: opt -passes='dce,gvn-hoist,loweratomic' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=NPM-MORE
; RUN: opt -passes='loop(indvars,licm,loop-deletion,loop-idiom,loop-instsimplify,loop-reduce,simple-loop-unswitch),loop-unroll' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=NPM-LOOP
; RUN: opt -passes='instsimplify,verify' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=NPM-REQUIRED

; REQUIRES: asserts

; This test verifies that we don't run target independent IR-level
; optimizations on optnone functions.

; Function Attrs: noinline optnone
define i32 @foo(i32 %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %0 = load i32, i32* %x.addr, align 4
  %dec = add nsw i32 %0, -1
  store i32 %dec, i32* %x.addr, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %dec
}

attributes #0 = { optnone noinline }

; Nothing that runs at -O0 gets skipped (except when the Bye extension is present).
; O0-NOT: Skipping pass
; CHECK-EXT-NPM-O0: Skipping pass {{.*}}Bye
; CHECK-NOEXT-NPM-O0-NOT: Skipping pass

; IR passes run at -O1 and higher.
; O1-DAG: Skipping pass 'Aggressive Dead Code Elimination'
; O1-DAG: Skipping pass 'Combine redundant instructions'
; O1-DAG: Skipping pass 'Early CSE'
; O1-DAG: Skipping pass 'Reassociate expressions'
; O1-DAG: Skipping pass 'Simplify the CFG'
; O1-DAG: Skipping pass 'Sparse Conditional Constant Propagation'
; NPM-O1-DAG: Skipping pass: SimplifyCFGPass on foo
; NPM-O1-DAG: Skipping pass: SROA
; NPM-O1-DAG: Skipping pass: EarlyCSEPass
; NPM-O1-DAG: Skipping pass: LowerExpectIntrinsicPass
; NPM-O1-DAG: Skipping pass: PromotePass
; NPM-O1-DAG: Skipping pass: InstCombinePass

; Additional IR passes run at -O2 and higher.
; O2O3-DAG: Skipping pass 'Global Value Numbering'
; O2O3-DAG: Skipping pass 'SLP Vectorizer'
; NPM-O2O3-DAG: Skipping pass: GVN
; NPM-O2O3-DAG: Skipping pass: SLPVectorizerPass

; Additional IR passes that opt doesn't turn on by default.
; MORE-DAG: Skipping pass 'Dead Code Elimination'
; NPM-MORE-DAG: Skipping pass: DCEPass
; NPM-MORE-DAG: Skipping pass: GVNHoistPass

; Loop IR passes that opt doesn't turn on by default.
; LOOP-DAG: Skipping pass 'Delete dead loops'
; LOOP-DAG: Skipping pass 'Induction Variable Simplification'
; LOOP-DAG: Skipping pass 'Loop Invariant Code Motion'
; LOOP-DAG: Skipping pass 'Loop Strength Reduction'
; LOOP-DAG: Skipping pass 'Recognize loop idioms'
; LOOP-DAG: Skipping pass 'Reroll loops'
; LOOP-DAG: Skipping pass 'Rotate Loops'
; LOOP-DAG: Skipping pass 'Simplify instructions in loops'
; LOOP-DAG: Skipping pass 'Unroll loops'
; LOOP-DAG: Skipping pass 'Unswitch loops'
; LoopPassManager should not be skipped over an optnone function
; NPM-LOOP-NOT: Skipping pass: PassManager
; NPM-LOOP-DAG: Skipping pass: LoopSimplifyPass on foo
; NPM-LOOP-DAG: Skipping pass: LCSSAPass
; NPM-LOOP-DAG: Skipping pass: IndVarSimplifyPass
; NPM-LOOP-DAG: Skipping pass: SimpleLoopUnswitchPass
; NPM-LOOP-DAG: Skipping pass: LoopUnrollPass
; NPM-LOOP-DAG: Skipping pass: LoopStrengthReducePass
; NPM-LOOP-DAG: Skipping pass: LoopDeletionPass
; NPM-LOOP-DAG: Skipping pass: LICMPass
; NPM-LOOP-DAG: Skipping pass: LoopIdiomRecognizePass
; NPM-LOOP-DAG: Skipping pass: LoopInstSimplifyPass

; NPM-REQUIRED-DAG: Skipping pass: InstSimplifyPass
; NPM-REQUIRED-DAG: Skipping pass InstSimplifyPass on foo due to optnone attribute
; NPM-REQUIRED-DAG: Running pass: VerifierPass
; NPM-REQUIRED-NOT: Skipping pass: VerifyPass
; NPM-REQUIRED-NOT: Skipping pass VerifyPass on foo due to optnone attribute
