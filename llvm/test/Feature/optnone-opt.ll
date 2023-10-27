; RUN: opt -passes='default<O0>' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=%llvmcheckext-NPM-O0
; RUN: opt -passes='default<O1>' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=NPM-O1
; RUN: opt -passes='default<O2>' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=NPM-O1 --check-prefix=NPM-O2O3
; RUN: opt -passes='default<O3>' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=NPM-O1 --check-prefix=NPM-O2O3
; RUN: opt -passes='dce,gvn-hoist,loweratomic' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=NPM-MORE
; RUN: opt -passes='loop(indvars,licm,loop-deletion,loop-idiom,loop-instsimplify,loop-reduce,simple-loop-unswitch),loop-unroll' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=NPM-LOOP
; RUN: opt -passes='instsimplify,verify' -S -debug-pass-manager %s 2>&1 | FileCheck %s --check-prefix=NPM-REQUIRED

; This test verifies that we don't run target independent IR-level
; optimizations on optnone functions.

; Function Attrs: noinline optnone
define i32 @foo(i32 %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %0 = load i32, ptr %x.addr, align 4
  %dec = add nsw i32 %0, -1
  store i32 %dec, ptr %x.addr, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %dec
}

attributes #0 = { optnone noinline }

; Nothing that runs at -O0 gets skipped (except when the Bye extension is present).
; CHECK-EXT-NPM-O0: Skipping pass {{.*}}Bye
; CHECK-NOEXT-NPM-O0-NOT: Skipping pass

; IR passes run at -O1 and higher.
; NPM-O1-DAG: Skipping pass: SimplifyCFGPass on foo
; NPM-O1-DAG: Skipping pass: SROA
; NPM-O1-DAG: Skipping pass: EarlyCSEPass
; NPM-O1-DAG: Skipping pass: LowerExpectIntrinsicPass
; NPM-O1-DAG: Skipping pass: InstCombinePass

; Additional IR passes run at -O2 and higher.
; NPM-O2O3-DAG: Skipping pass: GVN
; NPM-O2O3-DAG: Skipping pass: SLPVectorizerPass

; Additional IR passes that opt doesn't turn on by default.
; NPM-MORE-DAG: Skipping pass: DCEPass
; NPM-MORE-DAG: Skipping pass: GVNHoistPass

; Loop IR passes that opt doesn't turn on by default.
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
