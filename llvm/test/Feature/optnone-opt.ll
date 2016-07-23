; RUN: opt     -S -debug %s 2>&1 | FileCheck %s --check-prefix=OPT-O0
; RUN: opt -O1 -S -debug %s 2>&1 | FileCheck %s --check-prefix=OPT-O1
; RUN: opt -O2 -S -debug %s 2>&1 | FileCheck %s --check-prefix=OPT-O1 --check-prefix=OPT-O2O3
; RUN: opt -O3 -S -debug %s 2>&1 | FileCheck %s --check-prefix=OPT-O1 --check-prefix=OPT-O2O3
; RUN: opt -bb-vectorize -dce -die -loweratomic -S -debug %s 2>&1 | FileCheck %s --check-prefix=OPT-MORE
; RUN: opt -indvars -licm -loop-deletion -loop-extract -loop-idiom -loop-instsimplify -loop-reduce -loop-reroll -loop-rotate -loop-unroll -loop-unswitch -S -debug %s 2>&1 | FileCheck %s --check-prefix=OPT-LOOP

; REQUIRES: asserts

; This test verifies that we don't run target independent IR-level
; optimizations on optnone functions.

; Function Attrs: noinline optnone
define i32 @_Z3fooi(i32 %x) #0 {
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
  ret i32 0
}

attributes #0 = { optnone noinline }

; Nothing that runs at -O0 gets skipped.
; OPT-O0-NOT: Skipping pass

; IR passes run at -O1 and higher.
; OPT-O1-DAG: Skipping pass 'Aggressive Dead Code Elimination'
; OPT-O1-DAG: Skipping pass 'Combine redundant instructions'
; OPT-O1-DAG: Skipping pass 'Dead Store Elimination'
; OPT-O1-DAG: Skipping pass 'Early CSE'
; OPT-O1-DAG: Skipping pass 'Jump Threading'
; OPT-O1-DAG: Skipping pass 'MemCpy Optimization'
; OPT-O1-DAG: Skipping pass 'Reassociate expressions'
; OPT-O1-DAG: Skipping pass 'Simplify the CFG'
; OPT-O1-DAG: Skipping pass 'Sparse Conditional Constant Propagation'
; OPT-O1-DAG: Skipping pass 'SROA'
; OPT-O1-DAG: Skipping pass 'Tail Call Elimination'
; OPT-O1-DAG: Skipping pass 'Value Propagation'

; Additional IR passes run at -O2 and higher.
; OPT-O2O3-DAG: Skipping pass 'Global Value Numbering'
; OPT-O2O3-DAG: Skipping pass 'SLP Vectorizer'

; Additional IR passes that opt doesn't turn on by default.
; OPT-MORE-DAG: Skipping pass 'Basic-Block Vectorization'
; OPT-MORE-DAG: Skipping pass 'Dead Code Elimination'
; OPT-MORE-DAG: Skipping pass 'Dead Instruction Elimination'
; OPT-MORE-DAG: Skipping pass 'Lower atomic intrinsics

; Loop IR passes that opt doesn't turn on by default.
; OPT-LOOP-DAG: Skipping pass 'Delete dead loops'
; OPT-LOOP-DAG: Skipping pass 'Extract loops into new functions'
; OPT-LOOP-DAG: Skipping pass 'Induction Variable Simplification'
; OPT-LOOP-DAG: Skipping pass 'Loop Invariant Code Motion'
; OPT-LOOP-DAG: Skipping pass 'Loop Strength Reduction'
; OPT-LOOP-DAG: Skipping pass 'Recognize loop idioms'
; OPT-LOOP-DAG: Skipping pass 'Reroll loops'
; OPT-LOOP-DAG: Skipping pass 'Rotate Loops'
; OPT-LOOP-DAG: Skipping pass 'Simplify instructions in loops'
; OPT-LOOP-DAG: Skipping pass 'Unroll loops'
; OPT-LOOP-DAG: Skipping pass 'Unswitch loops'
