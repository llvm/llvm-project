; NOTE: Do not autogenerate
; RUN: llc -O1 -debug -global-isel -mtriple=aarch64-unknown-linux-gnu -filetype=null %s 2>&1 | FileCheck %s --check-prefix=LLC-Ox
; RUN: llc -O2 -debug -global-isel -mtriple=aarch64-unknown-linux-gnu -filetype=null %s 2>&1 | FileCheck %s --check-prefix=LLC-Ox
; RUN: llc -O3 -debug -global-isel -mtriple=aarch64-unknown-linux-gnu -filetype=null %s 2>&1 | FileCheck %s --check-prefix=LLC-Ox

; REQUIRES: asserts

; This test verifies that we don't run Machine Function optimizations
; on optnone functions.

define i32 @_Z3fooi(i32 %x) #0 {
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
  ret i32 0
}

attributes #0 = { optnone noinline }

; Machine Function passes run at -O1 and higher.
; LLC-Ox-DAG: Skipping pass 'Branch Probability Basic Block Placement'
; LLC-Ox-DAG: Skipping pass 'CodeGen Prepare'
; LLC-Ox-DAG: Skipping pass 'Control Flow Optimizer'
; LLC-Ox-DAG: Skipping pass 'Machine code sinking'
; LLC-Ox-DAG: Skipping pass 'Machine Common Subexpression Elimination'
; LLC-Ox-DAG: Skipping pass 'Shrink Wrapping analysis'
; LLC-Ox-DAG: Skipping pass 'Machine Copy Propagation Pass'
; LLC-Ox-DAG: Skipping pass 'Machine Instruction Scheduler'
; LLC-Ox-DAG: Skipping pass 'Machine Loop Invariant Code Motion'
; LLC-Ox-DAG: Skipping pass 'Optimize machine instruction PHIs'
; LLC-Ox-DAG: Skipping pass 'Peephole Optimizations'
; LLC-Ox-DAG: Skipping pass 'Post{{.*}}RA{{.*}}{{[Ss]}}cheduler'
; LLC-Ox-DAG: Skipping pass 'Remove dead machine instructions'
; LLC-Ox-DAG: Skipping pass 'Tail Duplication'
