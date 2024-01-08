; NVPTX customizes the list of passes so the test cannot find what it expects
; XFAIL: target=nvptx{{.*}}

; Note: -verify-machineinstrs is used in order to make this test compatible with EXPENSIVE_CHECKS.
; RUN: llc < %s -debug-pass=Structure -stop-after=loop-reduce -verify-machineinstrs -o /dev/null 2>&1 \
; RUN:   | FileCheck %s -check-prefix=STOP-AFTER
; STOP-AFTER: -loop-reduce
; STOP-AFTER: Dominator Tree Construction
; STOP-AFTER: Loop Strength Reduction
; STOP-AFTER-NEXT: Verify generated machine code
; STOP-AFTER-NEXT: Lazy Machine Block Frequency Analysis
; STOP-AFTER-NEXT: Machine Optimization Remark Emitter
; STOP-AFTER-NEXT: Stack Frame Layout Analysis
; STOP-AFTER-NEXT: MIR Printing Pass

; RUN: llc < %s -debug-pass=Structure -stop-before=loop-reduce -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-BEFORE
; STOP-BEFORE-NOT: -loop-reduce
; STOP-BEFORE: Dominator Tree Construction
; STOP-BEFORE-NOT: Loop Strength Reduction

; RUN: llc < %s -debug-pass=Structure -start-after=loop-reduce -o /dev/null 2>&1 | FileCheck %s -check-prefix=START-AFTER
; START-AFTER: -gc-lowering
; START-AFTER: FunctionPass Manager
; START-AFTER-NEXT: Lower Garbage Collection Instructions

; RUN: llc < %s -debug-pass=Structure -start-before=loop-reduce -o /dev/null 2>&1 | FileCheck %s -check-prefix=START-BEFORE
; START-BEFORE: -machine-branch-prob -regalloc-evict -regalloc-priority -domtree
; START-BEFORE: FunctionPass Manager
; START-BEFORE: Loop Strength Reduction
; START-BEFORE-NEXT: Lower Garbage Collection Instructions

; RUN: not --crash llc < %s -start-before=nonexistent -o /dev/null 2>&1 | FileCheck %s -check-prefix=NONEXISTENT-START-BEFORE
; RUN: not --crash llc < %s -stop-before=nonexistent -o /dev/null 2>&1 | FileCheck %s -check-prefix=NONEXISTENT-STOP-BEFORE
; RUN: not --crash llc < %s -start-after=nonexistent -o /dev/null 2>&1 | FileCheck %s -check-prefix=NONEXISTENT-START-AFTER
; RUN: not --crash llc < %s -stop-after=nonexistent -o /dev/null 2>&1 | FileCheck %s -check-prefix=NONEXISTENT-STOP-AFTER
; NONEXISTENT-START-BEFORE: "nonexistent" pass is not registered.
; NONEXISTENT-STOP-BEFORE: "nonexistent" pass is not registered.
; NONEXISTENT-START-AFTER: "nonexistent" pass is not registered.
; NONEXISTENT-STOP-AFTER: "nonexistent" pass is not registered.

; RUN: not --crash llc < %s -start-before=loop-reduce -start-after=loop-reduce -o /dev/null 2>&1 | FileCheck %s -check-prefix=DOUBLE-START
; RUN: not --crash llc < %s -stop-before=loop-reduce -stop-after=loop-reduce -o /dev/null 2>&1 | FileCheck %s -check-prefix=DOUBLE-STOP
; DOUBLE-START: start-before and start-after specified!
; DOUBLE-STOP: stop-before and stop-after specified!
