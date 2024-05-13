; RUN: opt --print-passes | FileCheck %s

; CHECK: Module passes:
; CHECK: no-op-module
; CHECK: Module analyses:
; CHECK: no-op-module
; CHECK: Module alias analyses:
; CHECK: globals-aa
; CHECK: CGSCC passes:
; CHECK: no-op-cgscc
; CHECK: CGSCC analyses:
; CHECK: no-op-cgscc
; CHECK: Function passes:
; CHECK: no-op-function
; CHECK: Function passes with params:
; CHECK: loop-unroll<O0;O1;O2;O3;full-unroll-max=N;no-partial;partial;no-peeling;peeling;no-profile-peeling;profile-peeling;no-runtime;runtime;no-upperbound;upperbound>
; CHECK: Function analyses:
; CHECK: no-op-function
; CHECK: Function alias analyses:
; CHECK: basic-aa
; CHECK: LoopNest passes:
; CHECK: no-op-loopnest
; CHECK: Loop passes:
; CHECK: no-op-loop
; CHECK: Loop passes with params:
; CHECK: simple-loop-unswitch<nontrivial;no-nontrivial;trivial;no-trivial>
; CHECK: Loop analyses:
; CHECK: no-op-loop
