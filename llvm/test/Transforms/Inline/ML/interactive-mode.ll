; RUN: rm -rf %t.rundir
; RUN: rm -rf %t.channel-basename.*
; RUN: mkdir %t.rundir
; RUN: cp %S/../../../../lib/Analysis/models/log_reader.py %t.rundir
; RUN: cp %S/../../../../lib/Analysis/models/interactive_host.py %t.rundir
; RUN: cp %S/Inputs/interactive_main.py %t.rundir
; RUN: %python %t.rundir/interactive_main.py %t.channel-basename \
; RUN:    opt -passes=scc-oz-module-inliner -interactive-model-runner-echo-reply \
; RUN:    -enable-ml-inliner=release --inliner-interactive-channel-base=%t.channel-basename %S/Inputs/test-module.ll -S -o /dev/null 2>%t.err | FileCheck %s
; RUN: cat %t.err | FileCheck %s --check-prefix=ADVICE

;; It'd be nice if we had stdout and stderr interleaved, but we don't, so
;; let's just check the features have non-zero values, and that we see as many
;; advices as observations, and that the advices flip-flop as intended.
; CHECK: context:
; CHECK-NEXT: observation: 0
; CHECK-NEXT: sroa_savings: 0
; CHECK:      unsimplified_common_instructions: 5
; CHECK:      callee_users: 3
; CHECK:      observation: 5
; CHECK-NOT:  observation: 6

; ADVICE:     inlining_decision: 1
; ADVICE:     inlining_decision: 0
; ADVICE:     inlining_decision: 1
; ADVICE:     inlining_decision: 0
; ADVICE:     inlining_decision: 1
; ADVICE:     inlining_decision: 0
