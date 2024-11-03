; REQUIRES: x86_64-linux
; RUN: rm -rf %t.rundir
; RUN: rm -rf %t.channel-basename.*
; RUN: mkdir %t.rundir
; RUN: cp %S/../../../../lib/Analysis/models/log_reader.py %t.rundir
; RUN: cp %S/../../../../lib/Analysis/models/interactive_host.py %t.rundir
; RUN: cp %S/Inputs/interactive_main.py %t.rundir
; RUN: %python %t.rundir/interactive_main.py %t.channel-basename \
; RUN:    opt -passes=scc-oz-module-inliner -interactive-model-runner-echo-reply \
; RUN:    -enable-ml-inliner=release -inliner-interactive-channel-base=%t.channel-basename %S/Inputs/test-module.ll -S -o /dev/null | FileCheck %s
; RUN: %python %t.rundir/interactive_main.py %t.channel-basename \
; RUN:    opt -passes=scc-oz-module-inliner -interactive-model-runner-echo-reply \
; RUN:    -inliner-interactive-include-default -enable-ml-inliner=release \
; RUN:    -inliner-interactive-channel-base=%t.channel-basename %S/Inputs/test-module.ll -S -o /dev/null | FileCheck %s -check-prefixes=CHECK,CHECK-DEFAULT


;; It'd be nice if we had stdout and stderr interleaved, but we don't, so
;; let's just check the features have non-zero values, and that we see as many
;; advices as observations, and that the advices flip-flop as intended.
; CHECK: context:
; CHECK-NEXT: observation: 0
; CHECK-NEXT: sroa_savings: 0
; CHECK:      unsimplified_common_instructions: 5
; CHECK:      callee_users: 3
; CHECK-DEFAULT: inlining_default: 0
; CHECK:      observation: 5
; CHECK-NOT:  observation: 6

; CHECK:      inlining_decision: 1
; CHECK-NEXT: inlining_decision: 0
; CHECK-NEXT: inlining_decision: 1
; CHECK-NEXT: inlining_decision: 0
; CHECK-NEXT: inlining_decision: 1
; CHECK-NEXT: inlining_decision: 0
