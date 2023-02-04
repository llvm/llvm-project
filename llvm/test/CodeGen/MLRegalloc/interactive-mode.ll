; RUN: rm -rf %t.rundir
; RUN: rm -rf %t.channel-basename.*
; RUN: mkdir %t.rundir
; RUN: cp %S/../../../lib/Analysis/models/log_reader.py %t.rundir
; RUN: cp %S/../../../lib/Analysis/models/interactive_host.py %t.rundir
; RUN: cp %S/Inputs/interactive_main.py %t.rundir
; RUN: %python %t.rundir/interactive_main.py %t.channel-basename \
; RUN:    llc -mtriple=x86_64-linux-unknown -regalloc=greedy -regalloc-enable-advisor=release -interactive-model-runner-echo-reply \
; RUN:    -regalloc-evict-interactive-channel-base=%t.channel-basename %S/Inputs/two-large-fcts.ll -o /dev/null 2>%t.err | FileCheck %s
; RUN: cat %t.err | FileCheck %s --check-prefix=ADVICE

;; Make sure we see both contexts. Also sanity-check that the advice is the
;; expected one - the index of the first legal register
; CHECK: context: SyFgets
; CHECK-NEXT: observation: 0
; CHECK-NEXT: mask: 0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
; CHECK: observation: 1
; CHECK-NEXT: mask: 0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
; CHECK: context: SyFgetsCopy
; CHECK-NEXT: observation: 0

; ADVICE: index_to_evict: 9
; ADVICE: index_to_evict: 10
