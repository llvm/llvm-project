; REQUIRES: x86_64-linux
; RUN: rm -rf %t.rundir
; RUN: rm -rf %t.channel-basename.* %t.chan.* %t.*.s
; RUN: mkdir %t.rundir
; RUN: cp %S/../../../lib/Analysis/models/log_reader.py %t.rundir
; RUN: cp %S/../../../lib/Analysis/models/interactive_host.py %t.rundir
; RUN: cp %S/Inputs/interactive_main.py %t.rundir
; RUN: %python %t.rundir/interactive_main.py %t.channel-basename \
; RUN:    llc -mtriple=x86_64-linux-unknown -regalloc=greedy -regalloc-enable-advisor=release -interactive-model-runner-echo-reply \
; RUN:    -regalloc-evict-interactive-channel-base=%t.channel-basename -regalloc-csr-cost-scale=0 %S/Inputs/two-large-fcts.ll -o /dev/null | FileCheck %s

;; Make sure we see both contexts. Also sanity-check that the advice is the
;; expected one - the index of the first legal register
; CHECK: context: SyFgets
; CHECK-NEXT: observation: 0
; CHECK-NEXT: mask: 0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
; CHECK: observation: 1
; CHECK-NEXT: mask: 0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
; CHECK: context: SyFgetsCopy
; CHECK-NEXT: observation: 0

; CHECK:      index_to_evict: 9
; CHECK-NEXT: index_to_evict: 10

;; Out-of-range priority advice is saturated, not converted.

; DEFINE: %{prio} = %python %t.rundir/interactive_main.py --priority=
; DEFINE: %{llc} = llc -mtriple=x86_64-linux-unknown -regalloc=greedy \
; DEFINE:   -regalloc-enable-priority-advisor=release %S/Inputs/two-large-fcts.ll

; RUN: %{prio}low %t.chan.low %{llc} \
; RUN:   -regalloc-priority-interactive-channel-base=%t.chan.low -o %t.low.s
; RUN: %{prio}high %t.chan.high %{llc} \
; RUN:   -regalloc-priority-interactive-channel-base=%t.chan.high -o %t.high.s
; RUN: %{prio}negative %t.chan.neg %{llc} \
; RUN:   -regalloc-priority-interactive-channel-base=%t.chan.neg -o %t.neg.s
; RUN: %{prio}huge %t.chan.huge %{llc} \
; RUN:   -regalloc-priority-interactive-channel-base=%t.chan.huge -o %t.huge.s

;; Without saturation, -1.0 wrapped to near UINT_MAX and 1e30 converted to 0.
; RUN: cmp %t.neg.s %t.low.s
; RUN: cmp %t.huge.s %t.high.s

;; Guard against the above passing vacuously: the two references must differ.
; RUN: not cmp -s %t.low.s %t.high.s
