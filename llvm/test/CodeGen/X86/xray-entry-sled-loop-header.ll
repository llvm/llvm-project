; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs \
; RUN:     -stop-after=xray-instrumentation < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=ASM

; Tail recursion elimination can leave the block holding the first instruction
; as a loop header. The entry sled must stay outside that loop, or it reports
; one entry per iteration against a single exit.

define ptr @tailrecurse(ptr %p) nounwind noinline "function-instrument"="xray-always" {
entry:
  br label %loop

loop:
  %cur = phi ptr [ %p, %entry ], [ %next, %loop ]
  %next = load ptr, ptr %cur, align 8
  %done = icmp eq ptr %next, null
  br i1 %done, label %exit, label %loop

exit:
  ret ptr %cur
}

; MIR-LABEL: name: tailrecurse
; MIR:       bb.0.entry:
; MIR-NEXT:    successors: %bb.1(0x80000000)
; MIR:         PATCHABLE_FUNCTION_ENTER
; MIR:       bb.1.loop
; MIR-NEXT:    successors: {{.*}}%bb.1
; MIR-NOT:     PATCHABLE_FUNCTION_ENTER

; The backedge target label must come after the sled.

; ASM-LABEL: tailrecurse:
; ASM:       .Lxray_sled_0:
; ASM-NEXT:    .ascii "\353\t"
; ASM-NEXT:    nopw 512(%rax,%rax)
; ASM:       .LBB0_1:
; ASM:         jne .LBB0_1
