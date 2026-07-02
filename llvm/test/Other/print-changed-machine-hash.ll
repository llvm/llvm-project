; REQUIRES: x86-registered-target
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=null -O0 \
; RUN:   -print-changed=quiet,hash-bb 2>&1 < %s | FileCheck %s \
; RUN:   --check-prefix=BB

define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 1
  ret i32 %a
}

; BB: *** IR Dump After X86 DAG->DAG Instruction Selection (x86-isel) on f ***
; BB: bb.0.entry:
; BB: ADD32ri
