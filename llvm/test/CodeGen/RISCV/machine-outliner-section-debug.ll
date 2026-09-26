; Check debug output for section-aware machine outlining.
;
; REQUIRES: asserts
; RUN: llc -mtriple=riscv32 -enable-machine-outliner=always \
; RUN:   -debug-only=machine-outliner \
; RUN:   %S/machine-outliner-multiple-sections.ll -o /dev/null 2>&1 | FileCheck %s

; CHECK: *** Section Aware Outlining is enabled for the target ***
; CHECK: Input section partitions: 4
; CHECK: .. section '.sec_shared': 3 candidates
; CHECK: .. section '<none>': 3 candidates
; CHECK: .. section '.sec_a': 2 candidates
; CHECK: .. section '.sec_b': 2 candidates
; CHECK: INHERITED SECTION: .sec_shared
; CHECK: INHERITED SECTION: .sec_a
; CHECK: INHERITED SECTION: .sec_b
