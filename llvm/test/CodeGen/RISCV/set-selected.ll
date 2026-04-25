; RUN: llc -mtriple=riscv32 < %s -stop-after=riscv-isel | FileCheck %s
; RUN: llc -mtriple=riscv32 < %s -stop-after=instruction-select -global-isel | FileCheck %s

; RUN: llc -mtriple=riscv32 < %s -stop-after=riscv-isel \
; RUN:   | llc -mtriple=riscv32 -x mir - -o /dev/null

; RUN: llc -mtriple=riscv32 < %s -stop-after=riscv-isel \
; RUN:   | llc -mtriple=riscv32 -x mir - -global-isel -o /dev/null

; RUN: llc -mtriple=riscv32 < %s  -global-isel -stop-after=instruction-select \
; RUN:   | llc -mtriple=riscv32 -x mir - -global-isel -o /dev/null

; RUN: llc -mtriple=riscv32 < %s -global-isel -stop-after=instruction-select  \
; RUN:   | llc -mtriple=riscv32 -x mir - -o /dev/null

;; This test checks that SDag sets the Machine Function Property Selected, so we can
;; catch when SDag output has been run through global isel.
;;
;; It tests:
;; - That `selected: true` is on the MIR from SDag and GISel
;; - That SDag will skip selection if it gets its own or GISel output
;; - That GISel will skip selection if it gets its own or SDag output

; CHECK: selected: true

define i64 @test_64bit_ops(i64 %a, i64 %b) {
entry:
  %shl = shl i64 %a, 32
  %or = or i64 %shl, %b
  ret i64 %or
}
