; RUN: llc %s -mtriple=arm-unknown-unknown -filetype=obj -o - \
; RUN:   | llvm-objdump -t - | FileCheck %s
; RUN: llc %s -mtriple=thumb-unknown-unknown -filetype=obj -o - \
; RUN:   | llvm-objdump -t - | FileCheck %s

; Check that the symbol address does not include the ARM/Thumb instruction
; indicator bit.
; CHECK: 00000000 g     F .text  {{[0-9]+}} test

define i32 @test() {
  ret i32 1
}
