; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-apple-darwin | FileCheck %s

; This test verifies that we choose "assembler variant 1" (which GCC
; uses for "new-style mnemonics" as opposed to POWER mnemonics) when
; processing multi-variant inline asm statements, on all subtargets.

; CHECK: subfe
; CHECK-NOT: sfe

define i32 @test(i32 %in1, i32 %in2) {
entry:
  %0 = tail call i32 asm "$(sfe$|subfe$) $0,$1,$2", "=r,r,r"(i32 %in1, i32 %in2)
  ret i32 %0
}

