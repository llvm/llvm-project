; RUN: llc -mtriple=x86_64-unknown-unknown -o - -stop-after machine-scheduler %s | FileCheck %s --check-prefix=PRE-RA
; RUN: llc -mtriple=x86_64-unknown-unknown -o - -stop-after prologepilog %s | FileCheck %s --check-prefix=POST-RA

; This test verifies that the virtual register references in machine function's
; liveins are cleared after register allocation.

define i32 @test(i32 %a, i32 %b) {
body:
  %c = mul i32 %a, %b
  ret i32 %c
}

; PRE-RA: liveins:
; PRE-RA-NEXT: - { reg: '%edi', virtual-reg: '%0' }
; PRE-RA-NEXT: - { reg: '%esi', virtual-reg: '%1' }

; POST-RA: liveins:
; POST-RA-NEXT: - { reg: '%edi' }
; POST-RA-NEXT: - { reg: '%esi' }
