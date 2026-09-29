; RUN: llc -mtriple=armv7s-apple-ios7.0 -show-mc-encoding %s -o - | FileCheck %s --check-prefix=CHECK-ARM
; RUN: llc -mtriple=thumbv7s-apple-ios7.0 -show-mc-encoding %s -o - | FileCheck %s --check-prefix=CHECK-THUMB
; RUN: llc -mtriple=thumbv7m-none-eabi -show-mc-encoding %s -o - | FileCheck %s --check-prefix=CHECK-THUMB

; In the ARM backend, most compares are glued to their uses so CPSR can't
; escape. However, for long ADCS chains (and last ditch fallback) the dependency
; is carried in the DAG because duplicating them can be more expensive than
; copying CPSR.

; Crafting a test for this was a little tricky, in case it breaks here are some
; notes on what I was tring to achieve:
;   + We want 2 long ADCS chains
;   + We want them to split after an initial common prefix (so that a single
;     CPSR is used twice).
;   + We want both chains to write CPSR post-split (so that the copy can't be
;     elided).
;   + We want the chains to be long enough that duplicating them is expensive.

define void @test_copy_cpsr(i128 %lhs, i128 %rhs, ptr %addr) {
; CHECK-ARM: test_copy_cpsr:
; CHECK-THUMB: test_copy_cpsr:

; CHECK-ARM: mrs [[TMP:r[0-9]+]], apsr @ encoding: [0x00,0x{{[0-9a-f]}}0,0x0f,0xe1]
; CHECK-ARM: msr APSR_nzcvq, [[TMP]] @ encoding: [0x0{{[0-9a-f]}},0xf0,0x28,0xe1]

  ; In Thumb mode v7M and v7AR have different MRS/MSR instructions that happen
  ; to overlap for the apsr case, so it's definitely worth checking both.
; CHECK-THUMB: mrs [[TMP:r[0-9]+]], apsr @ encoding: [0xef,0xf3,0x00,0x8{{[0-9a-f]}}]
; CHECK-THUMB: msr {{APSR|apsr}}_nzcvq, [[TMP]] @ encoding: [0x8{{[0-9a-f]}},0xf3,0x00,0x88]

  %sum = add i128 %lhs, %rhs
  store volatile i128 %sum, ptr %addr

  %rhs2.tmp1 = trunc i128 %rhs to i64
  %rhs2 = zext i64 %rhs2.tmp1 to i128

  %sum2 = add i128 %lhs, %rhs2
  store volatile i128 %sum2, ptr %addr

  ret void
}
