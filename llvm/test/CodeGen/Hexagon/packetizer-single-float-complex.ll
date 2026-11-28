; Check that -single-float-packet and -single-complex-packet keep the packetizer
; from putting two floating point or two complex instructions in one packet.
; sfmpy is both a floating point and a complex instruction; mpyi is only
; complex. Each function has two independent multiplies that the packetizer
; puts in one packet by default.
; Do not regenerate the checks with update_llc_test_checks.py: they only check
; whether the two multiplies share a packet, not the rest of the schedule.

; RUN: llc -mtriple=hexagon < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=hexagon -single-float-packet < %s \
; RUN:   | FileCheck %s --check-prefix=FLOAT
; RUN: llc -mtriple=hexagon -single-complex-packet < %s \
; RUN:   | FileCheck %s --check-prefix=COMPLEX

; Both multiplies in one packet.
; DEFAULT-LABEL: two_fmul:
; DEFAULT:     = sfmpy(
; DEFAULT-NOT: }
; DEFAULT:     = sfmpy(

; FLOAT-LABEL: two_fmul:
; FLOAT:     = sfmpy(
; FLOAT-NOT: = sfmpy(
; FLOAT:     }
; FLOAT:     = sfmpy(

; COMPLEX-LABEL: two_fmul:
; COMPLEX:     = sfmpy(
; COMPLEX-NOT: = sfmpy(
; COMPLEX:     }
; COMPLEX:     = sfmpy(
define void @two_fmul(float %a, float %b, float %c, float %d, ptr %p, ptr %q) {
  %x = fmul float %a, %b
  %y = fmul float %c, %d
  store float %x, ptr %p
  store float %y, ptr %q
  ret void
}

; mpyi is not a floating point instruction, so only -single-complex-packet
; separates the two.
; DEFAULT-LABEL: two_mul:
; DEFAULT:     = mpyi(
; DEFAULT-NOT: }
; DEFAULT:     = mpyi(

; FLOAT-LABEL: two_mul:
; FLOAT:     = mpyi(
; FLOAT-NOT: }
; FLOAT:     = mpyi(

; COMPLEX-LABEL: two_mul:
; COMPLEX:     = mpyi(
; COMPLEX-NOT: = mpyi(
; COMPLEX:     }
; COMPLEX:     = mpyi(
define void @two_mul(i32 %a, i32 %b, i32 %c, i32 %d, ptr %p, ptr %q) {
  %x = mul i32 %a, %b
  %y = mul i32 %c, %d
  store i32 %x, ptr %p
  store i32 %y, ptr %q
  ret void
}
