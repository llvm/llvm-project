; Ensure that multiplication is lowered to function calls when the 64-bit
; multiplier unit is not available in the hardware and that function calls
; are not used when the 64-bit multiplier unit is available in the hardware.
;
; RUN: llc < %s -march=mblaze | FileCheck -check-prefix=FUN %s
; RUN: llc < %s -march=mblaze -mattr=+mul,+mul64 | \
; RUN:      FileCheck -check-prefix=MUL %s

define i64 @test_i64(i64 %a, i64 %b) {
    ; FUN:        test_i64:
    ; MUL:        test_i64:

    %tmp.1 = mul i64 %a, %b
    ; FUN-NOT:    mul
    ; FUN:        brlid
    ; MUL-NOT:    brlid
    ; MUL:        mulh
    ; MUL:        mul

    ret i64 %tmp.1
    ; FUN:        rtsd
    ; MUL:        rtsd
}
