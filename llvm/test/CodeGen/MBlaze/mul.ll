; Ensure that multiplication is lowered to function calls when the multiplier
; unit is not available in the hardware and that function calls are not used
; when the multiplier unit is available in the hardware.
;
; RUN: llc < %s -march=mblaze | FileCheck -check-prefix=FUN %s
; RUN: llc < %s -march=mblaze -mattr=+mul | FileCheck -check-prefix=MUL %s

define i8 @test_i8(i8 %a, i8 %b) {
    ; FUN:        test_i8:
    ; MUL:        test_i8:

    %tmp.1 = mul i8 %a, %b
    ; FUN-NOT:    mul
    ; FUN:        brlid
    ; MUL-NOT:    brlid

    ret i8 %tmp.1
    ; FUN:        rtsd
    ; MUL:        rtsd
    ; MUL:        mul
}

define i16 @test_i16(i16 %a, i16 %b) {
    ; FUN:        test_i16:
    ; MUL:        test_i16:

    %tmp.1 = mul i16 %a, %b
    ; FUN-NOT:    mul
    ; FUN:        brlid
    ; MUL-NOT:    brlid

    ret i16 %tmp.1
    ; FUN:        rtsd
    ; MUL:        rtsd
    ; MUL:        mul
}

define i32 @test_i32(i32 %a, i32 %b) {
    ; FUN:        test_i32:
    ; MUL:        test_i32:

    %tmp.1 = mul i32 %a, %b
    ; FUN-NOT:    mul
    ; FUN:        brlid
    ; MUL-NOT:    brlid

    ret i32 %tmp.1
    ; FUN:        rtsd
    ; MUL:        rtsd
    ; MUL:        mul
}
