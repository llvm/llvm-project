; Ensure that multiplication is lowered to function calls when the multiplier
; unit is not available in the hardware and that function calls are not used
; when the multiplier unit is available in the hardware.
;
; RUN: llc < %s -march=mblaze | FileCheck -check-prefix=FUN %s
; RUN: llc < %s -march=mblaze -mattr=+div | FileCheck -check-prefix=DIV %s

define i8 @test_i8(i8 %a, i8 %b) {
    ; FUN:        test_i8:
    ; DIV:        test_i8:

    %tmp.1 = udiv i8 %a, %b
    ; FUN-NOT:    idiv
    ; FUN:        brlid
    ; DIV-NOT:    brlid
    ; DIV:        idiv

    %tmp.2 = sdiv i8 %a, %b
    ; FUN-NOT:    idiv
    ; FUN:        brlid
    ; DIV-NOT:    brlid
    ; DIV-NOT:    idiv
    ; DIV:        idivu

    %tmp.3 = add i8 %tmp.1, %tmp.2
    ret i8 %tmp.3
    ; FUN:        rtsd
    ; DIV:        rtsd
}

define i16 @test_i16(i16 %a, i16 %b) {
    ; FUN:        test_i16:
    ; DIV:        test_i16:

    %tmp.1 = udiv i16 %a, %b
    ; FUN-NOT:    idiv
    ; FUN:        brlid
    ; DIV-NOT:    brlid
    ; DIV:        idiv

    %tmp.2 = sdiv i16 %a, %b
    ; FUN-NOT:    idiv
    ; FUN:        brlid
    ; DIV-NOT:    brlid
    ; DIV-NOT:    idiv
    ; DIV:        idivu

    %tmp.3 = add i16 %tmp.1, %tmp.2
    ret i16 %tmp.3
    ; FUN:        rtsd
    ; DIV:        rtsd
}

define i32 @test_i32(i32 %a, i32 %b) {
    ; FUN:        test_i32:
    ; DIV:        test_i32:

    %tmp.1 = udiv i32 %a, %b
    ; FUN-NOT:    idiv
    ; FUN:        brlid
    ; DIV-NOT:    brlid
    ; DIV:        idiv

    %tmp.2 = sdiv i32 %a, %b
    ; FUN-NOT:    idiv
    ; FUN:        brlid
    ; DIV-NOT:    brlid
    ; DIV-NOT:    idiv
    ; DIV:        idivu

    %tmp.3 = add i32 %tmp.1, %tmp.2
    ret i32 %tmp.3
    ; FUN:        rtsd
    ; DIV:        rtsd
}
