; Ensure that shifts are lowered to loops when the barrel shifter unit is
; not available in the hardware and that loops are not used when the
; barrel shifter unit is available in the hardware.
;
; RUN: llc < %s -march=mblaze | FileCheck -check-prefix=FUN %s
; RUN: llc < %s -march=mblaze -mattr=+barrel | FileCheck -check-prefix=SHT %s

define i8 @test_i8(i8 %a, i8 %b) {
    ; FUN:        test_i8:
    ; SHT:        test_i8:

    %tmp.1 = shl i8 %a, %b
    ; FUN:        andi
    ; FUN:        add
    ; FUN:        bnei
    ; SHT-NOT:    bnei

    ret i8 %tmp.1
    ; FUN:        rtsd
    ; SHT:        rtsd
    ; FUN-NOT:    bsll
    ; SHT-NEXT:   bsll
}

define i8 @testc_i8(i8 %a, i8 %b) {
    ; FUN:        testc_i8:
    ; SHT:        testc_i8:

    %tmp.1 = shl i8 %a, 5
    ; FUN:        andi
    ; FUN:        add
    ; FUN:        bnei
    ; SHT-NOT:    andi
    ; SHT-NOT:    add
    ; SHT-NOT:    bnei

    ret i8 %tmp.1
    ; FUN:        rtsd
    ; SHT:        rtsd
    ; FUN-NOT:    bsll
    ; SHT-NEXT:   bslli
}

define i16 @test_i16(i16 %a, i16 %b) {
    ; FUN:        test_i16:
    ; SHT:        test_i16:

    %tmp.1 = shl i16 %a, %b
    ; FUN:        andi
    ; FUN:        add
    ; FUN:        bnei
    ; SHT-NOT:    bnei

    ret i16 %tmp.1
    ; FUN:        rtsd
    ; SHT:        rtsd
    ; FUN-NOT:    bsll
    ; SHT-NEXT:   bsll
}

define i16 @testc_i16(i16 %a, i16 %b) {
    ; FUN:        testc_i16:
    ; SHT:        testc_i16:

    %tmp.1 = shl i16 %a, 5
    ; FUN:        andi
    ; FUN:        add
    ; FUN:        bnei
    ; SHT-NOT:    andi
    ; SHT-NOT:    add
    ; SHT-NOT:    bnei

    ret i16 %tmp.1
    ; FUN:        rtsd
    ; SHT:        rtsd
    ; FUN-NOT:    bsll
    ; SHT-NEXT:   bslli
}

define i32 @test_i32(i32 %a, i32 %b) {
    ; FUN:        test_i32:
    ; SHT:        test_i32:

    %tmp.1 = shl i32 %a, %b
    ; FUN:        andi
    ; FUN:        add
    ; FUN:        bnei
    ; SHT-NOT:    andi
    ; SHT-NOT:    bnei

    ret i32 %tmp.1
    ; FUN:        rtsd
    ; SHT:        rtsd
    ; FUN-NOT:    bsll
    ; SHT-NEXT:   bsll
}

define i32 @testc_i32(i32 %a, i32 %b) {
    ; FUN:        testc_i32:
    ; SHT:        testc_i32:

    %tmp.1 = shl i32 %a, 5
    ; FUN:        andi
    ; FUN:        add
    ; FUN:        bnei
    ; SHT-NOT:    andi
    ; SHT-NOT:    add
    ; SHT-NOT:    bnei

    ret i32 %tmp.1
    ; FUN:        rtsd
    ; SHT:        rtsd
    ; FUN-NOT:    bsll
    ; SHT-NEXT:   bslli
}
