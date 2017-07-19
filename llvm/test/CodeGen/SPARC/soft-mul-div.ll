; RUN: llc -march=sparc -mcpu=v7 -O0 < %s | FileCheck %s

define i32 @test_mul32(i32 %a, i32 %b) #0 {
    ; CHECK-LABEL: test_mul32
    ; CHECK:       call .umul
    %m = mul i32 %a, %b
    ret i32 %m
}

define i16 @test_mul16(i16 %a, i16 %b) #0 {
    ; CHECK-LABEL: test_mul16
    ; CHECK:       call .umul
    %m = mul i16 %a, %b
    ret i16 %m
}

define i8 @test_mul8(i8 %a, i8 %b) #0 {
    ; CHECK-LABEL: test_mul8
    ; CHECK:       call .umul
    %m = mul i8 %a, %b
    ret i8 %m
}

define i32 @test_sdiv32(i32 %a, i32 %b) #0 {
    ; CHECK-LABEL: test_sdiv32
    ; CHECK:       call .div
    %d = sdiv i32 %a, %b
    ret i32 %d
}

define i16 @test_sdiv16(i16 %a, i16 %b) #0 {
    ; CHECK-LABEL: test_sdiv16
    ; CHECK:       call .div
    %d = sdiv i16 %a, %b
    ret i16 %d
}

define i8 @test_sdiv8(i8 %a, i8 %b) #0 {
    ; CHECK-LABEL: test_sdiv8
    ; CHECK:       call .div
    %d = sdiv i8 %a, %b
    ret i8 %d
}

define i32 @test_udiv32(i32 %a, i32 %b) #0 {
    ; CHECK-LABEL: test_udiv32
    ; CHECK:       call .udiv
    %d = udiv i32 %a, %b
    ret i32 %d
}

define i16 @test_udiv16(i16 %a, i16 %b) #0 {
    ; CHECK-LABEL: test_udiv16
    ; CHECK:       call .udiv
    %d = udiv i16 %a, %b
    ret i16 %d
}

define i8 @test_udiv8(i8 %a, i8 %b) #0 {
    ; CHECK-LABEL: test_udiv8
    ; CHECK:       call .udiv
    %d = udiv i8 %a, %b
    ret i8 %d
}

