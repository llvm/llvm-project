; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios7.0 | FileCheck %s

@var32 = global i32 0
@var64 = global i64 0

define void @test_and(i32 %in32, i64 %in64) {
; CHECK-LABEL: test_and:

  %val0 = and i32 %in32, 2863311530
  store volatile i32 %val0, ptr @var32
; CHECK: and {{w[0-9]+}}, {{w[0-9]+}}, #0xaaaaaaaa

  %val1 = and i32 %in32, 4293984240
  store volatile i32 %val1, ptr @var32
; CHECK: and {{w[0-9]+}}, {{w[0-9]+}}, #0xfff0fff0

  %val2 = and i64 %in64, 9331882296111890817
  store volatile i64 %val2, ptr @var64
; CHECK: and {{x[0-9]+}}, {{x[0-9]+}}, #0x8181818181818181

  %val3 = and i64 %in64, 18429855317404942275
  store volatile i64 %val3, ptr @var64
; CHECK: and {{x[0-9]+}}, {{x[0-9]+}}, #0xffc3ffc3ffc3ffc3

  ret void
}

define void @test_orr(i32 %in32, i64 %in64) {
; CHECK-LABEL: test_orr:

  %val0 = or i32 %in32, 2863311530
  store volatile i32 %val0, ptr @var32
; CHECK: orr {{w[0-9]+}}, {{w[0-9]+}}, #0xaaaaaaaa

  %val1 = or i32 %in32, 4293984240
  store volatile i32 %val1, ptr @var32
; CHECK: orr {{w[0-9]+}}, {{w[0-9]+}}, #0xfff0fff0

  %val2 = or i64 %in64, 9331882296111890817
  store volatile i64 %val2, ptr @var64
; CHECK: orr {{x[0-9]+}}, {{x[0-9]+}}, #0x8181818181818181

  %val3 = or i64 %in64, 18429855317404942275
  store volatile i64 %val3, ptr @var64
; CHECK: orr {{x[0-9]+}}, {{x[0-9]+}}, #0xffc3ffc3ffc3ffc3

  ret void
}

define void @test_eor(i32 %in32, i64 %in64) {
; CHECK-LABEL: test_eor:

  %val0 = xor i32 %in32, 2863311530
  store volatile i32 %val0, ptr @var32
; CHECK: eor {{w[0-9]+}}, {{w[0-9]+}}, #0xaaaaaaaa

  %val1 = xor i32 %in32, 4293984240
  store volatile i32 %val1, ptr @var32
; CHECK: eor {{w[0-9]+}}, {{w[0-9]+}}, #0xfff0fff0

  %val2 = xor i64 %in64, 9331882296111890817
  store volatile i64 %val2, ptr @var64
; CHECK: eor {{x[0-9]+}}, {{x[0-9]+}}, #0x8181818181818181

  %val3 = xor i64 %in64, 18429855317404942275
  store volatile i64 %val3, ptr @var64
; CHECK: eor {{x[0-9]+}}, {{x[0-9]+}}, #0xffc3ffc3ffc3ffc3

  ret void
}

define void @test_mov(i32 %in32, i64 %in64) {
; CHECK-LABEL: test_mov:
  %val0 = add i32 %in32, 2863311530
  store i32 %val0, ptr @var32
; CHECK: mov {{w[0-9]+}}, #-1431655766

  %val1 = add i64 %in64, 11068046444225730969
  store i64 %val1, ptr @var64
; CHECK: mov {{x[0-9]+}}, #-7378697629483820647

  ret void
; CHECK: ret
}
