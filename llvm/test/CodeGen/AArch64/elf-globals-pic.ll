; RUN: llc -mtriple=arm64 -relocation-model=pic -o - %s -mcpu=cyclone | FileCheck %s --check-prefix=CHECK-PIC
; RUN: llc -mtriple=arm64 -O0 -fast-isel -relocation-model=pic -o - %s -mcpu=cyclone | FileCheck %s --check-prefix=CHECK-FAST-PIC

@var8 = external global i8, align 1
@var16 = external global i16, align 2
@var32 = external global i32, align 4
@var64 = external global i64, align 8

define i8 @test_i8(i8 %new) {
  %val = load i8, ptr @var8, align 1
  store i8 %new, ptr @var8
  ret i8 %val
; CHECK-PIC-LABEL: test_i8:
; CHECK-PIC: adrp x[[HIREG:[0-9]+]], :got:var8
; CHECK-PIC: ldr x[[VAR_ADDR:[0-9]+]], [x[[HIREG]], :got_lo12:var8]
; CHECK-PIC: ldrb {{w[0-9]+}}, [x[[VAR_ADDR]]]

; CHECK-FAST-PIC: adrp x[[HIREG:[0-9]+]], :got:var8
; CHECK-FAST-PIC: ldr x[[VARADDR:[0-9]+]], [x[[HIREG]], :got_lo12:var8]
; CHECK-FAST-PIC: ldr {{w[0-9]+}}, [x[[VARADDR]]]
}

define i16 @test_i16(i16 %new) {
  %val = load i16, ptr @var16, align 2
  store i16 %new, ptr @var16
  ret i16 %val
}

define i32 @test_i32(i32 %new) {
  %val = load i32, ptr @var32, align 4
  store i32 %new, ptr @var32
  ret i32 %val
}

define i64 @test_i64(i64 %new) {
  %val = load i64, ptr @var64, align 8
  store i64 %new, ptr @var64
  ret i64 %val
}

define ptr @test_addr() {
  ret ptr @var64
}

@hiddenvar = hidden global i32 0, align 4
@protectedvar = protected global i32 0, align 4

define i32 @test_vis() {
  %lhs = load i32, ptr @hiddenvar, align 4
  %rhs = load i32, ptr @protectedvar, align 4
  %ret = add i32 %lhs, %rhs
  ret i32 %ret
; CHECK-PIC-LABEL: test_vis:
; CHECK-PIC: adrp {{x[0-9]+}}, hiddenvar
; CHECK-PIC: ldr {{w[0-9]+}}, [{{x[0-9]+}}, :lo12:hiddenvar]
; CHECK-PIC: adrp {{x[0-9]+}}, protectedvar
; CHECK-PIC: ldr {{w[0-9]+}}, [{{x[0-9]+}}, :lo12:protectedvar]
}

@var_default = external global [2 x i32]

define i32 @test_default_align() {
  %val = load i32, ptr @var_default
  ret i32 %val
}

define i64 @test_default_unaligned() {
  %val = load i64, ptr @var_default
  ret i64 %val
}
