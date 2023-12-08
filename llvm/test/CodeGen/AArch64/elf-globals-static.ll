; RUN: llc -mtriple=arm64 -o - %s -mcpu=cyclone | FileCheck %s
; RUN: llc -mtriple=arm64 -o - %s -O0 -fast-isel -mcpu=cyclone | FileCheck %s --check-prefix=CHECK-FAST

@var8 = external dso_local global i8, align 1
@var16 = external dso_local global i16, align 2
@var32 = external dso_local global i32, align 4
@var64 = external dso_local global i64, align 8

define i8 @test_i8(i8 %new) {
  %val = load i8, ptr @var8, align 1
  store i8 %new, ptr @var8
  ret i8 %val
; CHECK-LABEL: test_i8:
; CHECK: adrp x[[HIREG:[0-9]+]], var8
; CHECK: ldrb {{w[0-9]+}}, [x[[HIREG]], :lo12:var8]
; CHECK: strb {{w[0-9]+}}, [x[[HIREG]], :lo12:var8]

; CHECK-PIC-LABEL: test_i8:
; CHECK-PIC: adrp x[[HIREG:[0-9]+]], :got:var8
; CHECK-PIC: ldr x[[VAR_ADDR:[0-9]+]], [x[[HIREG]], :got_lo12:var8]
; CHECK-PIC: ldrb {{w[0-9]+}}, [x[[VAR_ADDR]]]

; CHECK-FAST-LABEL: test_i8:
; CHECK-FAST: adrp x[[HIREG:[0-9]+]], var8
; CHECK-FAST: ldrb {{w[0-9]+}}, [x[[HIREG]], :lo12:var8]
}

define i16 @test_i16(i16 %new) {
  %val = load i16, ptr @var16, align 2
  store i16 %new, ptr @var16
  ret i16 %val
; CHECK-LABEL: test_i16:
; CHECK: adrp x[[HIREG:[0-9]+]], var16
; CHECK: ldrh {{w[0-9]+}}, [x[[HIREG]], :lo12:var16]
; CHECK: strh {{w[0-9]+}}, [x[[HIREG]], :lo12:var16]

; CHECK-FAST-LABEL: test_i16:
; CHECK-FAST: adrp x[[HIREG:[0-9]+]], var16
; CHECK-FAST: ldrh {{w[0-9]+}}, [x[[HIREG]], :lo12:var16]
}

define i32 @test_i32(i32 %new) {
  %val = load i32, ptr @var32, align 4
  store i32 %new, ptr @var32
  ret i32 %val
; CHECK-LABEL: test_i32:
; CHECK: adrp x[[HIREG:[0-9]+]], var32
; CHECK: ldr {{w[0-9]+}}, [x[[HIREG]], :lo12:var32]
; CHECK: str {{w[0-9]+}}, [x[[HIREG]], :lo12:var32]

; CHECK-FAST-LABEL: test_i32:
; CHECK-FAST: adrp x[[HIREG:[0-9]+]], var32
; CHECK-FAST: add {{x[0-9]+}}, x[[HIREG]], :lo12:var32
}

define i64 @test_i64(i64 %new) {
  %val = load i64, ptr @var64, align 8
  store i64 %new, ptr @var64
  ret i64 %val
; CHECK-LABEL: test_i64:
; CHECK: adrp x[[HIREG:[0-9]+]], var64
; CHECK: ldr {{x[0-9]+}}, [x[[HIREG]], :lo12:var64]
; CHECK: str {{x[0-9]+}}, [x[[HIREG]], :lo12:var64]

; CHECK-FAST-LABEL: test_i64:
; CHECK-FAST: adrp x[[HIREG:[0-9]+]], var64
; CHECK-FAST: add {{x[0-9]+}}, x[[HIREG]], :lo12:var64
}

define ptr @test_addr() {
  ret ptr @var64
; CHECK-LABEL: test_addr:
; CHECK: adrp [[HIREG:x[0-9]+]], var64
; CHECK: add x0, [[HIREG]], :lo12:var64

; CHECK-FAST-LABEL: test_addr:
; CHECK-FAST: adrp [[HIREG:x[0-9]+]], var64
; CHECK-FAST: add x0, [[HIREG]], :lo12:var64
}

@var_default = external dso_local global [2 x i32]

define i32 @test_default_align() {
  %val = load i32, ptr @var_default
  ret i32 %val
; CHECK-LABEL: test_default_align:
; CHECK: adrp x[[HIREG:[0-9]+]], var_default
; CHECK: ldr w0, [x[[HIREG]], :lo12:var_default]
}

define i64 @test_default_unaligned() {
  %val = load i64, ptr @var_default
  ret i64 %val
; CHECK-LABEL: test_default_unaligned:
; CHECK: adrp [[HIREG:x[0-9]+]], var_default
; CHECK: add x[[ADDR:[0-9]+]], [[HIREG]], :lo12:var_default
; CHECK: ldr x0, [x[[ADDR]]]
}
