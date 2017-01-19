; RUN: llc -mtriple=arm64-none-linux-gnu -code-model=large < %s | FileCheck %s

@var8 = global i8 0
@var16 = global i16 0
@var32 = global i32 0
@var64 = global i64 0

define i8* @global_addr() {
; CHECK-LABEL: global_addr:
  ret i8* @var8
  ; The movz/movk calculation should end up returned directly in x0.
; CHECK: movz x0, #:abs_g0_nc:var8
; CHECK: movk x0, #:abs_g1_nc:var8
; CHECK: movk x0, #:abs_g2_nc:var8
; CHECK: movk x0, #:abs_g3:var8
; CHECK-NEXT: ret
}

define i8 @global_i8() {
; CHECK-LABEL: global_i8:
  %val = load i8, i8* @var8
  ret i8 %val
; CHECK: movz x[[ADDR_REG:[0-9]+]], #:abs_g0_nc:var8
; CHECK: movk x[[ADDR_REG]], #:abs_g1_nc:var8
; CHECK: movk x[[ADDR_REG]], #:abs_g2_nc:var8
; CHECK: movk x[[ADDR_REG]], #:abs_g3:var8
; CHECK: ldrb w0, [x[[ADDR_REG]]]
}

define i16 @global_i16() {
; CHECK-LABEL: global_i16:
  %val = load i16, i16* @var16
  ret i16 %val
; CHECK: movz x[[ADDR_REG:[0-9]+]], #:abs_g0_nc:var16
; CHECK: movk x[[ADDR_REG]], #:abs_g1_nc:var16
; CHECK: movk x[[ADDR_REG]], #:abs_g2_nc:var16
; CHECK: movk x[[ADDR_REG]], #:abs_g3:var16
; CHECK: ldrh w0, [x[[ADDR_REG]]]
}

define i32 @global_i32() {
; CHECK-LABEL: global_i32:
  %val = load i32, i32* @var32
  ret i32 %val
; CHECK: movz x[[ADDR_REG:[0-9]+]], #:abs_g0_nc:var32
; CHECK: movk x[[ADDR_REG]], #:abs_g1_nc:var32
; CHECK: movk x[[ADDR_REG]], #:abs_g2_nc:var32
; CHECK: movk x[[ADDR_REG]], #:abs_g3:var32
; CHECK: ldr w0, [x[[ADDR_REG]]]
}

define i64 @global_i64() {
; CHECK-LABEL: global_i64:
  %val = load i64, i64* @var64
  ret i64 %val
; CHECK: movz x[[ADDR_REG:[0-9]+]], #:abs_g0_nc:var64
; CHECK: movk x[[ADDR_REG]], #:abs_g1_nc:var64
; CHECK: movk x[[ADDR_REG]], #:abs_g2_nc:var64
; CHECK: movk x[[ADDR_REG]], #:abs_g3:var64
; CHECK: ldr x0, [x[[ADDR_REG]]]
}

define <2 x i64> @constpool() {
; CHECK-LABEL: constpool:
  ret <2 x i64> <i64 123456789, i64 987654321100>

; CHECK: movz x[[ADDR_REG:[0-9]+]], #:abs_g0_nc:[[CPADDR:.LCPI[0-9]+_[0-9]+]]
; CHECK: movk x[[ADDR_REG]], #:abs_g1_nc:[[CPADDR]]
; CHECK: movk x[[ADDR_REG]], #:abs_g2_nc:[[CPADDR]]
; CHECK: movk x[[ADDR_REG]], #:abs_g3:[[CPADDR]]
; CHECK: ldr q0, [x[[ADDR_REG]]]
}
