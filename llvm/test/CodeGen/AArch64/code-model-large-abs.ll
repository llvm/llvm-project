; RUN: llc -mtriple=aarch64-linux-gnu -code-model=large -o - %s | FileCheck %s

@var8 = dso_local global i8 0
@var16 = dso_local global i16 0
@var32 = dso_local global i32 0
@var64 = dso_local global i64 0

define dso_local ptr @global_addr() {
; CHECK-LABEL: global_addr:
  ret ptr @var8
  ; The movz/movk calculation should end up returned directly in x0.
; CHECK: movz x0, #:abs_g0_nc:var8
; CHECK: movk x0, #:abs_g1_nc:var8
; CHECK: movk x0, #:abs_g2_nc:var8
; CHECK: movk x0, #:abs_g3:var8
; CHECK-NEXT: ret
}

define dso_local i8 @global_i8() {
; CHECK-LABEL: global_i8:
  %val = load i8, ptr @var8
  ret i8 %val
; CHECK: movz x[[ADDR_REG:[0-9]+]], #:abs_g0_nc:var8
; CHECK: movk x[[ADDR_REG]], #:abs_g1_nc:var8
; CHECK: movk x[[ADDR_REG]], #:abs_g2_nc:var8
; CHECK: movk x[[ADDR_REG]], #:abs_g3:var8
; CHECK: ldrb w0, [x[[ADDR_REG]]]
}

define dso_local i16 @global_i16() {
; CHECK-LABEL: global_i16:
  %val = load i16, ptr @var16
  ret i16 %val
; CHECK: movz x[[ADDR_REG:[0-9]+]], #:abs_g0_nc:var16
; CHECK: movk x[[ADDR_REG]], #:abs_g1_nc:var16
; CHECK: movk x[[ADDR_REG]], #:abs_g2_nc:var16
; CHECK: movk x[[ADDR_REG]], #:abs_g3:var16
; CHECK: ldrh w0, [x[[ADDR_REG]]]
}

define dso_local i32 @global_i32() {
; CHECK-LABEL: global_i32:
  %val = load i32, ptr @var32
  ret i32 %val
; CHECK: movz x[[ADDR_REG:[0-9]+]], #:abs_g0_nc:var32
; CHECK: movk x[[ADDR_REG]], #:abs_g1_nc:var32
; CHECK: movk x[[ADDR_REG]], #:abs_g2_nc:var32
; CHECK: movk x[[ADDR_REG]], #:abs_g3:var32
; CHECK: ldr w0, [x[[ADDR_REG]]]
}

define dso_local i64 @global_i64() {
; CHECK-LABEL: global_i64:
  %val = load i64, ptr @var64
  ret i64 %val
; CHECK: movz x[[ADDR_REG:[0-9]+]], #:abs_g0_nc:var64
; CHECK: movk x[[ADDR_REG]], #:abs_g1_nc:var64
; CHECK: movk x[[ADDR_REG]], #:abs_g2_nc:var64
; CHECK: movk x[[ADDR_REG]], #:abs_g3:var64
; CHECK: ldr x0, [x[[ADDR_REG]]]
}

define dso_local <2 x i64> @constpool() {
; CHECK-LABEL: constpool:
  ret <2 x i64> <i64 123456789, i64 987654321100>

; CHECK: movz x[[ADDR_REG:[0-9]+]], #:abs_g0_nc:[[CPADDR:.LCPI[0-9]+_[0-9]+]]
; CHECK: movk x[[ADDR_REG]], #:abs_g1_nc:[[CPADDR]]
; CHECK: movk x[[ADDR_REG]], #:abs_g2_nc:[[CPADDR]]
; CHECK: movk x[[ADDR_REG]], #:abs_g3:[[CPADDR]]
; CHECK: ldr q0, [x[[ADDR_REG]]]
}
