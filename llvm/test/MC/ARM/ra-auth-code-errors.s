// RUN: not llvm-mc -triple=thumbv7 %s -o - 2>&1 | FileCheck  %s --strict-whitespace
        .text
    .syntax unified
    .code 16
    .thumb_func
    .global f
f:
    .fnstart
    .save {r11-ra_auth_code}
// CHECK: [[# @LINE - 1]]:16: error: pseudo-register not allowed
// CHECK-NEXT:    .save {r11-ra_auth_code}
// CHECK-NEXT:               ^
    .save {r11, ra_auth_code, r12}
// CHECK: [[# @LINE - 1]]:31: warning: duplicated register (r12) in register list
// CHECK-NEXT:    .save {r11, ra_auth_code, r12}
// CHECK-NEXT:                              ^
    .save {ra_auth_code-r13}
// CHECK: [[# @LINE - 1]]:12: error: pseudo-register not allowed
// CHECK-NEXT:    .save {ra_auth_code-r13}
// CHECK-NEXT:           ^
    push {ra_auth_code}
// CHECK: [[# @LINE - 1]]:11: error: pseudo-register not allowed
// CHECK-NEXT:    push {ra_auth_code}
// CHECK-NEXT:          ^
    push {r11, ra_auth_code}
// CHECK: [[# @LINE - 1]]:16: error: pseudo-register not allowed
// CHECK-NEXT:    push {r11, ra_auth_code}
// CHECK-NEXT:               ^
    push {ra_auth_code, r12}
// CHECK: [[# @LINE - 1]]:11: error: pseudo-register not allowed
// CHECK-NEXT:    push {ra_auth_code, r12}
// CHECK-NEXT:          ^
    push {ra_auth_code, r13}
// CHECK: [[# @LINE - 1]]:11: error: pseudo-register not allowed
// CHECK-NEXT:    push {ra_auth_code, r13}
// CHECK-NEXT:          ^
