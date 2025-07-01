// RUN: not llvm-mc -triple aarch64-none-linux-gnu_ilp32 \
// RUN:   < %s 2> %t2 -filetype=obj >/dev/null
// RUN: FileCheck --check-prefix=ERROR %s --implicit-check-not=error: < %t2

.xword sym-.
// ERROR: [[#@LINE-1]]:8: error: 8 byte PC relative data relocation is not supported in ILP32

.xword sym+16
// ERROR: [[#@LINE-1]]:8: error: 8 byte absolute data relocation is not supported in ILP32

.xword sym@AUTH(da,42)
// ERROR: [[#@LINE-1]]:8: error: 8 byte absolute data relocation is not supported in ILP32

.xword sym@AUTH(da,42,addr)
// ERROR: [[#@LINE-1]]:8: error: 8 byte absolute data relocation is not supported in ILP32

movz x7, #:abs_g3:some_label
// ERROR: [[#@LINE-1]]:1: error: absolute MOV relocation is not supported in ILP32
// ERROR:        movz x7, #:abs_g3:some_label

movz x3, #:abs_g2:some_label
// ERROR: [[#@LINE-1]]:1: error: absolute MOV relocation is not supported in ILP32
// ERROR: movz x3, #:abs_g2:some_label

movz x19, #:abs_g2_s:some_label
// ERROR: [[#@LINE-1]]:1: error: absolute MOV relocation is not supported in ILP32
// ERROR: movz x19, #:abs_g2_s:some_label

movk x5, #:abs_g2_nc:some_label
// ERROR: [[#@LINE-1]]:1: error: absolute MOV relocation is not supported in ILP32
// ERROR: movk x5, #:abs_g2_nc:some_label

movz x19, #:abs_g1_s:some_label
// ERROR: [[#@LINE-1]]:1: error: absolute MOV relocation is not supported in ILP32
// ERROR: movz x19, #:abs_g1_s:some_label

movk x5, #:abs_g1_nc:some_label
// ERROR: [[#@LINE-1]]:1: error: absolute MOV relocation is not supported in ILP32
// ERROR: movk x5, #:abs_g1_nc:some_label

movz x3, #:dtprel_g2:var
// ERROR: [[#@LINE-1]]:1: error: absolute MOV relocation is not supported in ILP32
// ERROR: movz x3, #:dtprel_g2:var

movk x9, #:dtprel_g1_nc:var
// ERROR: [[#@LINE-1]]:1: error: absolute MOV relocation is not supported in ILP32
// ERROR: movk x9, #:dtprel_g1_nc:var

movz x3, #:tprel_g2:var
// ERROR: [[#@LINE-1]]:1: error: absolute MOV relocation is not supported in ILP32
// ERROR: movz x3, #:tprel_g2:var

movk x9, #:tprel_g1_nc:var
// ERROR: [[#@LINE-1]]:1: error: absolute MOV relocation is not supported in ILP32
// ERROR: movk x9, #:tprel_g1_nc:var

movz x15, #:gottprel_g1:var
// ERROR: [[#@LINE-1]]:1: error: absolute MOV relocation is not supported in ILP32
// ERROR: movz x15, #:gottprel_g1:var

movk x13, #:gottprel_g0_nc:var
// ERROR: [[#@LINE-1]]:1: error: absolute MOV relocation is not supported in ILP32
// ERROR: movk x13, #:gottprel_g0_nc:var

ldr x10, [x0, #:gottprel_lo12:var]
// ERROR: [[#@LINE-1]]:1: error: 64-bit load/store relocation is not supported in ILP32
// ERROR: ldr x10, [x0, #:gottprel_lo12:var]

ldr x24, [x23, #:got_lo12:sym]
// ERROR: [[#@LINE-1]]:1: error: 64-bit load/store relocation is not supported in ILP32

ldr x24, [x23, #:got_auth_lo12:sym]
// ERROR: [[#@LINE-1]]:1: error: 64-bit load/store relocation is not supported in ILP32

add x24, x23, #:got_auth_lo12:sym
// ERROR: [[#@LINE-1]]:1: error: ADD AUTH relocation is not supported in ILP32

ldr x24, [x23, :gottprel_lo12:sym]
// ERROR: [[#@LINE-1]]:1: error: 64-bit load/store relocation is not supported in ILP32

ldr x10, [x0, #:gottprel_lo12:var]
// ERROR: [[#@LINE-1]]:1: error: 64-bit load/store relocation is not supported in ILP32
// ERROR: ldr x10, [x0, #:gottprel_lo12:var]

ldr x24, [x23, #:got_lo12:sym]
// ERROR: [[#@LINE-1]]:1: error: 64-bit load/store relocation is not supported in ILP32

ldr x24, [x23, :gottprel_lo12:sym]
// ERROR: [[#@LINE-1]]:1: error: 64-bit load/store relocation is not supported in ILP32

ldr x24, :got_auth:sym
// ERROR: [[#@LINE-1]]:1: error: LDR AUTH relocation is not supported in ILP32

adr x24, :got_auth:sym
// ERROR: [[#@LINE-1]]:1: error: ADR AUTH relocation is not supported in ILP32

adrp x24, :tlsdesc_auth:sym
// ERROR: [[#@LINE-1]]:1: error: ADRP AUTH relocation is not supported in ILP32

ldr x24, [x23, :tlsdesc_auth_lo12:sym]
// ERROR: [[#@LINE-1]]:1: error: 64-bit load/store AUTH relocation is not supported in ILP32

add x24, x23, :tlsdesc_auth_lo12:sym
// ERROR: [[#@LINE-1]]:1: error: ADD AUTH relocation is not supported in ILP32
