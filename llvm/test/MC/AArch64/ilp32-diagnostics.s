// RUN: not llvm-mc -triple aarch64-none-linux-gnu_ilp32 \
// RUN:   < %s 2> %t2 -filetype=obj >/dev/null
// RUN: FileCheck --check-prefix=ERROR %s --implicit-check-not=error: < %t2

.xword sym-.
// ERROR: [[#@LINE-1]]:8: error: ILP32 8 byte PC relative data relocation not supported (LP64 eqv: PREL64)

.xword sym+16
// ERROR: [[#@LINE-1]]:8: error: ILP32 8 byte absolute data relocation not supported (LP64 eqv: ABS64)

.xword sym@AUTH(da,42)
// ERROR: [[#@LINE-1]]:8: error: ILP32 8 byte absolute data relocation not supported (LP64 eqv: AUTH_ABS64)

.xword sym@AUTH(da,42,addr)
// ERROR: [[#@LINE-1]]:8: error: ILP32 8 byte absolute data relocation not supported (LP64 eqv: AUTH_ABS64)

movz x7, #:abs_g3:some_label
// ERROR: [[#@LINE-1]]:1: error: ILP32 absolute MOV relocation not supported (LP64 eqv: MOVW_UABS_G3)
// ERROR:        movz x7, #:abs_g3:some_label

movz x3, #:abs_g2:some_label
// ERROR: [[#@LINE-1]]:1: error: ILP32 absolute MOV relocation not supported (LP64 eqv: MOVW_UABS_G2)
// ERROR: movz x3, #:abs_g2:some_label

movz x19, #:abs_g2_s:some_label
// ERROR: [[#@LINE-1]]:1: error: ILP32 absolute MOV relocation not supported (LP64 eqv: MOVW_SABS_G2)
// ERROR: movz x19, #:abs_g2_s:some_label

movk x5, #:abs_g2_nc:some_label
// ERROR: [[#@LINE-1]]:1: error: ILP32 absolute MOV relocation not supported (LP64 eqv: MOVW_UABS_G2_NC)
// ERROR: movk x5, #:abs_g2_nc:some_label

movz x19, #:abs_g1_s:some_label
// ERROR: [[#@LINE-1]]:1: error: ILP32 absolute MOV relocation not supported (LP64 eqv: MOVW_SABS_G1)
// ERROR: movz x19, #:abs_g1_s:some_label

movk x5, #:abs_g1_nc:some_label
// ERROR: [[#@LINE-1]]:1: error: ILP32 absolute MOV relocation not supported (LP64 eqv: MOVW_UABS_G1_NC)
// ERROR: movk x5, #:abs_g1_nc:some_label

movz x3, #:dtprel_g2:var
// ERROR: [[#@LINE-1]]:1: error: ILP32 absolute MOV relocation not supported (LP64 eqv: TLSLD_MOVW_DTPREL_G2)
// ERROR: movz x3, #:dtprel_g2:var

movk x9, #:dtprel_g1_nc:var
// ERROR: [[#@LINE-1]]:1: error: ILP32 absolute MOV relocation not supported (LP64 eqv: TLSLD_MOVW_DTPREL_G1_NC)
// ERROR: movk x9, #:dtprel_g1_nc:var

movz x3, #:tprel_g2:var
// ERROR: [[#@LINE-1]]:1: error: ILP32 absolute MOV relocation not supported (LP64 eqv: TLSLE_MOVW_TPREL_G2)
// ERROR: movz x3, #:tprel_g2:var

movk x9, #:tprel_g1_nc:var
// ERROR: [[#@LINE-1]]:1: error: ILP32 absolute MOV relocation not supported (LP64 eqv: TLSLE_MOVW_TPREL_G1_NC)
// ERROR: movk x9, #:tprel_g1_nc:var

movz x15, #:gottprel_g1:var
// ERROR: [[#@LINE-1]]:1: error: ILP32 absolute MOV relocation not supported (LP64 eqv: TLSIE_MOVW_GOTTPREL_G1)
// ERROR: movz x15, #:gottprel_g1:var

movk x13, #:gottprel_g0_nc:var
// ERROR: [[#@LINE-1]]:1: error: ILP32 absolute MOV relocation not supported (LP64 eqv: TLSIE_MOVW_GOTTPREL_G0_NC)
// ERROR: movk x13, #:gottprel_g0_nc:var

ldr x10, [x0, #:gottprel_lo12:var]
// ERROR: [[#@LINE-1]]:1: error: ILP32 64-bit load/store relocation not supported (LP64 eqv: TLSIE_LD64_GOTTPREL_LO12_NC)
// ERROR: ldr x10, [x0, #:gottprel_lo12:var]

ldr x24, [x23, #:got_lo12:sym]
// ERROR: [[#@LINE-1]]:1: error: ILP32 64-bit load/store relocation not supported (LP64 eqv: LD64_GOT_LO12_NC)

ldr x24, [x23, :gottprel_lo12:sym]
// ERROR: [[#@LINE-1]]:1: error: ILP32 64-bit load/store relocation not supported (LP64 eqv: TLSIE_LD64_GOTTPREL_LO12_NC)

ldr x10, [x0, #:gottprel_lo12:var]
// ERROR: [[#@LINE-1]]:1: error: ILP32 64-bit load/store relocation not supported (LP64 eqv: TLSIE_LD64_GOTTPREL_LO12_NC)
// ERROR: ldr x10, [x0, #:gottprel_lo12:var]

ldr x24, [x23, #:got_lo12:sym]
// ERROR: [[#@LINE-1]]:1: error: ILP32 64-bit load/store relocation not supported (LP64 eqv: LD64_GOT_LO12_NC)

ldr x24, [x23, :gottprel_lo12:sym]
// ERROR: [[#@LINE-1]]:1: error: ILP32 64-bit load/store relocation not supported (LP64 eqv: TLSIE_LD64_GOTTPREL_LO12_NC)
