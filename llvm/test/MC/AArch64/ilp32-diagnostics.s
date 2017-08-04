// RUN: not llvm-mc -triple aarch64-none-linux-gnu -target-abi=ilp32 \
// RUN:  < %s 2> %t2 -filetype=obj >/dev/null
// RUN: FileCheck --check-prefix=CHECK-ERROR %s < %t2

        .xword sym-.
// CHECK-ERROR: error: ILP32 8 byte PC relative data relocation not supported (LP64 eqv: PREL64)
// CHECK-ERROR: ^

        .xword sym+16
// CHECK-ERROR: error: ILP32 8 byte absolute data relocation not supported (LP64 eqv: ABS64)
// CHECK-ERROR: ^

        movz x7, #:abs_g3:some_label
// CHECK-ERROR: error: ILP32 absolute MOV relocation not supported (LP64 eqv: MOVW_UABS_G3)
// CHECK-ERROR:        movz x7, #:abs_g3:some_label
// CHECK-ERROR:        ^

        movz x3, #:abs_g2:some_label
// CHECK-ERROR: error: ILP32 absolute MOV relocation not supported (LP64 eqv: MOVW_UABS_G2)
// CHECK-ERROR: movz x3, #:abs_g2:some_label
// CHECK-ERROR: ^

        movz x19, #:abs_g2_s:some_label
// CHECK-ERROR: error: ILP32 absolute MOV relocation not supported (LP64 eqv: MOVW_SABS_G2)
// CHECK-ERROR: movz x19, #:abs_g2_s:some_label
// CHECK-ERROR: ^

        movk x5, #:abs_g2_nc:some_label
// CHECK-ERROR: error: ILP32 absolute MOV relocation not supported (LP64 eqv: MOVW_UABS_G2_NC)
// CHECK-ERROR: movk x5, #:abs_g2_nc:some_label
// CHECK-ERROR: ^

        movz x19, #:abs_g1_s:some_label
// CHECK-ERROR: error: ILP32 absolute MOV relocation not supported (LP64 eqv: MOVW_SABS_G1)
// CHECK-ERROR: movz x19, #:abs_g1_s:some_label
// CHECK-ERROR: ^

        movk x5, #:abs_g1_nc:some_label
// CHECK-ERROR: error: ILP32 absolute MOV relocation not supported (LP64 eqv: MOVW_UABS_G1_NC)
// CHECK-ERROR: movk x5, #:abs_g1_nc:some_label
// CHECK-ERROR: ^

        movz x3, #:dtprel_g2:var
// CHECK-ERROR: error: ILP32 absolute MOV relocation not supported (LP64 eqv: TLSLD_MOVW_DTPREL_G2)
// CHECK-ERROR: movz x3, #:dtprel_g2:var
// CHECK-ERROR: ^

        movk x9, #:dtprel_g1_nc:var
// CHECK-ERROR: error: ILP32 absolute MOV relocation not supported (LP64 eqv: TLSLD_MOVW_DTPREL_G1_NC)
// CHECK-ERROR: movk x9, #:dtprel_g1_nc:var
// CHECK-ERROR: ^

        movz x3, #:tprel_g2:var
// CHECK-ERROR: error: ILP32 absolute MOV relocation not supported (LP64 eqv: TLSLE_MOVW_TPREL_G2)
// CHECK-ERROR: movz x3, #:tprel_g2:var
// CHECK-ERROR: ^

        movk x9, #:tprel_g1_nc:var
// CHECK-ERROR: error: ILP32 absolute MOV relocation not supported (LP64 eqv: TLSLE_MOVW_TPREL_G1_NC)
// CHECK-ERROR: movk x9, #:tprel_g1_nc:var
// CHECK-ERROR: ^

        movz x15, #:gottprel_g1:var
// CHECK-ERROR: error: ILP32 absolute MOV relocation not supported (LP64 eqv: TLSIE_MOVW_GOTTPREL_G1)
// CHECK-ERROR: movz x15, #:gottprel_g1:var
// CHECK-ERROR: ^

        movk x13, #:gottprel_g0_nc:var
// CHECK-ERROR: error: ILP32 absolute MOV relocation not supported (LP64 eqv: TLSIE_MOVW_GOTTPREL_G0_NC)
// CHECK-ERROR: movk x13, #:gottprel_g0_nc:var
// CHECK-ERROR: ^

        ldr x10, [x0, #:gottprel_lo12:var]
// CHECK-ERROR: error: ILP32 64-bit load/store relocation not supported (LP64 eqv: TLSIE_LD64_GOTTPREL_LO12_NC)
// CHECK-ERROR: ldr x10, [x0, #:gottprel_lo12:var]
// CHECK-ERROR: ^

   ldr x24, [x23, #:got_lo12:sym]
// CHECK-ERROR: error: ILP32 64-bit load/store relocation not supported (LP64 eqv: LD64_GOT_LO12_NC)
// CHECK-ERROR: ^

   ldr x24, [x23, :gottprel_lo12:sym]
// CHECK-ERROR: error: ILP32 64-bit load/store relocation not supported (LP64 eqv: TLSIE_LD64_GOTTPREL_LO12_NC)
// CHECK-ERROR: ^

        ldr x10, [x0, #:gottprel_lo12:var]
// CHECK-ERROR: error: ILP32 64-bit load/store relocation not supported (LP64 eqv: TLSIE_LD64_GOTTPREL_LO12_NC)
// CHECK-ERROR: ldr x10, [x0, #:gottprel_lo12:var]
// CHECK-ERROR: ^

   ldr x24, [x23, #:got_lo12:sym]
// CHECK-ERROR: error: ILP32 64-bit load/store relocation not supported (LP64 eqv: LD64_GOT_LO12_NC)
// CHECK-ERROR: ^

   ldr x24, [x23, :gottprel_lo12:sym]
// CHECK-ERROR: error: ILP32 64-bit load/store relocation not supported (LP64 eqv: TLSIE_LD64_GOTTPREL_LO12_NC)
// CHECK-ERROR: ^
