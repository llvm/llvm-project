// RUN: llvm-mc -triple=aarch64-linux-gnu -filetype=obj -o - %s| llvm-readobj -r - | FileCheck %s
// RUN: llvm-mc -target-abi=ilp32 -triple=aarch64-linux-gnu -filetype=obj \
// RUN: -o - %s| llvm-readobj -r - | FileCheck -check-prefix=CHECK-ILP32 %s
        .text
// These should produce an ADRP/ADD pair to calculate the address of
// testfn. The important point is that LLVM shouldn't think it can deal with the
// relocation on the ADRP itself (even though it knows everything about the
// relative offsets of testfn and foo) because its value depends on where this
// object file's .text section gets relocated in memory.
        adrp x0, sym
        adrp x0, :got:sym
        adrp x0, :gottprel:sym
        adrp x0, :tlsdesc:sym

        .global sym
sym:
// CHECK: R_AARCH64_ADR_PREL_PG_HI21 sym
// CHECK: R_AARCH64_ADR_GOT_PAGE sym
// CHECK: R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 sym
// CHECK: R_AARCH64_TLSDESC_ADR_PAGE21 sym
// CHECK-ILP32: R_AARCH64_P32_ADR_PREL_PG_HI21 sym
// CHECK-ILP32: R_AARCH64_P32_ADR_GOT_PAGE sym
// CHECK-ILP32: R_AARCH64_P32_TLSIE_ADR_GOTTPREL_PAGE21 sym
// CHECK-ILP32: R_AARCH64_P32_TLSDESC_ADR_PAGE21 sym
