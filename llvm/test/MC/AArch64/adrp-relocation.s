// RUN: llvm-mc -triple=aarch64-linux-gnu -filetype=obj -o - %s| llvm-readobj -r - | FileCheck %s
// RUN: llvm-mc -triple=aarch64-linux-gnu_ilp32 -filetype=obj \
// RUN: -o - %s| llvm-readobj -r - | FileCheck -check-prefix=CHECK-ILP32 %s
        .text
// This tests that LLVM doesn't think it can deal with the relocation on the ADRP
// itself (even though it knows everything about the relative offsets of sym and
// the adrp instruction) because its value depends on where this object file's
// .text section gets relocated in memory.
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
