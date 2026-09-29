// RUN: llvm-mc -triple=aarch64-linux-gnu -filetype=obj -o - %s | llvm-readobj -r - | FileCheck %s

// CHECK: R_AARCH64_ADR_PREL_PG_HI21_NC sym
adrp x0, :pg_hi21_nc:sym

.globl sym
sym:
