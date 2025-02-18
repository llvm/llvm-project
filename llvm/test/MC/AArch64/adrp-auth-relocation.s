// RUN: llvm-mc -triple=aarch64-linux-gnu -filetype=obj -o - %s | llvm-readobj -r - | FileCheck %s
// RUN: not llvm-mc -triple=aarch64-linux-gnu_ilp32 -filetype=obj \
// RUN:   -o /dev/null %s 2>&1 | FileCheck -check-prefix=CHECK-ILP32 %s

.text
adrp x0, :got_auth:sym

.global sym
sym:

// CHECK: R_AARCH64_AUTH_ADR_GOT_PAGE sym
// CHECK-ILP32: error: ILP32 ADRP AUTH relocation not supported (LP64 eqv: AUTH_ADR_GOT_PAGE)
