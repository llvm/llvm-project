// RUN: not llvm-mc -triple arm64-apple-ios -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s
        .global _foo
        adrp x0, (_foo + 1)@PAGE
        adrp x0, (_foo - 1)@PAGE
        adrp x0, (_foo + 0x7fffff)@PAGE
        adrp x0, (_foo - 0x800000)@PAGE

        // CHECK-NOT: error:
        // CHECK: error: addend too big for relocation
        // CHECK:      adrp x0, (_foo + 0x800000)@PAGE
        // CHECK: error: addend too big for relocation
        // CHECK:      adrp x0, (_foo - 0x800001)@PAGE
        adrp x0, (_foo + 0x800000)@PAGE
        adrp x0, (_foo - 0x800001)@PAGE
