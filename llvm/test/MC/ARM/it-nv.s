@ RUN: not llvm-mc -triple thumbv7m-apple-macho %s 2> %t.errs
@ RUN: FileCheck %s < %t.errs --check-prefix=CHECK-ERRS

@ CHECK-ERRS: error: unpredictable IT predicate sequence
@ CHECK-ERRS:     ite al
@ CHECK-ERRS: error: unpredictable IT predicate sequence
@ CHECK-ERRS:     itee al
@ CHECK-ERRS: error: unpredictable IT predicate sequence
@ CHECK-ERRS:     itet al
@ CHECK-ERRS: error: unpredictable IT predicate sequence
@ CHECK-ERRS:     itte al
@ CHECK-ERRS: error: unpredictable IT predicate sequence
@ CHECK-ERRS:     ittte al
    ite al
    itee al
    itet al
    itte al
    ittte al

@ CHECK-ERRS-NOT: error
    it al
    nop

    itt al
    nop
    nop

    ittt al
    nop
    nop
    nop

    itttt al
    nop
    nop
    nop
    nop

    ite eq
    nopeq
    nopne

    iteet hi
    nophi
    nopls
    nopls
    nophi
