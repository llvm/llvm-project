@ RUN: llvm-mc -triple=thumbv7-apple-ios -o - %s | FileCheck %s

        itte eq
        dmbeq #11
        dsbeq #7
        isbne #15
@ CHECK: itte eq
@ CHECK-NEXT: dmbeq ish
@ CHECK-NEXT: dsbeq nsh
@ CHECK-NEXT: isbne sy

        itet le
        dmble
        dsbgt
        isble
@ CHECK: itet le
@ CHECK-NEXT: dmble sy
@ CHECK-NEXT: dsbgt sy
@ CHECK-NEXT: isble sy

        itt gt
        cdpgt  p7, #1, c1, c1, c1, #4
        cdp2gt  p7, #1, c1, c1, c1, #4
@ CHECK: itt gt
@ CHECK-NEXT: cdpgt  p7, #1, c1, c1, c1, #4
@ CHECK-NEXT: cdp2gt  p7, #1, c1, c1, c1, #4

        itt ne
        mcrne p0, #0, r0, c0, c0, #0
        mcr2ne p0, #0, r0, c0, c0, #0
@ CHECK: itt ne
@ CHECK-NEXT: mcrne p0, #0, r0, c0, c0, #0
@ CHECK-NEXT: mcr2ne p0, #0, r0, c0, c0, #0

        ite le
        mcrrle  p7, #15, r5, r4, c1
        mcrr2gt  p7, #15, r5, r4, c1
@ CHECK: ite le
@ CHECK-NEXT: mcrrle  p7, #15, r5, r4, c1
@ CHECK-NEXT: mcrr2gt  p7, #15, r5, r4, c1

        ite eq
        mrceq p9, #1, r1, c2, c2
        mrc2ne p12, #3, r3, c3, c4
@ CHECK: ite eq
@ CHECK-NEXT: mrceq p9, #1, r1, c2, c2
@ CHECK-NEXT: mrc2ne p12, #3, r3, c3, c4

        itt lo
        mrrclo  p7, #1, r5, r4, c1
        mrrc2lo  p7, #1, r5, r4, c1
@ CHECK: itt lo
@ CHECK-NEXT: mrrclo  p7, #1, r5, r4, c1
@ CHECK-NEXT: mrrc2lo  p7, #1, r5, r4, c1
