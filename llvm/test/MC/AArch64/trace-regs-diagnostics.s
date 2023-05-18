// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2>&1 | FileCheck %s
        // Write-only
        mrs x12, trcoslar
        mrs x10, trclar
// CHECK: error: expected readable system register
// CHECK-NEXT:         mrs x12, trcoslar
// CHECK-NEXT:                  ^
// CHECK-NEXT: error: expected readable system register
// CHECK-NEXT:         mrs x10, trclar
// CHECK-NEXT:                  ^

        // Read-only
        msr trcstatr, x0
        msr trcidr8, x13
        msr trcidr9, x25
        msr trcidr10, x2
        msr trcidr11, x19
        msr trcidr12, x15
        msr trcidr13, x24
        msr trcidr0, x20
        msr trcidr1, x5
        msr trcidr2, x18
        msr trcidr3, x10
        msr trcidr4, x1
        msr trcidr5, x10
        msr trcidr6, x4
        msr trcidr7, x0
        msr trcoslsr, x23
        msr trcpdsr, x21
        msr trcdevaff0, x4
        msr trcdevaff1, x17
        msr trclsr, x18
        msr trcauthstatus, x10
        msr trcdevarch, x8
        msr trcdevid, x11
        msr trcdevtype, x1
        msr trcpidr4, x2
        msr trcpidr5, x7
        msr trcpidr6, x17
        msr trcpidr7, x5
        msr trcpidr0, x0
        msr trcpidr1, x16
        msr trcpidr2, x29
        msr trcpidr3, x1
        msr trccidr0, x27
        msr trccidr1, x1
        msr trccidr2, x24
        msr trccidr3, x8
// CHECK: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcstatr, x0
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr8, x13
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr9, x25
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr10, x2
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr11, x19
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr12, x15
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr13, x24
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr0, x20
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr1, x5
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr2, x18
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr3, x10
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr4, x1
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr5, x10
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr6, x4
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcidr7, x0
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcoslsr, x23
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcpdsr, x21
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcdevaff0, x4
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcdevaff1, x17
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trclsr, x18
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcauthstatus, x10
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcdevarch, x8
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcdevid, x11
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcdevtype, x1
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcpidr4, x2
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcpidr5, x7
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcpidr6, x17
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcpidr7, x5
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcpidr0, x0
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcpidr1, x16
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcpidr2, x29
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trcpidr3, x1
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trccidr0, x27
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trccidr1, x1
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trccidr2, x24
// CHECK-NEXT:             ^
// CHECK-NEXT: error: expected writable system register or pstate
// CHECK-NEXT:         msr trccidr3, x8
// CHECK-NEXT:             ^
