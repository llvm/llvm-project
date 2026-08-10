# RUN: not llvm-mc -triple mips-unknown-unknown %s 2>&1 | FileCheck %s

# Error checking for malformed .module directives (and .set fp=...).

    .module fp=3
# CHECK: :[[@LINE-1]]:17: error: unsupported value, expected 'xx', '32' or '64'
# CHECK-NEXT: .module fp=3
# CHECK-NEXT:             ^

# FIXME: Add separate test for .set fp=xx/32/64.
    .set fp=xx,6
# CHECK: :[[@LINE-1]]:15: error: unexpected token, expected end of statement
# CHECK-NEXT: .set fp=xx,6
# CHECK-NEXT:           ^

    .module
# CHECK: :[[@LINE-1]]:12: error: expected .module option identifier
# CHECK-NEXT: .module
# CHECK-NEXT:        ^

    .module 34
# CHECK: :[[@LINE-1]]:13: error: expected .module option identifier
# CHECK-NEXT: .module 34
# CHECK-NEXT:         ^
