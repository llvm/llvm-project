// RUN: not llvm-mc -triple armv7-apple-darwin -filetype=obj %s 2>&1 | FileCheck %s

// Check that the relocation size is valid.
// Check lower bound of edge case.
_foo1_valid:
    // CHECK-NOT: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    b   _foo1_valid+0x2000004
// Check outside of range of the largest accepted positive number
_foo1:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    b   _foo1+0x2000008

// Check Same as above, for smallest negative value
_foo2_valid:
    // CHECK-NOT: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    b   _foo2_valid-0x1FFFFF8
_foo2:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    b   _foo2-0x1FFFFFC

// Edge case - subtracting positive number
_foo3:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    b   _foo3-0x2000010

// Edge case - adding negative number
_foo4:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    b   _foo4+0x2000008

_foo5:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    bl  _foo5+0x2000008

_foo6:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    blx _foo6+0x2000008

// blx instruction is aligned to 16-bits.
_foo7_blx:
    // CHECK-NOT:[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    blx _foo7_blx+0x1FFFFFE

// Other branch instructions require 32-bit alignment.
_foo7:
    // CHECK:[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    bl _foo7_blx+0x1FFFFFE

_foo8:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    ble _foo8+0x2000008

_foo9:
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation out of range
    beq _foo9+0x2000008

    // Check that the relocation alignment is valid.
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    bl  _foo1+0x101
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    blx _foo1+0x101
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    b   _foo1+0x101
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    ble _foo1+0x101
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: error: Relocation not aligned
    beq _foo1+0x101
