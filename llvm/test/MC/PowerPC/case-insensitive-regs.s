# RUN: llvm-mc -triple powerpc64le-unknown-unknown %s 2>&1 | FileCheck %s

# Test that upper case registers are accepted.

# CHECK-LABEL: test:
# CHECK-NEXT: ld 1, 0(3)
# CHECK-NEXT: blr

test:
    ld %R1, 0(%R3)
    blr
