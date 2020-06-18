## Negative tests:
##  - Feed integer value to string type attribute.
##  - Feed string value to integer type attribute.
##  - Invalid arch string.

# RUN: not llvm-mc %s -triple=riscv32 -filetype=asm 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple=riscv64 -filetype=asm 2>&1 | FileCheck %s

.attribute arch, "foo"
# CHECK: [[@LINE-1]]:18: error: bad arch string foo

.attribute arch, "rv32i2p0_y2p0"
# CHECK: [[@LINE-1]]:18: error: bad arch string y2p0

.attribute stack_align, "16"
# CHECK: [[@LINE-1]]:25: error: expected numeric constant

.attribute unaligned_access, "0"
# CHECK: [[@LINE-1]]:30: error: expected numeric constant

.attribute priv_spec, "2"
# CHECK: [[@LINE-1]]:23: error: expected numeric constant

.attribute priv_spec_minor, "0"
# CHECK: [[@LINE-1]]:29: error: expected numeric constant

.attribute priv_spec_revision, "0"
# CHECK: [[@LINE-1]]:32: error: expected numeric constant

.attribute arch, 30
# CHECK: [[@LINE-1]]:18: error: expected string constant
