# RUN: llvm-mc -triple i386-unknown-unknown -motorola-integers %s | FileCheck %s

# CHECK: .set a, 2882400009
.set a, $aBcDeF09
# CHECK: .set b, 256
.set b, $0100
# CHECK: .set c, 10
.set c, %01010
# CHECK: .set d, 1
.set d, %1
