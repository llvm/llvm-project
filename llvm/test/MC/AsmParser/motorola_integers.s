# RUN: llvm-mc -triple i386-unknown-unknown -motorola-integers %s | FileCheck %s

# CHECK: a = 2882400009
a = $aBcDeF09
# CHECK: b = 256
b = $0100
# CHECK: c = 10
c = %01010
# CHECK: d = 1
d = %1
