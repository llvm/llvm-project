# RUN: llvm-mc -triple hexagon -filetype=obj %s | llvm-objdump -d - | FileCheck %s

# CHECK: 11 df 75 f1
r17 = xor(r21, lr)

# CHECK: 1d df 35 f3
sp = sub(lr, r21)

# CHECK: 15 c0 3e 71
fp.l = #21
