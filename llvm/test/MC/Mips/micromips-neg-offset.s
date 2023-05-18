# Check decoding beqz instruction with a negative offset

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips -mcpu=mips32r6 %s -o - \
# RUN:   | llvm-objdump -d --mattr=micromips - | FileCheck %s

# CHECK: 0:   8f 7e        beqzc16  $6, 0xfffffffc <.text+0xfffffffffffffffc>
# CHECK: 2:   cf fe        bc16     0xfffffffe <.text+0xfffffffffffffffe>
# CHECK: 4:   b7 ff ff fe  balc     0x0 <.text>

beqz16  $6, -4
b16     -4
balc    -4
