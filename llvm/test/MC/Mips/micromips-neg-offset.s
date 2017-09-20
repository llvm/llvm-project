# Check decoding beqz instruction with a negative offset

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux -mattr=micromips %s -o - \
# RUN:   | llvm-objdump -d -mattr=micromips - | FileCheck %s

# CHECK: 0:   8f 7e   beqz16  $6, -4

beqz16  $6, -4
