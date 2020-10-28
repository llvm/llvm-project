## A snippet from https://github.com/riscv/riscv-v-spec.

# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v < %s \
# RUN: | llvm-objdump -d --mattr=+experimental-v - \
# RUN: | FileCheck %s --check-prefix=CHECK-INST

loop:
    vsetvli a3, a0, e16,m4,ta,ma  # vtype = 16-bit integer vectors
# CHECK-INST: d7 76 65 0c    vsetvli a3, a0, e16,m4,ta,ma
    vle16.v v4, (a1)              # Get 16b vector
# CHECK-INST: 07 d2 05 02    vle16.v   v4, (a1)
    slli t1, a3, 1                # Multiply length by two bytes/element
# CHECK-INST: 13 93 16 00    slli    t1, a3, 1
    add a1, a1, t1                # Bump pointer
# CHECK-INST: b3 85 65 00    add     a1, a1, t1
    vwmul.vx v8, v4, x10          # 32b in <v8--v15>
# CHECK-INST: 57 64 45 ee    vwmul.vx        v8, v4, a0

    vsetvli x0, a0, e32,m8,ta,ma  # Operate on 32b values
# CHECK-INST: 57 70 b5 0c    vsetvli zero, a0, e32,m8,ta,ma
    vsrl.vi v8, v8, 3
# CHECK-INST: 57 b4 81 a2    vsrl.vi v8, v8, 3
    vse32.v v8, (a2)              # Store vector of 32b
# CHECK-INST: 27 64 06 02    vse32.v   v8, (a2)
    slli t1, a3, 2                # Multiply length by four bytes/element
# CHECK-INST: 13 93 26 00    slli    t1, a3, 2
    add a2, a2, t1                # Bump pointer
# CHECK-INST: 33 06 66 00    add     a2, a2, t1
    sub a0, a0, a3                # Decrement count
# CHECK-INST: 33 05 d5 40    sub     a0, a0, a3
    bnez a0, loop                 # Any more?
# CHECK-INST: e3 1a 05 fc    bnez    a0, -44
