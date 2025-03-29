## A snippet from https://github.com/riscv/riscv-v-spec.

# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v < %s \
# RUN:   | llvm-objdump -d --mattr=+v - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST

loop:
    vsetvli a3, a0, e16,m4,ta,ma  # vtype = 16-bit integer vectors
# CHECK-INST: 0ca576d7 vsetvli a3, a0, e16, m4, ta, ma
    vle16.v v4, (a1)              # Get 16b vector
# CHECK-INST: 0205d207 vle16.v v4, (a1)
    slli t1, a3, 1                # Multiply length by two bytes/element
# CHECK-INST: 00169313 slli t1, a3, 0x1
    add a1, a1, t1                # Bump pointer
# CHECK-INST: 006585b3 add a1, a1, t1
    vwmul.vx v8, v4, x10          # 32b in <v8--v15>
# CHECK-INST: ee456457 vwmul.vx v8, v4, a0

    vsetvli x0, a0, e32,m8,ta,ma  # Operate on 32b values
# CHECK-INST: 0d357057 vsetvli zero, a0, e32, m8, ta, ma
    vsrl.vi v8, v8, 3
# CHECK-INST: a281b457 vsrl.vi v8, v8, 0x3
    vse32.v v8, (a2)              # Store vector of 32b
# CHECK-INST: 02066427 vse32.v v8, (a2)
    slli t1, a3, 2                # Multiply length by four bytes/element
# CHECK-INST: 00269313 slli t1, a3, 0x2
    add a2, a2, t1                # Bump pointer
# CHECK-INST: 00660633 add a2, a2, t1
    sub a0, a0, a3                # Decrement count
# CHECK-INST: 40d50533 sub a0, a0, a3
    bnez a0, loop                 # Any more?
# CHECK-INST: fc051ae3 bnez a0, 0x0
