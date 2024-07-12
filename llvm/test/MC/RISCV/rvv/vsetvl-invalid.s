# RUN: llvm-mc -filetype=obj -triple=riscv32 %s \
# RUN:     | llvm-objdump -d --mattr=+v - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s \
# RUN:     | llvm-objdump -d --mattr=+v - | FileCheck %s

# CHECK: vsetvli a1, a0, e64, m1, tu, mu
.word 0x018575d7

# CHECK: vsetvli a1, a0, 0x1c
.word 0x01c575d7

# CHECK: vsetvli a1, a0, 0x24
.word 0x024575d7

# CHECK: vsetvli a1, a0, 0x29
.word 0x029575d7

# CHECK: vsetvli a1, a0, 0x110
.word 0x110575d7

# CHECK: vsetvli a1, a0, e64, mf8, tu, mu
.word 0x01d575d7

# CHECK: vsetivli a1, 0x10, e8, m4, tu, mu
.word 0xc02875d7

# CHECK: vsetivli a1, 0x10, 0xc
.word 0xc0c875d7

# CHECK: vsetivli a1, 0x10, 0x14
.word 0xc14875d7

# CHECK: vsetivli a1, 0x10, 0x38
.word 0xc38875d7

# CHECK: vsetivli a1, 0x10, 0x103
.word 0xd03875d7

# CHECK: vsetivli a1, 0x10, e8, mf4, tu, mu
.word 0xc06875d7
