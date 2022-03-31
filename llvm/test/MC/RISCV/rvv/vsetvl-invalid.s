# RUN: llvm-mc -filetype=obj -triple=riscv32 %s \
# RUN:     | llvm-objdump -d --mattr=+v - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s \
# RUN:     | llvm-objdump -d --mattr=+v - | FileCheck %s

# CHECK: vsetvli a1, a0, e64, m1, tu, mu
.word 0x018575d7

# CHECK: vsetvli a1, a0, 28
.word 0x01c575d7

# CHECK: vsetvli a1, a0, 36
.word 0x024575d7

# CHECK: vsetvli a1, a0, 41
.word 0x029575d7

# CHECK: vsetvli a1, a0, 272
.word 0x110575d7

# CHECK: vsetvli a1, a0, e64, mf8, tu, mu
.word 0x01d575d7

# CHECK: vsetivli a1, 16, e8, m4, tu, mu
.word 0xc02875d7

# CHECK: vsetivli a1, 16, 12
.word 0xc0c875d7

# CHECK: vsetivli a1, 16, 20
.word 0xc14875d7

# CHECK: vsetivli a1, 16, 56
.word 0xc38875d7

# CHECK: vsetivli a1, 16, 259
.word 0xd03875d7

# CHECK: vsetivli a1, 16, e8, mf4, tu, mu
.word 0xc06875d7
