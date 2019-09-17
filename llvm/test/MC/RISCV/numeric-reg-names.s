# RUN: llvm-mc -triple riscv32 < %s -riscv-arch-reg-names \
# RUN:     | FileCheck -check-prefix=CHECK-NUMERIC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d -M numeric - \
# RUN:     | FileCheck -check-prefix=CHECK-NUMERIC %s

# CHECK-NUMERIC: addi x10, x0, 1
# CHECK-NUMERIC-NEXT: addi x10, x0, 1
addi a0, x0, 1
addi a0, zero, 1

# CHECK-NUMERIC: addi x10, x1, 1
# CHECK-NUMERIC-NEXT: addi x10, x1, 1
addi a0, x1, 1
addi a0, ra, 1

# CHECK-NUMERIC: addi x10, x2, 1
# CHECK-NUMERIC-NEXT: addi x10, x2, 1
addi a0, x2, 1
addi a0, sp, 1

# CHECK-NUMERIC: addi x10, x3, 1
# CHECK-NUMERIC-NEXT: addi x10, x3, 1
addi a0, x3, 1
addi a0, gp, 1

# CHECK-NUMERIC: addi x10, x4, 1
# CHECK-NUMERIC-NEXT: addi x10, x4, 1
addi a0, x4, 1
addi a0, tp, 1

# CHECK-NUMERIC: addi x10, x5, 1
# CHECK-NUMERIC-NEXT: addi x10, x5, 1
addi a0, x5, 1
addi a0, t0, 1

# CHECK-NUMERIC: addi x10, x6, 1
# CHECK-NUMERIC-NEXT: addi x10, x6, 1
addi a0, x6, 1
addi a0, t1, 1

# CHECK-NUMERIC: addi x10, x7, 1
# CHECK-NUMERIC-NEXT: addi x10, x7, 1
addi a0, x7, 1
addi a0, t2, 1

# CHECK-NUMERIC: addi x10, x8, 1
# CHECK-NUMERIC-NEXT: addi x10, x8, 1
# CHECK-NUMERIC-NEXT: addi x10, x8, 1
addi a0, x8, 1
addi a0, s0, 1
addi a0, fp, 1

# CHECK-NUMERIC: addi x10, x9, 1
# CHECK-NUMERIC-NEXT: addi x10, x9, 1
addi a0, x9, 1
addi a0, s1, 1

# CHECK-NUMERIC: addi x10, x10, 1
# CHECK-NUMERIC-NEXT: addi x10, x10, 1
addi a0, x10, 1
addi a0, a0, 1

# CHECK-NUMERIC: addi x10, x11, 1
# CHECK-NUMERIC-NEXT: addi x10, x11, 1
addi a0, x11, 1
addi a0, a1, 1

# CHECK-NUMERIC: addi x10, x12, 1
# CHECK-NUMERIC-NEXT: addi x10, x12, 1
addi a0, x12, 1
addi a0, a2, 1

# CHECK-NUMERIC: addi x10, x13, 1
# CHECK-NUMERIC-NEXT: addi x10, x13, 1
addi a0, x13, 1
addi a0, a3, 1

# CHECK-NUMERIC: addi x10, x14, 1
# CHECK-NUMERIC-NEXT: addi x10, x14, 1
addi a0, x14, 1
addi a0, a4, 1

# CHECK-NUMERIC: addi x10, x15, 1
# CHECK-NUMERIC-NEXT: addi x10, x15, 1
addi a0, x15, 1
addi a0, a5, 1

# CHECK-NUMERIC: addi x10, x16, 1
# CHECK-NUMERIC-NEXT: addi x10, x16, 1
addi a0, x16, 1
addi a0, a6, 1

# CHECK-NUMERIC: addi x10, x17, 1
# CHECK-NUMERIC-NEXT: addi x10, x17, 1
addi a0, x17, 1
addi a0, a7, 1

# CHECK-NUMERIC: addi x10, x18, 1
# CHECK-NUMERIC-NEXT: addi x10, x18, 1
addi a0, x18, 1
addi a0, s2, 1

# CHECK-NUMERIC: addi x10, x19, 1
# CHECK-NUMERIC-NEXT: addi x10, x19, 1
addi a0, x19, 1
addi a0, s3, 1

# CHECK-NUMERIC: addi x10, x20, 1
# CHECK-NUMERIC-NEXT: addi x10, x20, 1
addi a0, x20, 1
addi a0, s4, 1

# CHECK-NUMERIC: addi x10, x21, 1
# CHECK-NUMERIC-NEXT: addi x10, x21, 1
addi a0, x21, 1
addi a0, s5, 1

# CHECK-NUMERIC: addi x10, x22, 1
# CHECK-NUMERIC-NEXT: addi x10, x22, 1
addi a0, x22, 1
addi a0, s6, 1

# CHECK-NUMERIC: addi x10, x23, 1
# CHECK-NUMERIC-NEXT: addi x10, x23, 1
addi a0, x23, 1
addi a0, s7, 1

# CHECK-NUMERIC: addi x10, x24, 1
# CHECK-NUMERIC-NEXT: addi x10, x24, 1
addi a0, x24, 1
addi a0, s8, 1

# CHECK-NUMERIC: addi x10, x25, 1
# CHECK-NUMERIC-NEXT: addi x10, x25, 1
addi a0, x25, 1
addi a0, s9, 1

# CHECK-NUMERIC: addi x10, x26, 1
# CHECK-NUMERIC-NEXT: addi x10, x26, 1
addi a0, x26, 1
addi a0, s10, 1

# CHECK-NUMERIC: addi x10, x27, 1
# CHECK-NUMERIC-NEXT: addi x10, x27, 1
addi a0, x27, 1
addi a0, s11, 1

# CHECK-NUMERIC: addi x10, x28, 1
# CHECK-NUMERIC-NEXT: addi x10, x28, 1
addi a0, x28, 1
addi a0, t3, 1

# CHECK-NUMERIC: addi x10, x29, 1
# CHECK-NUMERIC-NEXT: addi x10, x29, 1
addi a0, x29, 1
addi a0, t4, 1

# CHECK-NUMERIC: addi x10, x30, 1
# CHECK-NUMERIC-NEXT: addi x10, x30, 1
addi a0, x30, 1
addi a0, t5, 1

# CHECK-NUMERIC: addi x10, x31, 1
# CHECK-NUMERIC-NEXT: addi x10, x31, 1
addi a0, x31, 1
addi a0, t6, 1
