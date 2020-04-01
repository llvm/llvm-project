## Test .attribute effects.
## We do not provide '-mattr=' and '.option rvc' and enable extensions through
## '.attribute arch'.

# RUN: llvm-mc -triple riscv32 -filetype=obj %s \
# RUN:   | llvm-objdump --triple=riscv32 -d -M no-aliases - \
# RUN:   | FileCheck -check-prefix=CHECK-INST %s

# RUN: llvm-mc -triple riscv64 -filetype=obj %s \
# RUN:   | llvm-objdump --triple=riscv64 -d -M no-aliases - \
# RUN:   | FileCheck -check-prefix=CHECK-INST %s

.attribute arch, "rv64i2p0_m2p0_a2p0_d2p0_c2p0"

# CHECK-INST: lr.w t0, (t1)
lr.w t0, (t1)

# CHECK-INST: c.addi a3, -32
c.addi a3, -32

# CHECK-INST: fmadd.d fa0, fa1, fa2, fa3, dyn
fmadd.d f10, f11, f12, f13, dyn

# CHECK-INST: fmadd.s fa0, fa1, fa2, fa3, dyn
fmadd.s f10, f11, f12, f13, dyn

# CHECK-INST: addi ra, sp, 2
addi ra, sp, 2

# CHECK-INST: mul a4, ra, s0
mul a4, ra, s0

# CHECK-INST: addw a2, a3, a4
addw a2, a3, a4
