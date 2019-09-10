# RUN: llvm-mc %s -triple=riscv64 -mattr=+a -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+a \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+a < %s \
# RUN:     | llvm-objdump -d -mattr=+a -M no-aliases - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+a < %s \
# RUN:     | llvm-objdump -d -mattr=+a - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

# The below tests for lr.d, sc.d and amo*.d, using `0(reg)` are actually
# implemented using a custom parser, but we test them as if they're aliases.

# CHECK-INST: lr.d a1, (a0)
# CHECK-ALIAS: lr.d a1, (a0)
lr.d a1, 0(a0)

# CHECK-INST: lr.d.aq a1, (a0)
# CHECK-ALIAS: lr.d.aq a1, (a0)
lr.d.aq a1, 0(a0)

# CHECK-INST: lr.d.rl a1, (a0)
# CHECK-ALIAS: lr.d.rl a1, (a0)
lr.d.rl a1, 0(a0)

# CHECK-INST: lr.d.aqrl a1, (a0)
# CHECK-ALIAS: lr.d.aqrl a1, (a0)
lr.d.aqrl a1, 0(a0)

# CHECK-INST: sc.d a2, a1, (a0)
# CHECK-ALIAS: sc.d a2, a1, (a0)
sc.d a2, a1, 0(a0)

# CHECK-INST: sc.d.aq a2, a1, (a0)
# CHECK-ALIAS: sc.d.aq a2, a1, (a0)
sc.d.aq a2, a1, 0(a0)

# CHECK-INST: sc.d.rl a2, a1, (a0)
# CHECK-ALIAS: sc.d.rl a2, a1, (a0)
sc.d.rl a2, a1, 0(a0)

# CHECK-INST: sc.d.aqrl a2, a1, (a0)
# CHECK-ALIAS: sc.d.aqrl a2, a1, (a0)
sc.d.aqrl a2, a1, 0(a0)

# CHECK-INST: amoswap.d a2, a1, (a0)
# CHECK-ALIAS: amoswap.d a2, a1, (a0)
amoswap.d a2, a1, 0(a0)

# CHECK-INST: amoswap.d.aq a2, a1, (a0)
# CHECK-ALIAS: amoswap.d.aq a2, a1, (a0)
amoswap.d.aq a2, a1, 0(a0)

# CHECK-INST: amoswap.d.rl a2, a1, (a0)
# CHECK-ALIAS: amoswap.d.rl a2, a1, (a0)
amoswap.d.rl a2, a1, 0(a0)

# CHECK-INST: amoswap.d.aqrl a2, a1, (a0)
# CHECK-ALIAS: amoswap.d.aqrl a2, a1, (a0)
amoswap.d.aqrl a2, a1, 0(a0)

# CHECK-INST: amoadd.d a2, a1, (a0)
# CHECK-ALIAS: amoadd.d a2, a1, (a0)
amoadd.d a2, a1, 0(a0)

# CHECK-INST: amoadd.d.aq a2, a1, (a0)
# CHECK-ALIAS: amoadd.d.aq a2, a1, (a0)
amoadd.d.aq a2, a1, 0(a0)

# CHECK-INST: amoadd.d.rl a2, a1, (a0)
# CHECK-ALIAS: amoadd.d.rl a2, a1, (a0)
amoadd.d.rl a2, a1, 0(a0)

# CHECK-INST: amoadd.d.aqrl a2, a1, (a0)
# CHECK-ALIAS: amoadd.d.aqrl a2, a1, (a0)
amoadd.d.aqrl a2, a1, 0(a0)

# CHECK-INST: amoxor.d a2, a1, (a0)
# CHECK-ALIAS: amoxor.d a2, a1, (a0)
amoxor.d a2, a1, 0(a0)

# CHECK-INST: amoxor.d.aq a2, a1, (a0)
# CHECK-ALIAS: amoxor.d.aq a2, a1, (a0)
amoxor.d.aq a2, a1, 0(a0)

# CHECK-INST: amoxor.d.rl a2, a1, (a0)
# CHECK-ALIAS: amoxor.d.rl a2, a1, (a0)
amoxor.d.rl a2, a1, 0(a0)

# CHECK-INST: amoxor.d.aqrl a2, a1, (a0)
# CHECK-ALIAS: amoxor.d.aqrl a2, a1, (a0)
amoxor.d.aqrl a2, a1, 0(a0)

# CHECK-INST: amoand.d a2, a1, (a0)
# CHECK-ALIAS: amoand.d a2, a1, (a0)
amoand.d a2, a1, 0(a0)

# CHECK-INST: amoand.d.aq a2, a1, (a0)
# CHECK-ALIAS: amoand.d.aq a2, a1, (a0)
amoand.d.aq a2, a1, 0(a0)

# CHECK-INST: amoand.d.rl a2, a1, (a0)
# CHECK-ALIAS: amoand.d.rl a2, a1, (a0)
amoand.d.rl a2, a1, 0(a0)

# CHECK-INST: amoand.d.aqrl a2, a1, (a0)
# CHECK-ALIAS: amoand.d.aqrl a2, a1, (a0)
amoand.d.aqrl a2, a1, 0(a0)

# CHECK-INST: amoor.d a2, a1, (a0)
# CHECK-ALIAS: amoor.d a2, a1, (a0)
amoor.d a2, a1, 0(a0)

# CHECK-INST: amoor.d.aq a2, a1, (a0)
# CHECK-ALIAS: amoor.d.aq a2, a1, (a0)
amoor.d.aq a2, a1, 0(a0)

# CHECK-INST: amoor.d.rl a2, a1, (a0)
# CHECK-ALIAS: amoor.d.rl a2, a1, (a0)
amoor.d.rl a2, a1, 0(a0)

# CHECK-INST: amoor.d.aqrl a2, a1, (a0)
# CHECK-ALIAS: amoor.d.aqrl a2, a1, (a0)
amoor.d.aqrl a2, a1, 0(a0)

# CHECK-INST: amomin.d a2, a1, (a0)
# CHECK-ALIAS: amomin.d a2, a1, (a0)
amomin.d a2, a1, 0(a0)

# CHECK-INST: amomin.d.aq a2, a1, (a0)
# CHECK-ALIAS: amomin.d.aq a2, a1, (a0)
amomin.d.aq a2, a1, 0(a0)

# CHECK-INST: amomin.d.rl a2, a1, (a0)
# CHECK-ALIAS: amomin.d.rl a2, a1, (a0)
amomin.d.rl a2, a1, 0(a0)

# CHECK-INST: amomin.d.aqrl a2, a1, (a0)
# CHECK-ALIAS: amomin.d.aqrl a2, a1, (a0)
amomin.d.aqrl a2, a1, 0(a0)

# CHECK-INST: amomax.d a2, a1, (a0)
# CHECK-ALIAS: amomax.d a2, a1, (a0)
amomax.d a2, a1, 0(a0)

# CHECK-INST: amomax.d.aq a2, a1, (a0)
# CHECK-ALIAS: amomax.d.aq a2, a1, (a0)
amomax.d.aq a2, a1, 0(a0)

# CHECK-INST: amomax.d.rl a2, a1, (a0)
# CHECK-ALIAS: amomax.d.rl a2, a1, (a0)
amomax.d.rl a2, a1, 0(a0)

# CHECK-INST: amomax.d.aqrl a2, a1, (a0)
# CHECK-ALIAS: amomax.d.aqrl a2, a1, (a0)
amomax.d.aqrl a2, a1, 0(a0)

# CHECK-INST: amominu.d a2, a1, (a0)
# CHECK-ALIAS: amominu.d a2, a1, (a0)
amominu.d a2, a1, 0(a0)

# CHECK-INST: amominu.d.aq a2, a1, (a0)
# CHECK-ALIAS: amominu.d.aq a2, a1, (a0)
amominu.d.aq a2, a1, 0(a0)

# CHECK-INST: amominu.d.rl a2, a1, (a0)
# CHECK-ALIAS: amominu.d.rl a2, a1, (a0)
amominu.d.rl a2, a1, 0(a0)

# CHECK-INST: amominu.d.aqrl a2, a1, (a0)
# CHECK-ALIAS: amominu.d.aqrl a2, a1, (a0)
amominu.d.aqrl a2, a1, 0(a0)

# CHECK-INST: amomaxu.d a2, a1, (a0)
# CHECK-ALIAS: amomaxu.d a2, a1, (a0)
amomaxu.d a2, a1, 0(a0)

# CHECK-INST: amomaxu.d.aq a2, a1, (a0)
# CHECK-ALIAS: amomaxu.d.aq a2, a1, (a0)
amomaxu.d.aq a2, a1, 0(a0)

# CHECK-INST: amomaxu.d.rl a2, a1, (a0)
# CHECK-ALIAS: amomaxu.d.rl a2, a1, (a0)
amomaxu.d.rl a2, a1, 0(a0)

# CHECK-INST: amomaxu.d.aqrl a2, a1, (a0)
# CHECK-ALIAS: amomaxu.d.aqrl a2, a1, (a0)
amomaxu.d.aqrl a2, a1, 0(a0)
