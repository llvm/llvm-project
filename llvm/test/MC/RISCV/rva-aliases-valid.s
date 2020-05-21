# RUN: llvm-mc %s -triple=riscv32 -mattr=+a -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+a \
# RUN:     | FileCheck -check-prefixes=CHECK-S,CHECK-S-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+a -riscv-no-aliases\
# RUN:     | FileCheck -check-prefixes=CHECK-S-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+a \
# RUN:     | FileCheck -check-prefixes=CHECK-S,CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+a < %s \
# RUN:     | llvm-objdump -d --mattr=+a -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+a < %s \
# RUN:     | llvm-objdump -d --mattr=+a - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+a < %s \
# RUN:     | llvm-objdump -d --mattr=+a -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+a < %s \
# RUN:     | llvm-objdump -d --mattr=+a - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-S-OBJ %s

# COM: The following check prefixes are used in this test:
# COM: CHECK-S                 Match the .s output with aliases enabled
# COM: CHECK-S-NOALIAS         Match the .s output with aliases disabled
# COM: CHECK-OBJ               Match the objdumped object output with aliases enabled
# COM: CHECK-OBJ-NOALIAS       Match the objdumped object output with aliases enabled
# COM: CHECK-S-OBJ             Match both the .s and objdumped object output with
# COM:                         aliases enabled
# COM: CHECK-S-OBJ-NOALIAS     Match both the .s and objdumped object output with
# COM:                         aliases disabled

# The below tests for lr.w, sc.w and amo*.w, using `0(reg)` are actually
# implemented using a custom parser, but we test them as if they're aliases.

# CHECK-S: lr.w a1, (a0)
# CHECK-S-NOALIAS: lr.w a1, (a0)
# CHECK-OBJ: lr.w a1, (a0)
# CHECK-OBJ-NOALIAS: lr.w a1, (a0)
lr.w a1, 0(a0)

# CHECK-S: lr.w.aq a1, (a0)
# CHECK-S-NOALIAS: lr.w.aq a1, (a0)
# CHECK-OBJ: lr.w.aq a1, (a0)
# CHECK-OBJ-NOALIAS: lr.w.aq a1, (a0)
lr.w.aq a1, 0(a0)

# CHECK-S: lr.w.rl a1, (a0)
# CHECK-S-NOALIAS: lr.w.rl a1, (a0)
# CHECK-OBJ: lr.w.rl a1, (a0)
# CHECK-OBJ-NOALIAS: lr.w.rl a1, (a0)
lr.w.rl a1, 0(a0)

# CHECK-S: lr.w.aqrl a1, (a0)
# CHECK-S-NOALIAS: lr.w.aqrl a1, (a0)
# CHECK-OBJ: lr.w.aqrl a1, (a0)
# CHECK-OBJ-NOALIAS: lr.w.aqrl a1, (a0)
lr.w.aqrl a1, 0(a0)

# CHECK-S: sc.w a2, a1, (a0)
# CHECK-S-NOALIAS: sc.w a2, a1, (a0)
# CHECK-OBJ: sc.w a2, a1, (a0)
# CHECK-OBJ-NOALIAS: sc.w a2, a1, (a0)
sc.w a2, a1, 0(a0)

# CHECK-S: sc.w.aq a2, a1, (a0)
# CHECK-S-NOALIAS: sc.w.aq a2, a1, (a0)
# CHECK-OBJ: sc.w.aq a2, a1, (a0)
# CHECK-OBJ-NOALIAS: sc.w.aq a2, a1, (a0)
sc.w.aq a2, a1, 0(a0)

# CHECK-S: sc.w.rl a2, a1, (a0)
# CHECK-S-NOALIAS: sc.w.rl a2, a1, (a0)
# CHECK-OBJ: sc.w.rl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: sc.w.rl a2, a1, (a0)
sc.w.rl a2, a1, 0(a0)

# CHECK-S: sc.w.aqrl a2, a1, (a0)
# CHECK-S-NOALIAS: sc.w.aqrl a2, a1, (a0)
# CHECK-OBJ: sc.w.aqrl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: sc.w.aqrl a2, a1, (a0)
sc.w.aqrl a2, a1, 0(a0)

# CHECK-S: amoswap.w a2, a1, (a0)
# CHECK-S-NOALIAS: amoswap.w a2, a1, (a0)
# CHECK-OBJ: amoswap.w a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoswap.w a2, a1, (a0)
amoswap.w a2, a1, 0(a0)

# CHECK-S: amoswap.w.aq a2, a1, (a0)
# CHECK-S-NOALIAS: amoswap.w.aq a2, a1, (a0)
# CHECK-OBJ: amoswap.w.aq a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoswap.w.aq a2, a1, (a0)
amoswap.w.aq a2, a1, 0(a0)

# CHECK-S: amoswap.w.rl a2, a1, (a0)
# CHECK-S-NOALIAS: amoswap.w.rl a2, a1, (a0)
# CHECK-OBJ: amoswap.w.rl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoswap.w.rl a2, a1, (a0)
amoswap.w.rl a2, a1, 0(a0)

# CHECK-S: amoswap.w.aqrl a2, a1, (a0)
# CHECK-S-NOALIAS: amoswap.w.aqrl a2, a1, (a0)
# CHECK-OBJ: amoswap.w.aqrl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoswap.w.aqrl a2, a1, (a0)
amoswap.w.aqrl a2, a1, 0(a0)

# CHECK-S: amoadd.w a2, a1, (a0)
# CHECK-S-NOALIAS: amoadd.w a2, a1, (a0)
# CHECK-OBJ: amoadd.w a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoadd.w a2, a1, (a0)
amoadd.w a2, a1, 0(a0)

# CHECK-S: amoadd.w.aq a2, a1, (a0)
# CHECK-S-NOALIAS: amoadd.w.aq a2, a1, (a0)
# CHECK-OBJ: amoadd.w.aq a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoadd.w.aq a2, a1, (a0)
amoadd.w.aq a2, a1, 0(a0)

# CHECK-S: amoadd.w.rl a2, a1, (a0)
# CHECK-S-NOALIAS: amoadd.w.rl a2, a1, (a0)
# CHECK-OBJ: amoadd.w.rl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoadd.w.rl a2, a1, (a0)
amoadd.w.rl a2, a1, 0(a0)

# CHECK-S: amoadd.w.aqrl a2, a1, (a0)
# CHECK-S-NOALIAS: amoadd.w.aqrl a2, a1, (a0)
# CHECK-OBJ: amoadd.w.aqrl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoadd.w.aqrl a2, a1, (a0)
amoadd.w.aqrl a2, a1, 0(a0)

# CHECK-S: amoxor.w a2, a1, (a0)
# CHECK-S-NOALIAS: amoxor.w a2, a1, (a0)
# CHECK-OBJ: amoxor.w a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoxor.w a2, a1, (a0)
amoxor.w a2, a1, 0(a0)

# CHECK-S: amoxor.w.aq a2, a1, (a0)
# CHECK-S-NOALIAS: amoxor.w.aq a2, a1, (a0)
# CHECK-OBJ: amoxor.w.aq a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoxor.w.aq a2, a1, (a0)
amoxor.w.aq a2, a1, 0(a0)

# CHECK-S: amoxor.w.rl a2, a1, (a0)
# CHECK-S-NOALIAS: amoxor.w.rl a2, a1, (a0)
# CHECK-OBJ: amoxor.w.rl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoxor.w.rl a2, a1, (a0)
amoxor.w.rl a2, a1, 0(a0)

# CHECK-S: amoxor.w.aqrl a2, a1, (a0)
# CHECK-S-NOALIAS: amoxor.w.aqrl a2, a1, (a0)
# CHECK-OBJ: amoxor.w.aqrl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoxor.w.aqrl a2, a1, (a0)
amoxor.w.aqrl a2, a1, 0(a0)

# CHECK-S: amoand.w a2, a1, (a0)
# CHECK-S-NOALIAS: amoand.w a2, a1, (a0)
# CHECK-OBJ: amoand.w a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoand.w a2, a1, (a0)
amoand.w a2, a1, 0(a0)

# CHECK-S: amoand.w.aq a2, a1, (a0)
# CHECK-S-NOALIAS: amoand.w.aq a2, a1, (a0)
# CHECK-OBJ: amoand.w.aq a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoand.w.aq a2, a1, (a0)
amoand.w.aq a2, a1, 0(a0)

# CHECK-S: amoand.w.rl a2, a1, (a0)
# CHECK-S-NOALIAS: amoand.w.rl a2, a1, (a0)
# CHECK-OBJ: amoand.w.rl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoand.w.rl a2, a1, (a0)
amoand.w.rl a2, a1, 0(a0)

# CHECK-S: amoand.w.aqrl a2, a1, (a0)
# CHECK-S-NOALIAS: amoand.w.aqrl a2, a1, (a0)
# CHECK-OBJ: amoand.w.aqrl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoand.w.aqrl a2, a1, (a0)
amoand.w.aqrl a2, a1, 0(a0)

# CHECK-S: amoor.w a2, a1, (a0)
# CHECK-S-NOALIAS: amoor.w a2, a1, (a0)
# CHECK-OBJ: amoor.w a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoor.w a2, a1, (a0)
amoor.w a2, a1, 0(a0)

# CHECK-S: amoor.w.aq a2, a1, (a0)
# CHECK-S-NOALIAS: amoor.w.aq a2, a1, (a0)
# CHECK-OBJ: amoor.w.aq a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoor.w.aq a2, a1, (a0)
amoor.w.aq a2, a1, 0(a0)

# CHECK-S: amoor.w.rl a2, a1, (a0)
# CHECK-S-NOALIAS: amoor.w.rl a2, a1, (a0)
# CHECK-OBJ: amoor.w.rl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoor.w.rl a2, a1, (a0)
amoor.w.rl a2, a1, 0(a0)

# CHECK-S: amoor.w.aqrl a2, a1, (a0)
# CHECK-S-NOALIAS: amoor.w.aqrl a2, a1, (a0)
# CHECK-OBJ: amoor.w.aqrl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amoor.w.aqrl a2, a1, (a0)
amoor.w.aqrl a2, a1, 0(a0)

# CHECK-S: amomin.w a2, a1, (a0)
# CHECK-S-NOALIAS: amomin.w a2, a1, (a0)
# CHECK-OBJ: amomin.w a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amomin.w a2, a1, (a0)
amomin.w a2, a1, 0(a0)

# CHECK-S: amomin.w.aq a2, a1, (a0)
# CHECK-S-NOALIAS: amomin.w.aq a2, a1, (a0)
# CHECK-OBJ: amomin.w.aq a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amomin.w.aq a2, a1, (a0)
amomin.w.aq a2, a1, 0(a0)

# CHECK-S: amomin.w.rl a2, a1, (a0)
# CHECK-S-NOALIAS: amomin.w.rl a2, a1, (a0)
# CHECK-OBJ: amomin.w.rl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amomin.w.rl a2, a1, (a0)
amomin.w.rl a2, a1, 0(a0)

# CHECK-S: amomin.w.aqrl a2, a1, (a0)
# CHECK-S-NOALIAS: amomin.w.aqrl a2, a1, (a0)
# CHECK-OBJ: amomin.w.aqrl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amomin.w.aqrl a2, a1, (a0)
amomin.w.aqrl a2, a1, 0(a0)

# CHECK-S: amomax.w a2, a1, (a0)
# CHECK-S-NOALIAS: amomax.w a2, a1, (a0)
# CHECK-OBJ: amomax.w a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amomax.w a2, a1, (a0)
amomax.w a2, a1, 0(a0)

# CHECK-S: amomax.w.aq a2, a1, (a0)
# CHECK-S-NOALIAS: amomax.w.aq a2, a1, (a0)
# CHECK-OBJ: amomax.w.aq a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amomax.w.aq a2, a1, (a0)
amomax.w.aq a2, a1, 0(a0)

# CHECK-S: amomax.w.rl a2, a1, (a0)
# CHECK-S-NOALIAS: amomax.w.rl a2, a1, (a0)
# CHECK-OBJ: amomax.w.rl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amomax.w.rl a2, a1, (a0)
amomax.w.rl a2, a1, 0(a0)

# CHECK-S: amomax.w.aqrl a2, a1, (a0)
# CHECK-S-NOALIAS: amomax.w.aqrl a2, a1, (a0)
# CHECK-OBJ: amomax.w.aqrl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amomax.w.aqrl a2, a1, (a0)
amomax.w.aqrl a2, a1, 0(a0)

# CHECK-S: amominu.w a2, a1, (a0)
# CHECK-S-NOALIAS: amominu.w a2, a1, (a0)
# CHECK-OBJ: amominu.w a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amominu.w a2, a1, (a0)
amominu.w a2, a1, 0(a0)

# CHECK-S: amominu.w.aq a2, a1, (a0)
# CHECK-S-NOALIAS: amominu.w.aq a2, a1, (a0)
# CHECK-OBJ: amominu.w.aq a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amominu.w.aq a2, a1, (a0)
amominu.w.aq a2, a1, 0(a0)

# CHECK-S: amominu.w.rl a2, a1, (a0)
# CHECK-S-NOALIAS: amominu.w.rl a2, a1, (a0)
# CHECK-OBJ: amominu.w.rl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amominu.w.rl a2, a1, (a0)
amominu.w.rl a2, a1, 0(a0)

# CHECK-S: amominu.w.aqrl a2, a1, (a0)
# CHECK-S-NOALIAS: amominu.w.aqrl a2, a1, (a0)
# CHECK-OBJ: amominu.w.aqrl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amominu.w.aqrl a2, a1, (a0)
amominu.w.aqrl a2, a1, 0(a0)

# CHECK-S: amomaxu.w a2, a1, (a0)
# CHECK-S-NOALIAS: amomaxu.w a2, a1, (a0)
# CHECK-OBJ: amomaxu.w a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amomaxu.w a2, a1, (a0)
amomaxu.w a2, a1, 0(a0)

# CHECK-S: amomaxu.w.aq a2, a1, (a0)
# CHECK-S-NOALIAS: amomaxu.w.aq a2, a1, (a0)
# CHECK-OBJ: amomaxu.w.aq a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amomaxu.w.aq a2, a1, (a0)
amomaxu.w.aq a2, a1, 0(a0)

# CHECK-S: amomaxu.w.rl a2, a1, (a0)
# CHECK-S-NOALIAS: amomaxu.w.rl a2, a1, (a0)
# CHECK-OBJ: amomaxu.w.rl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amomaxu.w.rl a2, a1, (a0)
amomaxu.w.rl a2, a1, 0(a0)

# CHECK-S: amomaxu.w.aqrl a2, a1, (a0)
# CHECK-S-NOALIAS: amomaxu.w.aqrl a2, a1, (a0)
# CHECK-OBJ: amomaxu.w.aqrl a2, a1, (a0)
# CHECK-OBJ-NOALIAS: amomaxu.w.aqrl a2, a1, (a0)
amomaxu.w.aqrl a2, a1, 0(a0)
