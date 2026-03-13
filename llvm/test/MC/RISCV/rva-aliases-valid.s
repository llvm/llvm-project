# RUN: llvm-mc %s -triple=riscv32 -mattr=+a -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+a \
# RUN:     | FileCheck -check-prefixes=CHECK-S,CHECK-S-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+a -M no-aliases\
# RUN:     | FileCheck -check-prefixes=CHECK-S-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+a \
# RUN:     | FileCheck -check-prefixes=CHECK-S,CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+a < %s \
# RUN:     | llvm-objdump -d --mattr=+a -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+a < %s \
# RUN:     | llvm-objdump -d --mattr=+a - \
# RUN:     | FileCheck --check-prefix=CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+a < %s \
# RUN:     | llvm-objdump -d --mattr=+a -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-NOALIAS,CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+a < %s \
# RUN:     | llvm-objdump -d --mattr=+a - \
# RUN:     | FileCheck --check-prefix=CHECK-S-OBJ %s

# The following check prefixes are used in this test:
# CHECK-S                 Match the .s output with aliases enabled
# CHECK-S-NOALIAS         Match the .s output with aliases disabled
# CHECK-OBJ               Match the objdumped object output with aliases enabled
# CHECK-OBJ-NOALIAS       Match the objdumped object output with aliases enabled
# CHECK-S-OBJ             Match both the .s and objdumped object output with
#                         aliases enabled
# CHECK-S-OBJ-NOALIAS     Match both the .s and objdumped object output with
#                         aliases disabled

# The below tests for lr.w, sc.w and amo*.w, using `0(reg)` are actually
# implemented using a custom parser, but we test them as if they're aliases.
# CHECK-S: {{^}}
# CHECK-S-NOALIAS: {{^}}
# CHECK-OBJ-NOALIAS: {{^}}

# CHECK-S-OBJ: lr.w a1, (a0)
# CHECK-S-OBJ-NOALIAS: lr.w a1, (a0)
lr.w a1, 0(a0)

# CHECK-S-OBJ: lr.w.aq a1, (a0)
# CHECK-S-OBJ-NOALIAS: lr.w.aq a1, (a0)
lr.w.aq a1, 0(a0)

# CHECK-S-OBJ: lr.w.rl a1, (a0)
# CHECK-S-OBJ-NOALIAS: lr.w.rl a1, (a0)
lr.w.rl a1, 0(a0)

# CHECK-S-OBJ: lr.w.aqrl a1, (a0)
# CHECK-S-OBJ-NOALIAS: lr.w.aqrl a1, (a0)
lr.w.aqrl a1, 0(a0)

# CHECK-S-OBJ: sc.w a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: sc.w a2, a1, (a0)
sc.w a2, a1, 0(a0)

# CHECK-S-OBJ: sc.w.aq a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: sc.w.aq a2, a1, (a0)
sc.w.aq a2, a1, 0(a0)

# CHECK-S-OBJ: sc.w.rl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: sc.w.rl a2, a1, (a0)
sc.w.rl a2, a1, 0(a0)

# CHECK-S-OBJ: sc.w.aqrl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: sc.w.aqrl a2, a1, (a0)
sc.w.aqrl a2, a1, 0(a0)

# CHECK-S-OBJ: amoswap.w a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoswap.w a2, a1, (a0)
amoswap.w a2, a1, 0(a0)

# CHECK-S-OBJ: amoswap.w.aq a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoswap.w.aq a2, a1, (a0)
amoswap.w.aq a2, a1, 0(a0)

# CHECK-S-OBJ: amoswap.w.rl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoswap.w.rl a2, a1, (a0)
amoswap.w.rl a2, a1, 0(a0)

# CHECK-S-OBJ: amoswap.w.aqrl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoswap.w.aqrl a2, a1, (a0)
amoswap.w.aqrl a2, a1, 0(a0)

# CHECK-S-OBJ: amoadd.w a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoadd.w a2, a1, (a0)
amoadd.w a2, a1, 0(a0)

# CHECK-S-OBJ: amoadd.w.aq a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoadd.w.aq a2, a1, (a0)
amoadd.w.aq a2, a1, 0(a0)

# CHECK-S-OBJ: amoadd.w.rl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoadd.w.rl a2, a1, (a0)
amoadd.w.rl a2, a1, 0(a0)

# CHECK-S-OBJ: amoadd.w.aqrl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoadd.w.aqrl a2, a1, (a0)
amoadd.w.aqrl a2, a1, 0(a0)

# CHECK-S-OBJ: amoxor.w a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoxor.w a2, a1, (a0)
amoxor.w a2, a1, 0(a0)

# CHECK-S-OBJ: amoxor.w.aq a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoxor.w.aq a2, a1, (a0)
amoxor.w.aq a2, a1, 0(a0)

# CHECK-S-OBJ: amoxor.w.rl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoxor.w.rl a2, a1, (a0)
amoxor.w.rl a2, a1, 0(a0)

# CHECK-S-OBJ: amoxor.w.aqrl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoxor.w.aqrl a2, a1, (a0)
amoxor.w.aqrl a2, a1, 0(a0)

# CHECK-S-OBJ: amoand.w a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoand.w a2, a1, (a0)
amoand.w a2, a1, 0(a0)

# CHECK-S-OBJ: amoand.w.aq a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoand.w.aq a2, a1, (a0)
amoand.w.aq a2, a1, 0(a0)

# CHECK-S-OBJ: amoand.w.rl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoand.w.rl a2, a1, (a0)
amoand.w.rl a2, a1, 0(a0)

# CHECK-S-OBJ: amoand.w.aqrl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoand.w.aqrl a2, a1, (a0)
amoand.w.aqrl a2, a1, 0(a0)

# CHECK-S-OBJ: amoor.w a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoor.w a2, a1, (a0)
amoor.w a2, a1, 0(a0)

# CHECK-S-OBJ: amoor.w.aq a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoor.w.aq a2, a1, (a0)
amoor.w.aq a2, a1, 0(a0)

# CHECK-S-OBJ: amoor.w.rl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoor.w.rl a2, a1, (a0)
amoor.w.rl a2, a1, 0(a0)

# CHECK-S-OBJ: amoor.w.aqrl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amoor.w.aqrl a2, a1, (a0)
amoor.w.aqrl a2, a1, 0(a0)

# CHECK-S-OBJ: amomin.w a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amomin.w a2, a1, (a0)
amomin.w a2, a1, 0(a0)

# CHECK-S-OBJ: amomin.w.aq a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amomin.w.aq a2, a1, (a0)
amomin.w.aq a2, a1, 0(a0)

# CHECK-S-OBJ: amomin.w.rl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amomin.w.rl a2, a1, (a0)
amomin.w.rl a2, a1, 0(a0)

# CHECK-S-OBJ: amomin.w.aqrl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amomin.w.aqrl a2, a1, (a0)
amomin.w.aqrl a2, a1, 0(a0)

# CHECK-S-OBJ: amomax.w a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amomax.w a2, a1, (a0)
amomax.w a2, a1, 0(a0)

# CHECK-S-OBJ: amomax.w.aq a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amomax.w.aq a2, a1, (a0)
amomax.w.aq a2, a1, 0(a0)

# CHECK-S-OBJ: amomax.w.rl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amomax.w.rl a2, a1, (a0)
amomax.w.rl a2, a1, 0(a0)

# CHECK-S-OBJ: amomax.w.aqrl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amomax.w.aqrl a2, a1, (a0)
amomax.w.aqrl a2, a1, 0(a0)

# CHECK-S-OBJ: amominu.w a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amominu.w a2, a1, (a0)
amominu.w a2, a1, 0(a0)

# CHECK-S-OBJ: amominu.w.aq a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amominu.w.aq a2, a1, (a0)
amominu.w.aq a2, a1, 0(a0)

# CHECK-S-OBJ: amominu.w.rl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amominu.w.rl a2, a1, (a0)
amominu.w.rl a2, a1, 0(a0)

# CHECK-S-OBJ: amominu.w.aqrl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amominu.w.aqrl a2, a1, (a0)
amominu.w.aqrl a2, a1, 0(a0)

# CHECK-S-OBJ: amomaxu.w a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amomaxu.w a2, a1, (a0)
amomaxu.w a2, a1, 0(a0)

# CHECK-S-OBJ: amomaxu.w.aq a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amomaxu.w.aq a2, a1, (a0)
amomaxu.w.aq a2, a1, 0(a0)

# CHECK-S-OBJ: amomaxu.w.rl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amomaxu.w.rl a2, a1, (a0)
amomaxu.w.rl a2, a1, 0(a0)

# CHECK-S-OBJ: amomaxu.w.aqrl a2, a1, (a0)
# CHECK-S-OBJ-NOALIAS: amomaxu.w.aqrl a2, a1, (a0)
amomaxu.w.aqrl a2, a1, 0(a0)
