# REQUIRES: aarch64-registered-target

# ppr register class initialization testcase 
# ideally we should use PTRUE_{B/H?S/D} instead of FADDV_VPZ_D for isolated testcase; but exegesis does not support PTRUE_{B/H?S/D} yet;
# RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=FADDV_VPZ_D.o --opcode-name=FADDV_VPZ_D 2>&1 | FileCheck %s --check-prefix=PPR
# RUN: llvm-objdump -d FADDV_VPZ_D.o | FileCheck %s --check-prefix=PPR_ASM
# PPR-NOT: setRegTo is not implemented, results will be unreliable
# PPR: assembled_snippet: {{.*}}C0035FD6
# PPR_ASM: {{0|4}}:	{{.*}} ptrue p{{[0-9]|1[0-5]}}

# zpr register class initialization testcase 
# ideally we should use DUPM_ZI instead of FADDV_VPZ_S for isolated testcase; but exegesis does not support DUPM_ZI yet;
# RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=FADDV_VPZ_S.o --opcode-name=FADDV_VPZ_S 2>&1 | FileCheck %s --check-prefix=ZPR
# RUN: llvm-objdump -d FADDV_VPZ_S.o | FileCheck %s --check-prefix=ZPR_ASM
# ZPR-NOT: setRegTo is not implemented, results will be unreliable
# ZPR: assembled_snippet: {{.*}}C0035FD6
# ZPR_ASM: {{4|8}}: {{.*}} dupm z{{[0-9]|[1-2][0-9]|3[0-1]}}.s, #0x1

# fpr64 register class initialization testcase
# RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=ADDVv4i16v.o --opcode-name=ADDVv4i16v 2>&1 | FileCheck %s --check-prefix=FPR64
# RUN: llvm-objdump -d ADDVv4i16v.o | FileCheck %s --check-prefix=FPR64-ASM
# FPR64-NOT: setRegTo is not implemented, results will be unreliable
# FPR64: assembled_snippet: {{.*}}C0035FD6
# FPR64-ASM: {{0|4}}:	{{.*}} fmov d{{[0-9]|[1-2][0-9]|3[0-1]}}, #2.0{{.*}}

# fpr128 register class initialization testcase
# RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency --dump-object-to-disk=ADDVv16i8v.o --opcode-name=ADDVv16i8v 2>&1 | FileCheck %s --check-prefix=FPR128
# RUN: llvm-objdump -d ADDVv16i8v.o | FileCheck %s --check-prefix=FPR128-ASM
# FPR128-NOT: setRegTo is not implemented, results will be unreliable
# FPR128: assembled_snippet: {{.*}}C0035FD6
# FPR128-ASM: {{0|4}}:	{{.*}} movi v{{[0-9]|[1-2][0-9]|3[0-1]}}.2d, {{#0x0|#0000000000000000}}
