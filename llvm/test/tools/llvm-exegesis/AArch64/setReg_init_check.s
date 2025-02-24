# REQUIRES: aarch64-registered-target

# ppr register class initialization testcase 
# ideally we should use PTRUE_{B/H?S/D} instead of FADDV_VPZ_D for isolated testcase; but exegesis does not support PTRUE_{B/H?S/D} yet;
# RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency -opcode-name=FADDV_VPZ_D 2>&1 | FileCheck %s --check-prefix=PPR
# PPR-NOT: setRegTo is not implemented, results will be unreliable
# PPR: assembled_snippet: {{.*}}C0035FD6

# zpr register class initialization testcase 
# ideally we should use DUPM_ZI instead of FADDV_VPZ_S for isolated testcase; but exegesis does not support DUPM_ZI yet;
# RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency -opcode-name=FADDV_VPZ_S 2>&1 | FileCheck %s --check-prefix=ZPR
# ZPR-NOT: setRegTo is not implemented, results will be unreliable
# ZPR: assembled_snippet: {{.*}}C0035FD6

# fpr64 register class initialization testcase
# RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency -opcode-name=ADDVv4i16v 2>&1 | FileCheck %s --check-prefix=FPR64
# FPR64-NOT: setRegTo is not implemented, results will be unreliable
# FPR64: assembled_snippet: {{.*}}C0035FD6

# fpr128 register class initialization testcase
# RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency -opcode-name=ADDVv16i8v 2>&1 | FileCheck %s --check-prefix=FPR128
# FPR128-NOT: setRegTo is not implemented, results will be unreliable
# FPR128: assembled_snippet: {{.*}}C0035FD6