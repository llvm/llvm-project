# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases=false \
# RUN:     | FileCheck -check-prefixes=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d -riscv-no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d -riscv-no-aliases=false - \
# RUN:     | FileCheck -check-prefixes=CHECK-ALIAS %s

# CHECK-INST: csrrs t4, 3202, zero
# CHECK-ALIAS: rdinstreth t4
rdinstreth x29
# CHECK-INST: csrrs s11, 3200, zero
# CHECK-ALIAS: rdcycleh s11
rdcycleh x27
# CHECK-INST: csrrs t3, 3201, zero
# CHECK-ALIAS: rdtimeh t3
rdtimeh x28
