# RUN: llvm-mc %s -triple=riscv64 -mattr=+zmmul -M no-aliases 2>&1 \
# RUN:  | FileCheck -check-prefixes=CHECK-INST %s

# CHECK-INST: mulw ra, sp, gp
mulw ra, sp, gp
