# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-p %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

# Imm overflow
pli.h a0, 0x400
# CHECK-ERROR: immediate must be an integer in the range [-512, 511]

pli.w a1, -0x201
# CHECK-ERROR: immediate must be an integer in the range [-512, 511]
