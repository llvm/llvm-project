# RUN: not llvm-mc -triple=riscv32 --mattr=+experimental-p %s 2>&1 \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

# Imm overflow
pli.h a0, 0x400
# CHECK-ERROR: immediate must be an integer in the range [-512, 511]
pli.b a0, 0x200
# CHECK-ERROR: immediate must be an integer in the range [0, 255]

# Must be even register
pwadd.b a1, a2, a3
# CHECK-ERROR: register must be even
pmqwacc.h a1, a1, a2
# CHECK-ERROR: register must be even

pslli.b a6, a7, 100
# CHECK-ERROR: immediate must be an integer in the range [0, 7]
pslli.h ra, sp, 100
# CHECK-ERROR: immediate must be an integer in the range [0, 15]
psslai.h t0, t1, 100
# CHECK-ERROR: immediate must be an integer in the range [0, 15]
sslai a4, a5, -1
# CHECK-ERROR: immediate must be an integer in the range [0, 31]
