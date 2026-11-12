@ RUN: not llvm-mc -triple armv8.1m.main -mattr=+mve < %s 2>&1 | FileCheck %s

  sqrshr    r1, r1
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Rda register and Rm register can't be identical
  uqrshl   r1, r1
@ CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Rda register and Rm register can't be identical
