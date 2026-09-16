# RUN: not llvm-mc %s -triple=mips -mcpu=mips32r6 2>&1 | \
# RUN: FileCheck %s --check-prefix=MIPS32-OR-R6
# RUN: not llvm-mc %s -triple=mips -mcpu=mips32r2 2>&1 | \
# RUN: FileCheck %s --check-prefix=MIPS32-OR-R6
# RUN: not llvm-mc %s -triple=mips64 -mcpu=mips64r6 2>&1 | \
# RUN: FileCheck %s --check-prefix=MIPS32-OR-R6
# RUN: not llvm-mc %s -triple=mips64 -mcpu=mips64r2 2>&1 | \
# RUN: FileCheck %s --check-prefix=MIPS64-NOT-R6

  .text
  ddiv $25, $11
  # MIPS32-OR-R6: :[[@LINE-1]]:{{[0-9]+}}: error: instruction requires a CPU feature not currently enabled

  ddiv $25, $0
  # MIPS64-NOT-R6: :[[@LINE-1]]:3: warning: division by zero

  ddiv $0,$0
  # MIPS64-NOT-R6: :[[@LINE-1]]:3: warning: dividing zero by zero

  ddiv $t0, $t1, f0
  # MIPS64-NOT-R6: :[[@LINE-1]]:3: error: expected immediate operand kind
