# RUN: not llvm-mc %s -arch=mips -mcpu=mips32r6 2>&1 | \
# RUN: FileCheck %s --check-prefix=R6
# RUN: not llvm-mc %s -arch=mips64 -mcpu=mips64r6 2>&1 | \
# RUN: FileCheck %s --check-prefix=R6
# RUN: llvm-mc %s -arch=mips -mcpu=mips32r2 2>&1 | \
# RUN: FileCheck %s --check-prefix=NOT-R6
# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64r2 2>&1 | \
# RUN: FileCheck %s --check-prefix=NOT-R6

  .text
  divu $25, 11
  # R6: :[[@LINE-1]]:3: error: instruction requires a CPU feature not currently enabled

  divu $25, $0
  # NOT-R6: :[[@LINE-1]]:3: warning: division by zero

  divu $0,$0
  # NOT-R6: :[[@LINE-1]]:3: warning: dividing zero by zero
