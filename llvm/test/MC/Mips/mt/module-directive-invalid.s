# RUN: not llvm-mc -arch=mips -mcpu=mips32r5 %s 2>%t1
# RUN: FileCheck %s < %t1

# CHECK: error: .module directive must appear before any code
  .set  nomips16
  .module mt
  nop
