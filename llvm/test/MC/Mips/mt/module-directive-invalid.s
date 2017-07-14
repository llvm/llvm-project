# RUN: not llvm-mc -arch=mips -mcpu=mips32r5 < %s 2>&1 | FileCheck %s

# CHECK: error: .module directive must appear before any code
  .set  nomips16
  .module mt
  nop
