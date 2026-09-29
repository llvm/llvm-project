# RUN: not llvm-mc -triple=hexagon -filetype=obj %s 2>&1 | FileCheck %s

# CHECK: 4:3: error: Cannot write to read-only register `PC'
{ pc = r0
  r0 = r0 }
