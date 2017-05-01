# RUN: not llvm-mc -arch=hexagon -filetype=obj -mv5 %s 2>&1 | FileCheck %s

# CHECK: 4:1: error: Cannot write to read-only register `PC'
pc = r0

# CHECK: 7:1: error: Cannot write to read-only register `PC'
c9 = r0
