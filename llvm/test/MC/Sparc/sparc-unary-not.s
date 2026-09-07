# RUN: llvm-mc %s -triple=sparc | FileCheck %s

# CHECK: and %g3, -32, %g3
and %g3, ~0x0000001f, %g3
