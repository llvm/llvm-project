# RUN: not llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=atom -o %t1 2>&1 | FileCheck %s

# CHECK: error: please specify an out-of-order cpu. 'atom' is an in-order cpu.

