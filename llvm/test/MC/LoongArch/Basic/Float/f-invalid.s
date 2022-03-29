# RUN: not llvm-mc --triple=loongarch32 -mattr=+f %s 2>&1 | FileCheck %s

# CHECK: :[[#@LINE+1]]:1: error: instruction requires the following: 'D' (Double-Precision Floating-Point)
fadd.d $fa0, $fa0, $fa0
