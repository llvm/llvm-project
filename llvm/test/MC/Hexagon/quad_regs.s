# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv66 -mhvx -filetype=obj %s | llvm-objdump --mcpu=hexagonv66 --mattr=+hvx -d - | FileCheck %s

# Test for quad register parsing and printing
# CHECK: { v3:0.w = vrmpyz(v0.b,r0.b) }
v3:0.w = vrmpyz(v0.b,r0.b)
