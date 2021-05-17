# RUN: llvm-mc -arch=hexagon -mv67 -mhvx -filetype=obj %s | llvm-objdump --mcpu=hexagonv67 --mattr=+hvx -d - | FileCheck %s

# CHECK: { v3:0.w = vrmpyz(v0.b,r0.ub) }
V3:0.w=vrmpyz(v0.b,r0.ub)
# CHECK: { v3:0.w += vrmpyz(v0.b,r0.ub) }
V3:0.w+=vrmpyz(v0.b,r0.ub)
# CHECK: { v3:0.w = vrmpyz(v0.b,r0.ub++) }
V3:0.w=vrmpyz(v0.b,r0.ub++)
# CHECK: { v3:0.w += vrmpyz(v0.b,r0.ub++) }
V3:0.w+=vrmpyz(v0.b,r0.ub++)
