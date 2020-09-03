# RUN: llvm-mc -arch=hexagon -mv67 -mhvx -filetype=obj %s | llvm-objdump --mhvx=v66 -d - | FileCheck %s

# CHECK: { v3:0.w = vrmpyz(v0.b,r0.ub) }
V3:0.w=vrmpyz(v0.b,r0.ub)
# CHECK: { v3:0.w += vrmpyz(v0.b,r0.ub) }
V3:0.w+=vrmpyz(v0.b,r0.ub)
# CHECK: { v3:0.w = vrmpyz(v0.b,r0.ub++) }
V3:0.w=vrmpyz(v0.b,r0.ub++)
# CHECK: { v3:0.w += vrmpyz(v0.b,r0.ub++) }
V3:0.w+=vrmpyz(v0.b,r0.ub++)
