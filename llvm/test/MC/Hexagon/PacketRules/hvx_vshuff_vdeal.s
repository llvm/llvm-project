# RUN: not llvm-mc -arch=hexagon -mv65 -mhvx -filetype=obj -o 1.o %s 2>&1 | FileCheck --implicit-check-not=error %s

{ v1 = v2; vshuff(v1,v3,r0) }
# CHECK: error: register `V1' modified more than once

{ v4 = v3; vdeal(v6,v4,r0) }
# CHECK: error: register `V4' modified more than once
