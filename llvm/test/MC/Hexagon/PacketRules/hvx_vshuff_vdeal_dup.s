# RUN: not llvm-mc -arch=hexagon -mv65 -mhvx -filetype=obj %s 2>&1 | FileCheck %s

{ vshuff(v0,v0,r0) }
# CHECK: error: register `V0' modified more than once

{ vdeal(v1,v1,r0) }
# CHECK: error: register `V1' modified more than once
