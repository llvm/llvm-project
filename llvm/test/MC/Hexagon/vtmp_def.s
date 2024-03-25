# RUN: not llvm-mc -triple=hexagon -mv65 -mhvx -filetype=obj %s 2>&1 | FileCheck %s

# CHECK: register `VTMP' modified more than once
{ vtmp.h=vgather(r0, m0, v1:0.w).h
  vtmp.h=vgather(r0, m0, v1:0.w).h }
