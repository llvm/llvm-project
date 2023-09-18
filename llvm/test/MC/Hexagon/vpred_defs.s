# RUN: llvm-mc -triple=hexagon -mv65 -filetype=asm -mhvx %s | FileCheck %s

# CHECK-NOT: error: register `{{.+}}' modified more than once

{ Q0 = VCMP.EQ(V0.h,V4.h)
  Q1 = VCMP.EQ(V1.h,V6.h)
  IF (Q3) VTMP.h = VGATHER(R0,M0,V3.h).h
  VMEM(R4++#1) = VTMP.new
}
