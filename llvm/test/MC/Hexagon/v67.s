# RUN: llvm-mc -arch=hexagon -mv67 -mattr=+hvx,+hvx-length128B -filetype=obj %s | llvm-objdump --mhvx=v66 -d - | FileCheck %s

# CHECK: 1a81e0e2 { v2.uw = vrotr(v0.uw,v1.uw) }
  v2.uw=vrotr(v0.uw, v1.uw)
