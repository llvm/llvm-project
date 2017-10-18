#RUN: llvm-mc -filetype=obj -triple=hexagon -mcpu=hexagonv60 -mhvx %s

{ vmem (r0 + #0) = v0
  r0 = memw(r0) } 
