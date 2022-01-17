# RUN: llvm-mc -arch=hexagon -filetype=obj %s | llvm-objdump -d - | FileCheck %s

{ r0=sub(#1,r0)
  r1=sub(#1, r0)
  r2=sub(#1, r0)
  dczeroa(r0) }

# CHECK: { r0 = sub(#1,r0)
# CHECK:   r1 = sub(#1,r0)
# CHECK:   r2 = sub(#1,r0)
# CHECK:   dczeroa(r0) }
