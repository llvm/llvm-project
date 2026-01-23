# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv75 -filetype=obj %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv75 -mhvx -filetype=obj %s | llvm-objdump -d - | FileCheck %s

r1=memw(r0)
{ r0=r0
  memw(r0)=r0.new }

# CHECK: { r1 = memw(r0+#0x0) }
# CHECK: { r0 = r0
# CHECK:   memw(r0+#0x0) = r0.new }
