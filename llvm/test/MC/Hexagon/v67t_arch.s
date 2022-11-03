# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv67t -filetype=obj %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s
# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv67t -mhvx -filetype=obj %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

r1=memw(r0)
{ r0=r0
  memw(r0)=r0.new }

# CHECK: { r1 = memw(r0+#0) }
# CHECK: { r0 = r0
# CHECK:   memw(r0+#0) = r0.new }
