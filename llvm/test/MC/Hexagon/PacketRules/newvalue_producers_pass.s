# RUN: llvm-mc -arch=hexagon -mhvx -mcpu=hexagonv65 -filetype=obj %s | llvm-objdump --mattr=+hvxv65 -d - | FileCheck %s

{ r0=r0
  memw(r0)=r0.new }
# CHECK: { r0 = r0
# CHECK:   memw(r0+#0) = r0.new }

{ v0=v0
  vmem(r0)=v0.new }
# CHECK: { v0 = v0
# CHECK:   vmem(r0+#0) = v0.new }

{ v1:0=v1:0
  vmem(r0)=v0.new }
# CHECK: { v1:0 = vcombine(v1,v0)
# CHECK:   vmem(r0+#0) = v0.new }

{ r0=r0
  if (cmp.eq(r0.new,r0)) jump:t 0x0 }
# CHECK: { r0 = r0
# CHECK:   if (cmp.eq(r0.new,r0)) jump:t 0x18

{ vtmp.h=vgather(r0,m0,v0.h).h
  vmem(r0)=vtmp.new }
# CHECK: { vtmp.h = vgather(r0,m0,v0.h).h
# CHECK:   vmem(r0+#0) = vtmp.new }

{ if (p0) r0=r0
  if (p0) memw(r0)=r0.new }
# CHECK: { if (p0) r0 = add(r0,#0)
# CHECK:   if (p0) memw(r0+#0) = r0.new }

{ r0=r0
  if (p0) memw(r0)=r0.new }
# CHECK: { r0 = r0
# CHECK:   if (p0) memw(r0+#0) = r0.new }

{ r0=r0
  if (!p0) memw(r0)=r0.new }
# CHECK: { r0 = r0
# CHECK:   if (!p0) memw(r0+#0) = r0.new }
