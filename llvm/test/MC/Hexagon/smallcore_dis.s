# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv67t -filetype=obj %s | llvm-objdump -d - | FileCheck %s
# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv67t -filetype=obj %s | llvm-objdump --mcpu=hexagonv67t -d - | FileCheck %s

    .text
{
  r1 = memb(r0)
  if (p0) memb(r0) = r2
}

# CHECK:      { r1 = memb(r0+#0)
# CHECK-NEXT:   if (p0) memb(r0+#0) = r2 }
