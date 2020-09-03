# RUN: llvm-mc -arch=hexagon -mv67t -filetype=obj %s | llvm-objdump -d - | FileCheck %s
{ r0=r0 }
.align 32
{ r0=r0 }

# CHECK: { r0 = r0
# CHECK:   nop
# CHECK:   nop }
# CHECK: { nop
# CHECK:   nop }
# CHECK: { nop
# CHECK:   nop
# CHECK:   nop }
# CHECK: { r0 = r0 }

