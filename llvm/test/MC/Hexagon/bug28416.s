# RUN: llvm-mc -arch=hexagon -filetype=obj -o - %s | llvm-objdump -d -
# r0 = r6 and jump ##undefined should compound to J4_jumpsetr

# CHECK: { immext(#0)
# CHECK:   r0 = r6 ; jump 0x0
# CHECK:   r1 = memub(r6+#21)
# CHECK:   memw(r9+#0) = r0 }
{  memw(r9) = r0
   r0 = r6
   r1 = memub(r6+#21)
   jump ##undefined }
