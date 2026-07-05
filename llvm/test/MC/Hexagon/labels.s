# RUN: llvm-mc -triple=hexagon -filetype=asm -o - %s | FileCheck %s

# CHECK: a:
a:

# CHECK: r1:
r1:

# CHECK: r3:
# CHECK: nop
r3:nop

# CHECK: r5:4 = combine(r5,r4)
r5:4 = r5:4

# CHECK: r0 = r1
# CHECK: p0 = tstbit(r0,#10)
# CHECK: if (!p0) jump
1:r0=r1; p0=tstbit(r0, #10); if !p0 jump 1b;

# CHECK: nop
# CHECK: r1 = add(r1,#4)
# CHECK: r5 = memw(r1+#0)
# CHECK: endloop0
b: { r5 = memw(r1)
     r1 = add(r1, #4) } : endloop0
