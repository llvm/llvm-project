# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zbp -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zbp \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-zbp < %s \
# RUN:     | llvm-objdump -d -r -M no-aliases --mattr=+experimental-zbp - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-zbp < %s \
# RUN:     | llvm-objdump -d -r --mattr=+experimental-zbp - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s

# The following check prefixes are used in this test:
# CHECK-S-OBJ            Match both the .s and objdumped object output with
#                        aliases enabled
# CHECK-S-OBJ-NOALIAS    Match both the .s and objdumped object output with
#                        aliases disabled

# CHECK-S-OBJ-NOALIAS: zext.h t0, t1
# CHECK-S-OBJ: zext.h t0, t1
zext.h x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 1
# CHECK-S-OBJ: rev.p t0, t1
rev.p x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 2
# CHECK-S-OBJ: rev2.n t0, t1
rev2.n x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 3
# CHECK-S-OBJ: rev.n t0, t1
rev.n x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 4
# CHECK-S-OBJ: rev4.b t0, t1
rev4.b x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 6
# CHECK-S-OBJ: rev2.b t0, t1
rev2.b x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 7
# CHECK-S-OBJ: rev.b t0, t1
rev.b x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 8
# CHECK-S-OBJ: rev8.h t0, t1
rev8.h x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 12
# CHECK-S-OBJ: rev4.h t0, t1
rev4.h x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 14
# CHECK-S-OBJ: rev2.h t0, t1
rev2.h x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 15
# CHECK-S-OBJ: rev.h t0, t1
rev.h x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 16
# CHECK-S-OBJ: rev16 t0, t1
rev16 x5, x6

# CHECK-S-OBJ-NOALIAS: rev8 t0, t1
# CHECK-S-OBJ: rev8 t0, t1
rev8 x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 28
# CHECK-S-OBJ: rev4 t0, t1
rev4 x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 30
# CHECK-S-OBJ: rev2 t0, t1
rev2 x5, x6

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 31
# CHECK-S-OBJ: rev t0, t1
rev x5, x6

# CHECK-S-OBJ-NOALIAS: shfli t0, t1, 1
# CHECK-S-OBJ: zip.n t0, t1
zip.n x5, x6

# CHECK-S-OBJ-NOALIAS: unshfli t0, t1, 1
# CHECK-S-OBJ: unzip.n t0, t1
unzip.n x5, x6

# CHECK-S-OBJ-NOALIAS: shfli t0, t1, 2
# CHECK-S-OBJ: zip2.b t0, t1
zip2.b x5, x6

# CHECK-S-OBJ-NOALIAS: unshfli t0, t1, 2
# CHECK-S-OBJ: unzip2.b t0, t1
unzip2.b x5, x6

# CHECK-S-OBJ-NOALIAS: shfli t0, t1, 3
# CHECK-S-OBJ: zip.b t0, t1
zip.b x5, x6

# CHECK-S-OBJ-NOALIAS: unshfli t0, t1, 3
# CHECK-S-OBJ: unzip.b t0, t1
unzip.b x5, x6

# CHECK-S-OBJ-NOALIAS: shfli t0, t1, 4
# CHECK-S-OBJ: zip4.h t0, t1
zip4.h x5, x6

# CHECK-S-OBJ-NOALIAS: unshfli t0, t1, 4
# CHECK-S-OBJ: unzip4.h t0, t1
unzip4.h x5, x6

# CHECK-S-OBJ-NOALIAS: shfli t0, t1, 6
# CHECK-S-OBJ: zip2.h t0, t1
zip2.h x5, x6

# CHECK-S-OBJ-NOALIAS: unshfli t0, t1, 6
# CHECK-S-OBJ: unzip2.h t0, t1
unzip2.h x5, x6

# CHECK-S-OBJ-NOALIAS: shfli t0, t1, 7
# CHECK-S-OBJ: zip.h t0, t1
zip.h x5, x6

# CHECK-S-OBJ-NOALIAS: unshfli t0, t1, 7
# CHECK-S-OBJ: unzip.h t0, t1
unzip.h x5, x6

# CHECK-S-OBJ-NOALIAS: shfli t0, t1, 8
# CHECK-S-OBJ: zip8 t0, t1
zip8 x5, x6

# CHECK-S-OBJ-NOALIAS: unshfli t0, t1, 8
# CHECK-S-OBJ: unzip8 t0, t1
unzip8 x5, x6

# CHECK-S-OBJ-NOALIAS: shfli t0, t1, 12
# CHECK-S-OBJ: zip4 t0, t1
zip4 x5, x6

# CHECK-S-OBJ-NOALIAS: unshfli t0, t1, 12
# CHECK-S-OBJ: unzip4 t0, t1
unzip4 x5, x6

# CHECK-S-OBJ-NOALIAS: shfli t0, t1, 14
# CHECK-S-OBJ: zip2 t0, t1
zip2 x5, x6

# CHECK-S-OBJ-NOALIAS: unshfli t0, t1, 14
# CHECK-S-OBJ: unzip2 t0, t1
unzip2 x5, x6

# CHECK-S-OBJ-NOALIAS: shfli t0, t1, 15
# CHECK-S-OBJ: zip t0, t1
zip x5, x6

# CHECK-S-OBJ-NOALIAS: unshfli t0, t1, 15
# CHECK-S-OBJ: unzip t0, t1
unzip x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 1
# CHECK-S-OBJ: orc.p t0, t1
orc.p x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 2
# CHECK-S-OBJ: orc2.n t0, t1
orc2.n x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 3
# CHECK-S-OBJ: orc.n t0, t1
orc.n x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 4
# CHECK-S-OBJ: orc4.b t0, t1
orc4.b x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 6
# CHECK-S-OBJ: orc2.b t0, t1
orc2.b x5, x6

# CHECK-S-OBJ-NOALIAS: orc.b t0, t1
# CHECK-S-OBJ: orc.b t0, t1
orc.b x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 8
# CHECK-S-OBJ: orc8.h t0, t1
orc8.h x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 12
# CHECK-S-OBJ: orc4.h t0, t1
orc4.h x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 14
# CHECK-S-OBJ: orc2.h t0, t1
orc2.h x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 15
# CHECK-S-OBJ: orc.h t0, t1
orc.h x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 16
# CHECK-S-OBJ: orc16 t0, t1
orc16 x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 24
# CHECK-S-OBJ: orc8 t0, t1
orc8 x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 28
# CHECK-S-OBJ: orc4 t0, t1
orc4 x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 30
# CHECK-S-OBJ: orc2 t0, t1
orc2 x5, x6

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 31
# CHECK-S-OBJ: orc t0, t1
orc x5, x6

# CHECK-S-OBJ-NOALIAS: rori t0, t1, 8
# CHECK-S-OBJ: rori t0, t1, 8
ror x5, x6, 8

# CHECK-S-OBJ-NOALIAS: grevi t0, t1, 13
# CHECK-S-OBJ: grevi t0, t1, 13
grev x5, x6, 13

# CHECK-S-OBJ-NOALIAS: gorci t0, t1, 13
# CHECK-S-OBJ: gorci t0, t1, 13
gorc x5, x6, 13

# CHECK-S-OBJ-NOALIAS: shfli t0, t1, 13
# CHECK-S-OBJ: shfli t0, t1, 13
shfl x5, x6, 13

# CHECK-S-OBJ-NOALIAS: unshfli t0, t1, 13
# CHECK-S-OBJ: unshfli t0, t1, 13
unshfl x5, x6, 13
