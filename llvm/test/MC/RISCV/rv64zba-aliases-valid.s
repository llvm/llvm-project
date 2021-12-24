# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zba -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s  -triple=riscv64 -mattr=+experimental-zba \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental-zba < %s \
# RUN:     | llvm-objdump -d -r -M no-aliases --mattr=+experimental-zba - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental-zba < %s \
# RUN:     | llvm-objdump -d -r --mattr=+experimental-zba - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s

# The following check prefixes are used in this test:
# CHECK-S-OBJ            Match both the .s and objdumped object output with
#                        aliases enabled
# CHECK-S-OBJ-NOALIAS    Match both the .s and objdumped object output with
#                        aliases disabled

# CHECK-S-OBJ-NOALIAS: add.uw t0, t1, zero
# CHECK-S-OBJ: zext.w t0, t1
zext.w x5, x6

# CHECK-S-OBJ-NOALIAS: addi t1, zero, -2
# CHECK-S-OBJ-NOALIAS-NEXT: add.uw t1, t1, zero
# CHECK-S-OBJ: li t1, -2
# CHECK-S-OBJ-NEXT: zext.w t1, t1
li x6, 0xfffffffe

# CHECK-S-OBJ-NOALIAS: lui t2, 699051
# CHECK-S-OBJ-NOALIAS-NEXT: addiw t2, t2, -1366
# CHECK-S-OBJ-NOALIAS-NEXT: add.uw t2, t2, zero
# CHECK-S-OBJ: lui t2, 699051
# CHECK-S-OBJ-NEXT: addiw t2, t2, -1366
# CHECK-S-OBJ-NEXT: zext.w t2, t2
li x7, 0xaaaaaaaa

# CHECK-S-OBJ-NOALIAS: lui t0, 768955
# CHECK-S-OBJ-NOALIAS-NEXT: slli.uw t0, t0, 12
# CHECK-S-OBJ-NOALIAS-NEXT: addi t0, t0, 1979
# CHECK-S-OBJ: lui t0, 768955
# CHECK-S-OBJ-NEXT: slli.uw t0, t0, 12
# CHECK-S-OBJ-NEXT: addi t0, t0, 1979
li x5, 0xbbbbb0007bb

# CHECK-S-OBJ-NOALIAS: lui t0, 768955
# CHECK-S-OBJ-NOALIAS-NEXT: slli.uw t0, t0, 4
# CHECK-S-OBJ: lui t0, 768955
# CHECK-S-OBJ-NEXT: slli.uw t0, t0, 4
li x5, 0xbbbbb0000

# CHECK-S-OBJ-NOALIAS: lui t1, 611378
# CHECK-S-OBJ-NOALIAS-NEXT: addiw t1, t1, 265
# CHECK-S-OBJ-NOALIAS-NEXT: sh1add t1, t1, t1
# CHECK-S-OBJ: lui t1, 611378
# CHECK-S-OBJ-NEXT: addiw t1, t1, 265
# CHECK-S-OBJ-NEXT: sh1add t1, t1, t1
li x6, -5372288229

# CHECK-S-OBJ-NOALIAS: lui t1, 437198
# CHECK-S-OBJ-NOALIAS-NEXT: addiw t1, t1, -265
# CHECK-S-OBJ-NOALIAS-NEXT: sh2add t1, t1, t1
# CHECK-S-OBJ: lui t1, 437198
# CHECK-S-OBJ-NEXT: addiw t1, t1, -265
# CHECK-S-OBJ-NEXT: sh2add t1, t1, t1
li x6, 8953813715

# CHECK-S-OBJ-NOALIAS: lui t1, 611378
# CHECK-S-OBJ-NOALIAS-NEXT: addiw t1, t1, 265
# CHECK-S-OBJ-NOALIAS-NEXT: sh2add t1, t1, t1
# CHECK-S-OBJ: lui t1, 611378
# CHECK-S-OBJ-NEXT: addiw t1, t1, 265
# CHECK-S-OBJ-NEXT: sh2add t1, t1, t1
li x6, -8953813715

# CHECK-S-OBJ-NOALIAS: lui t1, 437198
# CHECK-S-OBJ-NOALIAS-NEXT: addiw t1, t1, -265
# CHECK-S-OBJ-NOALIAS-NEXT: sh3add t1, t1, t1
# CHECK-S-OBJ: lui t1, 437198
# CHECK-S-OBJ-NEXT: addiw t1, t1, -265
# CHECK-S-OBJ-NEXT: sh3add t1, t1, t1
li x6, 16116864687

# CHECK-S-OBJ-NOALIAS: lui t1, 611378
# CHECK-S-OBJ-NOALIAS-NEXT: addiw t1, t1, 265
# CHECK-S-OBJ-NOALIAS-NEXT: sh3add t1, t1, t1
# CHECK-S-OBJ: lui t1, 611378
# CHECK-S-OBJ-NEXT: addiw t1, t1, 265
# CHECK-S-OBJ-NEXT: sh3add t1, t1, t1
li x6, -16116864687

# CHECK-S-OBJ-NOALIAS: lui t2, 768956
# CHECK-S-OBJ-NOALIAS-NEXT: addiw t2, t2, -1093
# CHECK-S-OBJ-NOALIAS-NEXT: slli.uw t2, t2, 12
# CHECK-S-OBJ-NOALIAS-NEXT: addi t2, t2, 1911
# CHECK-S-OBJ: lui t2, 768956
# CHECK-S-OBJ-NEXT: addiw t2, t2, -1093
# CHECK-S-OBJ-NEXT: slli.uw t2, t2, 12
# CHECK-S-OBJ-NEXT: addi t2, t2, 1911
li x7, 12900936431479

# CHECK-S-OBJ-NOALIAS: lui t1, 768955
# CHECK-S-OBJ-NOALIAS-NEXT: addiw t1, t1, 273
# CHECK-S-OBJ-NOALIAS-NEXT: slli.uw t1, t1, 12
# CHECK-S-OBJ-NOALIAS-NEXT: addi t1, t1, 273
# CHECK-S-OBJ: lui t1, 768955
# CHECK-S-OBJ-NEXT: addiw t1, t1, 273
# CHECK-S-OBJ-NEXT: slli.uw t1, t1, 12
# CHECK-S-OBJ-NEXT: addi t1, t1, 273
li x6, 12900925247761

# CHECK-S-OBJ-NOALIAS: lui t1, 768955
# CHECK-S-OBJ-NOALIAS-NEXT: addiw t1, t1, -1365
# CHECK-S-OBJ-NOALIAS-NEXT: slli.uw t1, t1, 12
# CHECK-S-OBJ-NOALIAS-NEXT: addi t1, t1, -1366
# CHECK-S-OBJ: lui t1, 768955
# CHECK-S-OBJ-NEXT: addiw t1, t1, -1365
# CHECK-S-OBJ-NEXT: slli.uw t1, t1, 12
# CHECK-S-OBJ-NEXT: addi t1, t1, -1366
li x6, 12900918536874

# CHECK-S-OBJ-NOALIAS: lui t1, 349525
# CHECK-S-OBJ-NOALIAS-NEXT: sh2add t1, t1, t1
# CHECK-S-OBJ-NOALIAS-NEXT: addi t1, t1, 1
# CHECK-S-OBJ: lui t1, 349525
# CHECK-S-OBJ-NEXT: sh2add t1, t1, t1
# CHECK-S-OBJ-NEXT: addi t1, t1, 1
li x6, 7158272001 # 0x0000_0001_aaaa_9001

# CHECK-S-OBJ-NOALIAS: lui t1, 349525
# CHECK-S-OBJ-NOALIAS-NEXT: sh3add t1, t1, t1
# CHECK-S-OBJ-NOALIAS-NEXT: addi t1, t1, 1
# CHECK-S-OBJ: lui t1, 349525
# CHECK-S-OBJ-NEXT: sh3add t1, t1, t1
# CHECK-S-OBJ-NEXT: addi t1, t1, 1
li x6, 12884889601 # 0x0000_0002_ffff_d001

# CHECK-S-OBJ-NOALIAS: lui t1, 768955
# CHECK-S-OBJ-NOALIAS-NEXT: sh1add t1, t1, t1
# CHECK-S-OBJ-NOALIAS-NEXT: addi t1, t1, 1
# CHECK-S-OBJ: lui t1, 768955
# CHECK-S-OBJ-NEXT: sh1add t1, t1, t1
# CHECK-S-OBJ-NEXT: addi t1, t1, 1
li x6, -3435982847 # 0xffff_ffff_3333_1001

# CHECK-S-OBJ-NOALIAS: lui t1, 768945
# CHECK-S-OBJ-NOALIAS-NEXT: sh2add t1, t1, t1
# CHECK-S-OBJ-NOALIAS-NEXT: addi t1, t1, 1
# CHECK-S-OBJ: lui t1, 768945
# CHECK-S-OBJ-NEXT: sh2add t1, t1, t1
# CHECK-S-OBJ-NEXT: addi t1, t1, 1
li x6, -5726842879 # 0xffff_fffe_aaa7_5001

# CHECK-S-OBJ-NOALIAS: lui t1, 768955
# CHECK-S-OBJ-NOALIAS-NEXT: sh3add t1, t1, t1
# CHECK-S-OBJ-NOALIAS-NEXT: addi t1, t1, 1
# CHECK-S-OBJ: lui t1, 768955
# CHECK-S-OBJ-NEXT: sh3add t1, t1, t1
# CHECK-S-OBJ-NEXT: addi t1, t1, 1
li x6, -10307948543 # 0xffff_fffd_9999_3001
