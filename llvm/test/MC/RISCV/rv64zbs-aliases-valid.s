# RUN: llvm-mc %s -triple=riscv64 -mattr=+zbs -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s  -triple=riscv64 -mattr=+zbs \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zbs < %s \
# RUN:     | llvm-objdump -d -r -M no-aliases --mattr=+zbs - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+zbs < %s \
# RUN:     | llvm-objdump -d -r --mattr=+zbs - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s

# The following check prefixes are used in this test:
# CHECK-S-OBJ            Match both the .s and objdumped object output with
#                        aliases enabled
# CHECK-S-OBJ-NOALIAS    Match both the .s and objdumped object output with
#                        aliases disabled

# CHECK-S-OBJ-NOALIAS: bseti t0, t1, 8
# CHECK-S-OBJ: bseti t0, t1, 8
bset x5, x6, 8

# CHECK-S-OBJ-NOALIAS: bclri t0, t1, 8
# CHECK-S-OBJ: bclri t0, t1, 8
bclr x5, x6, 8

# CHECK-S-OBJ-NOALIAS: binvi t0, t1, 8
# CHECK-S-OBJ: binvi t0, t1, 8
binv x5, x6, 8

# CHECK-S-OBJ-NOALIAS: bexti t0, t1, 8
# CHECK-S-OBJ: bexti t0, t1, 8
bext x5, x6, 8

# CHECK-S-OBJ-NOALIAS: addi t0, zero, 1365
# CHECK-S-OBJ-NOALIAS-NEXT: bseti t0, t0, 31
# CHECK-S-OBJ: li t0, 1365
# CHECK-S-OBJ-NEXT: bseti t0, t0, 31
li x5, 2147485013

# CHECK-S-OBJ-NOALIAS: lui t1, 572348
# CHECK-S-OBJ-NOALIAS-NEXT: addiw t1, t1, -1093
# CHECK-S-OBJ-NOALIAS-NEXT: bclri t1, t1, 44
# CHECK-S-OBJ-NOALIAS-NEXT: bclri t1, t1, 63
# CHECK-S-OBJ: lui t1, 572348
# CHECK-S-OBJ-NEXT: addiw t1, t1, -1093
# CHECK-S-OBJ-NEXT: bclri t1, t1, 44
# CHECK-S-OBJ-NEXT: bclri t1, t1, 63
li x6, 9223354442718100411

# CHECK-S-OBJ-NOALIAS: lui t1, 506812
# CHECK-S-OBJ-NOALIAS-NEXT: addiw t1, t1, -1093
# CHECK-S-OBJ-NOALIAS-NEXT: bseti t1, t1, 46
# CHECK-S-OBJ-NOALIAS-NEXT: bseti t1, t1, 63
# CHECK-S-OBJ: lui t1, 506812
# CHECK-S-OBJ-NEXT: addiw t1, t1, -1093
# CHECK-S-OBJ-NEXT: bseti t1, t1, 46
# CHECK-S-OBJ-NEXT: bseti t1, t1, 63
li x6, -9223301666034697285

# CHECK-S-OBJ-NOALIAS: bseti t2, zero, 11
# CHECK-S-OBJ: bseti t2, zero, 11
li x7, 2048
