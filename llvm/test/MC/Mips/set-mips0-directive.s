# RUN: llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 | \
# RUN:   FileCheck %s

    .text
    rotr  $7, $7, 22

    .set mips32r6
    mod   $2, $4, $6
    .set mips0
    rotr  $2, $2, 15

    .set mips3
    dadd  $4, $4, $4
    .set mips0
    rotr  $3, $3, 19

# CHECK: rotr  $7, $7, 22

# CHECK: .set mips32r6
# CHECK: mod   $2, $4, $6
# CHECK: .set mips0
# CHECK: rotr  $2, $2, 15

# CHECK: .set mips3
# CHECK: dadd  $4, $4, $4
# CHECK: .set mips0
# CHECK: rotr  $3, $3, 19
