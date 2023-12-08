# RUN: llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips1 | \
# RUN:   FileCheck %s

        .text
        .set noreorder
        .set mips1
        add $2, $2, $2
        .set mips2
        ll  $2,-2($2)
        .set mips3
        dadd $2,$2,$2
        .set mips4
        ldxc1 $f8,$2($4)
        .set mips5
        luxc1 $f19,$2($4)
        .set mips32
        clo  $2,$2
        .set mips32r2
        rotr    $2,15
        .set mips32
        .set mips32r3
        rotr    $2,15
        .set mips32
        .set mips32r5
        rotr    $2,15
        .set mips32r6
        mod $2, $4, $6
        .set mips64
        daddi $2, $2, 10
        .set mips64r2
        drotr32 $1,$14,15
        .set mips64
        .set mips64r3
        drotr32 $1,$14,15
        .set mips64
        .set mips64r5
        drotr32 $1,$14,15
        .set mips64r6
        mod $2, $4, $6

# CHECK: .set noreorder
# CHECK: .set mips1
# CHECK: add $2, $2, $2
# CHECK: .set mips2
# CHECK: ll  $2, -2($2)
# CHECK: .set mips3
# CHECK: dadd $2, $2, $2
# CHECK: .set mips4
# CHECK: ldxc1 $f8, $2($4)
# CHECK: .set mips5
# CHECK: luxc1 $f19, $2($4)
# CHECK: .set mips32
# CHECK: clo $2, $2
# CHECK: .set mips32r2
# CHECK: rotr $2, $2, 15
# CHECK: .set mips32
# CHECK: .set mips32r3
# CHECK: rotr $2, $2, 15
# CHECK: .set mips32
# CHECK: .set mips32r5
# CHECK: rotr $2, $2, 15
# CHECK: .set mips32r6
# CHECK: mod $2, $4, $6
# CHECK: .set mips64
# CHECK: daddi $2, $2, 10
# CHECK: .set mips64r2
# CHECK:  drotr32 $1, $14, 15
# CHECK: .set mips64
# CHECK: .set mips64r3
# CHECK:  drotr32 $1, $14, 15
# CHECK: .set mips64
# CHECK: .set mips64r5
# CHECK:  drotr32 $1, $14, 15
# CHECK: .set mips64r6
# CHECK: mod $2, $4, $6
