# RUN: not llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips1 2>%t1
# RUN: FileCheck %s < %t1

# FIXME: At the moment we emit the wrong error message if we try to assemble the
# ll instruction using an unsupported architecture so we just check for "error" 
# and ignore the rest of the message.

        .text
        .set noreorder
        .set mips1
        ll  $2,-2($2) # CHECK: error:
        .set mips2
        dadd $2,$2,$2 # CHECK: error: instruction requires a CPU feature not currently enabled
        .set mips3
        ldxc1 $f8,$2($4) # CHECK: error: instruction requires a CPU feature not currently enabled
        .set mips4
        luxc1 $f19,$2($4) # CHECK: error: instruction requires a CPU feature not currently enabled
        .set mips5
        clo  $2,$2 # CHECK: error: instruction requires a CPU feature not currently enabled
        .set mips32
        rotr    $2,15 # CHECK: error: instruction requires a CPU feature not currently enabled
        .set mips32r2
        mod $2, $4, $6 # CHECK: error:instruction requires a CPU feature not currently enabled
        .set mips64r3
        .set mips32r3
        daddi $2, $2, 10 # CHECK: error: instruction requires a CPU feature not currently enabled
        .set mips64r3
        .set mips32r5
        daddi $2, $2, 10 # CHECK: error: instruction requires a CPU feature not currently enabled
        .set mips32r6
        daddi $2, $2, 10 # CHECK: error: instruction requires a CPU feature not currently enabled
        .set mips64
        drotr32 $1,$14,15 # CHECK: error: instruction requires a CPU feature not currently enabled
        .set mips64r2
        mod $2, $4, $6 # CHECK: error: instruction requires a CPU feature not currently enabled
        .set mips64r6
        .set mips64r3
        mod $2, $4, $6 # CHECK: error: instruction requires a CPU feature not currently enabled
        .set mips64r6
        .set mips64r5
        mod $2, $4, $6 # CHECK: error: instruction requires a CPU feature not currently enabled

