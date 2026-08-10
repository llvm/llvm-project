# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN: FileCheck %s
# Check that the assembler can handle the documented syntax
# for ".set at" and set the correct value.
    .text
foo:
# CHECK: .set at=$1
# CHECK: lui  $1, 1
# CHECK: addu $1, $1, $2
# CHECK: lw   $2, 0($1)
    .set    at=$1
        lw      $2, 65536($2)
# CHECK: .set at=$2
# CHECK: lui  $2, 1
# CHECK: addu $2, $2, $1
# CHECK: lw   $1, 0($2)
    .set    at=$2
        lw      $1, 65536($1)
# CHECK: .set at=$3
# CHECK: lui  $3, 1
# CHECK: addu $3, $3, $1
# CHECK: lw   $1, 0($3)
    .set    at=$3
        lw      $1, 65536($1)
# CHECK: .set at=$4
# CHECK: lui  $4, 1
# CHECK: addu $4, $4, $1
# CHECK: lw   $1, 0($4)
    .set    at=$a0
        lw      $1, 65536($1)
# CHECK: .set at=$5
# CHECK: lui  $5, 1
# CHECK: addu $5, $5, $1
# CHECK: lw   $1, 0($5)
    .set    at=$a1
        lw      $1, 65536($1)
# CHECK: .set at=$6
# CHECK: lui  $6, 1
# CHECK: addu $6, $6, $1
# CHECK: lw   $1, 0($6)
    .set    at=$a2
        lw      $1, 65536($1)
# CHECK: .set at=$7
# CHECK: lui  $7, 1
# CHECK: addu $7, $7, $1
# CHECK: lw   $1, 0($7)
    .set    at=$a3
        lw      $1, 65536($1)
# CHECK: .set at=$8
# CHECK: lui  $8, 1
# CHECK: addu $8, $8, $1
# CHECK: lw   $1, 0($8)
    .set    at=$8
        lw      $1, 65536($1)
# CHECK: .set at=$9
# CHECK: lui  $9, 1
# CHECK: addu $9, $9, $1
# CHECK: lw   $1, 0($9)
    .set    at=$9
        lw      $1, 65536($1)
# CHECK: .set at=$10
# CHECK: lui  $10, 1
# CHECK: addu $10, $10, $1
# CHECK: lw   $1, 0($10)
    .set    at=$10
        lw      $1, 65536($1)
# CHECK: .set at=$11
# CHECK: lui  $11, 1
# CHECK: addu $11, $11, $1
# CHECK: lw   $1, 0($11)
    .set    at=$11
        lw      $1, 65536($1)
# CHECK: .set at=$12
# CHECK: lui  $12, 1
# CHECK: addu $12, $12, $1
# CHECK: lw   $1, 0($12)
    .set    at=$12
        lw      $1, 65536($1)
# CHECK: .set at=$13
# CHECK: lui  $13, 1
# CHECK: addu $13, $13, $1
# CHECK: lw   $1, 0($13)
    .set    at=$13
        lw      $1, 65536($1)
# CHECK: .set at=$14
# CHECK: lui  $14, 1
# CHECK: addu $14, $14, $1
# CHECK: lw   $1, 0($14)
    .set    at=$14
        lw      $1, 65536($1)
# CHECK: .set at=$15
# CHECK: lui  $15, 1
# CHECK: addu $15, $15, $1
# CHECK: lw   $1, 0($15)
    .set    at=$15
        lw      $1, 65536($1)
# CHECK: .set at=$16
# CHECK: lui  $16, 1
# CHECK: addu $16, $16, $1
# CHECK: lw   $1, 0($16)
    .set    at=$s0
        lw      $1, 65536($1)
# CHECK: .set at=$17
# CHECK: lui  $17, 1
# CHECK: addu $17, $17, $1
# CHECK: lw   $1, 0($17)
    .set    at=$s1
        lw      $1, 65536($1)
# CHECK: .set at=$18
# CHECK: lui  $18, 1
# CHECK: addu $18, $18, $1
# CHECK: lw   $1, 0($18)
    .set    at=$s2
        lw      $1, 65536($1)
# CHECK: .set at=$19
# CHECK: lui  $19, 1
# CHECK: addu $19, $19, $1
# CHECK: lw   $1, 0($19)
    .set    at=$s3
        lw      $1, 65536($1)
# CHECK: .set at=$20
# CHECK: lui  $20, 1
# CHECK: addu $20, $20, $1
# CHECK: lw   $1, 0($20)
    .set    at=$s4
        lw      $1, 65536($1)
# CHECK: .set at=$21
# CHECK: lui  $21, 1
# CHECK: addu $21, $21, $1
# CHECK: lw   $1, 0($21)
    .set    at=$s5
        lw      $1, 65536($1)
# CHECK: .set at=$22
# CHECK: lui  $22, 1
# CHECK: addu $22, $22, $1
# CHECK: lw   $1, 0($22)
    .set    at=$s6
        lw      $1, 65536($1)
# CHECK: .set at=$23
# CHECK: lui  $23, 1
# CHECK: addu $23, $23, $1
# CHECK: lw   $1, 0($23)
    .set    at=$s7
        lw      $1, 65536($1)
# CHECK: .set at=$24
# CHECK: lui  $24, 1
# CHECK: addu $24, $24, $1
# CHECK: lw   $1, 0($24)
    .set    at=$24
        lw      $1, 65536($1)
# CHECK: .set at=$25
# CHECK: lui  $25, 1
# CHECK: addu $25, $25, $1
# CHECK: lw   $1, 0($25)
    .set    at=$25
        lw      $1, 65536($1)
# CHECK: .set at=$26
# CHECK: lui  $26, 1
# CHECK: addu $26, $26, $1
# CHECK: lw   $1, 0($26)
    .set    at=$26
        lw      $1, 65536($1)
# CHECK: .set at=$27
# CHECK: lui  $27, 1
# CHECK: addu $27, $27, $1
# CHECK: lw   $1, 0($27)
    .set    at=$27
        lw      $1, 65536($1)
# CHECK: .set at=$28
# CHECK: lui  $gp, 1
# CHECK: addu $gp, $gp, $1
# CHECK: lw   $1, 0($gp)
    .set    at=$gp
        lw      $1, 65536($1)
# CHECK: .set at=$30
# CHECK: lui  $fp, 1
# CHECK: addu $fp, $fp, $1
# CHECK: lw   $1, 0($fp)
    .set    at=$fp
        lw      $1, 65536($1)
# CHECK: .set at=$29
# CHECK: lui  $sp, 1
# CHECK: addu $sp, $sp, $1
# CHECK: lw   $1, 0($sp)
    .set    at=$sp
        lw      $1, 65536($1)
# CHECK: .set at=$31
# CHECK: lui  $ra, 1
# CHECK: addu $ra, $ra, $1
# CHECK: lw   $1, 0($ra)
    .set    at=$ra
        lw      $1, 65536($1)
