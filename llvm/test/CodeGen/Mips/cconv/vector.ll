; RUN: llc < %s -march=mips -mcpu=mips32 -disable-mips-delay-filler | FileCheck %s --check-prefixes=ALL,MIPS32,MIPS32EB
; RUN: llc < %s -march=mips64 -relocation-model=pic -mcpu=mips64 -disable-mips-delay-filler | FileCheck %s --check-prefixes=ALL,MIPS64,MIPS64EB
; RUN: llc < %s -march=mips -mcpu=mips32r5 -mattr=+fp64,+msa -disable-mips-delay-filler | FileCheck %s --check-prefixes=ALL,MIPS32R5,MIPS32R5EB
; RUN: llc < %s -march=mips64 -relocation-model=pic -mcpu=mips64r5 -mattr=+fp64,+msa -disable-mips-delay-filler | FileCheck %s --check-prefixes=ALL,MIPS64R5
; RUN: llc < %s -march=mipsel -mcpu=mips32 -disable-mips-delay-filler | FileCheck %s --check-prefixes=ALL,MIPS32,MIPS32EL
; RUN: llc < %s -march=mips64el -relocation-model=pic -mcpu=mips64 -disable-mips-delay-filler | FileCheck %s --check-prefixes=ALL,MIPS64,MIPS64EL
; RUN: llc < %s -march=mipsel -mcpu=mips32r5 -mattr=+fp64,+msa -disable-mips-delay-filler | FileCheck %s --check-prefixes=ALL,MIPS32R5,MIPS32R5EL
; RUN: llc < %s -march=mips64el -relocation-model=pic -mcpu=mips64r5 -mattr=+fp64,+msa -disable-mips-delay-filler | FileCheck %s --check-prefixes=ALL,MIPS64R5



; Test that vector types are passed through the integer register set whether or
; not MSA is enabled. This is a ABI requirement for MIPS. For GCC compatibility
; we need to handle any power of 2 number of elements. We will test this
; exhaustively for combinations up to MSA register (128 bits) size.

; First set of tests are for argument passing.

define <2 x i8> @i8_2(<2 x i8> %a, <2 x i8> %b) {
; ALL-LABEL: i8_2:
; MIPS32EB-DAG: srl ${{[0-9]+}}, $5, 24
; MIPS32EB-DAG: srl ${{[0-9]+}}, $4, 24
; MIPS32EB-DAG: srl ${{[0-9]+}}, $5, 16
; MIPS32EB-DAG: srl ${{[0-9]+}}, $4, 16

; MIPS32EL: addu $1, $4, $5

; MIPS32R5-DAG: sw $4
; MIPS32R5-DAG: sw $5

; MIPS64EB-DAG: dsrl ${{[0-9]+}}, $5, 56
; MIPS64EB-DAG: dsrl ${{[0-9]+}}, $4, 56
; MIPS64EB-DAG: dsrl ${{[0-9]+}}, $5, 48
; MIPS64EB-DAG: dsrl ${{[0-9]+}}, $4, 48

; MIPS64EL-DAG: sll ${{[0-9]+}}, $4, 0
; MIPS64EL-DAG: sll ${{[0-9]+}}, $5, 0

; MIPS64R5-DAG: sd $4
; MIPS64R5-DAG: sd $5

  %1 = add <2 x i8> %a, %b
  ret <2 x i8> %1
}

; Test that vector spilled to the outgoing argument area have the expected
; offset from $sp.

define <2 x i8> @i8x2_7(<2 x i8> %a, <2 x i8> %b, <2 x i8> %c, <2 x i8> %d,
                        <2 x i8> %e, <2 x i8> %f, <2 x i8> %g) {
entry:

; MIPS32EB-DAG: srl ${{[0-9]+}}, $4, 24
; MIPS32EB-DAG: srl ${{[0-9]+}}, $5, 24
; MIPS32EB-DAG: srl ${{[0-9]+}}, $6, 24
; MIPS32EB-DAG: srl ${{[0-9]+}}, $7, 24

; MIPS32EL-DAG: andi ${{[0-9]+}}, $4, 65280
; MIPS32EL-DAG: andi ${{[0-9]+}}, $5, 65280
; MIPS32EL-DAG: andi ${{[0-9]+}}, $6, 65280
; MIPS32EL-DAG: andi ${{[0-9]+}}, $7, 65280

; MIPS32-DAG: lbu ${{[0-9]+}}, 16($sp)
; MIPS32-DAG; lbu ${{[0-9]+}}, 17($sp)
; MIPS32-DAG: lbu ${{[0-9]+}}, 20($sp)
; MIPS32-DAG: lbu ${{[0-9]+}}, 21($sp)
; MIPS32-DAG: lbu ${{[0-9]+}}, 24($sp)
; MIPS32-DAG: lbu ${{[0-9]+}}, 25($sp)

; MIPS32R5-DAG: sw $4, {{[0-9]+}}($sp)
; MIPS32R5-DAG: sw $5, {{[0-9]+}}($sp)
; MIPS32R5-DAG: sw $6, {{[0-9]+}}($sp)
; MIPS32R5-DAG: sw $7, {{[0-9]+}}($sp)

; MIPS32R5-DAG: lbu ${{[0-9]+}}, 40($sp)
; MIPS32R5-DAG: lbu ${{[0-9]+}}, 41($sp)
; MIPS32R5-DAG: lbu ${{[0-9]+}}, 42($sp)
; MIPS32R5-DAG: lbu ${{[0-9]+}}, 43($sp)
; MIPS32R5-DAG: lbu ${{[0-9]+}}, 44($sp)
; MIPS32R5-DAG: lbu ${{[0-9]+}}, 45($sp)
; MIPS32R5-DAG: lbu ${{[0-9]+}}, 46($sp)
; MIPS32R5-DAG: lbu ${{[0-9]+}}, 47($sp)
; MIPS32R5-DAG: lbu ${{[0-9]+}}, 48($sp)
; MIPS32R5-DAG: lbu ${{[0-9]+}}, 49($sp)
; MIPS32R5-DAG: lbu ${{[0-9]+}}, 50($sp)
; MIPS32R5-DAG: lbu ${{[0-9]+}}, 51($sp)

; MIPS64EB-DAG: dsrl ${{[0-9]+}}, $4, 48
; MIPS64EB-DAG: dsrl ${{[0-9]+}}, $5, 48
; MIPS64EB-DAG: dsrl ${{[0-9]+}}, $6, 48
; MIPS64EB-DAG: dsrl ${{[0-9]+}}, $7, 48
; MIPS64EB-DAG: dsrl ${{[0-9]+}}, $8, 48
; MIPS64EB-DAG: dsrl ${{[0-9]+}}, $9, 48
; MIPS64EB-DAG: dsrl ${{[0-9]+}}, $10, 48

; MIPS64R5-DAG: sd $4, {{[0-9]+}}($sp)
; MIPS64R5-DAG: sd $5, {{[0-9]+}}($sp)
; MIPS64R5-DAG: sd $6, {{[0-9]+}}($sp)
; MIPS64R5-DAG: sd $7, {{[0-9]+}}($sp)
; MIPS64R5-DAG: sd $8, {{[0-9]+}}($sp)
; MIPS64R5-DAG: sd $9, {{[0-9]+}}($sp)
; MIPS64R5-DAG: sd $10, {{[0-9]+}}($sp)

  %0 = add <2 x i8> %a, %b
  %1 = add <2 x i8> %0, %c
  %2 = add <2 x i8> %1, %d
  %3 = add <2 x i8> %2, %e
  %4 = add <2 x i8> %3, %f
  %5 = add <2 x i8> %4, %g
  ret <2 x i8> %5
}

define <4 x i8> @i8_4(<4 x i8> %a, <4 x i8> %b) {
; ALL-LABEL: i8_4:
; MIPS32-DAG: srl ${{[0-9]+}}, $5, 24
; MIPS32-DAG: srl ${{[0-9]+}}, $4, 24
; MIPS32-DAG: srl ${{[0-9]+}}, $5, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $4, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $5, 8
; MIPS32-DAG: srl ${{[0-9]+}}, $4, 8

; MIPS32R5-DAG: sw $4
; MIPS32R5-DAG: sw $5

; MIPS64-DAG: sll ${{[0-9]+}}, $4, 0
; MIPS64-DAG: sll ${{[0-9]+}}, $5, 0

; MIPS64R5-DAG: sll ${{[0-9]+}}, $4, 0
; MIPS64R5-DAG: sll ${{[0-9]+}}, $5, 0

  %1 = add <4 x i8> %a, %b
  ret <4 x i8> %1
}

define <8 x i8> @i8_8(<8 x i8> %a, <8 x i8> %b) {
; ALL-LABEL: i8_8:
; MIPS32-NOT: lw
; MIPS32-DAG: srl ${{[0-9]+}}, $7, 24
; MIPS32-DAG: srl ${{[0-9]+}}, $6, 24
; MIPS32-DAG: srl ${{[0-9]+}}, $7, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $6, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $7, 8
; MIPS32-DAG: srl ${{[0-9]+}}, $6, 8
; MIPS32-DAG: srl ${{[0-9]+}}, $5, 24
; MIPS32-DAG: srl ${{[0-9]+}}, $4, 24
; MIPS32-DAG: srl ${{[0-9]+}}, $5, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $4, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $5, 8
; MIPS32-DAG: srl ${{[0-9]+}}, $4, 8

; MIPS32R5-DAG: sw $4
; MIPS32R5-DAG: sw $5
; MIPS32R5-DAG: sw $6
; MIPS32R5-DAG: sw $7

; MIPS64-DAG: dsrl ${{[0-9]+}}, $5, 56
; MIPS64-DAG: dsrl ${{[0-9]+}}, $4, 56
; MIPS64-DAG: dsrl ${{[0-9]+}}, $5, 48
; MIPS64-DAG: dsrl ${{[0-9]+}}, $4, 48
; MIPS64-DAG: dsrl ${{[0-9]+}}, $5, 40
; MIPS64-DAG: dsrl ${{[0-9]+}}, $4, 40
; MIPS64-DAG: dsrl ${{[0-9]+}}, $5, 32
; MIPS64-DAG: dsrl ${{[0-9]+}}, $4, 32
; MIPS64-DAG: sll $[[R0:[0-9]+]], $4, 0
; MIPS64-DAG: sll $[[R1:[0-9]+]], $5, 0
; MIPS64-DAG: srl ${{[0-9]+}}, $[[R1]], 24
; MIPS64-DAG: srl ${{[0-9]+}}, $[[R0]], 24
; MIPS64-DAG: srl ${{[0-9]+}}, $[[R1]], 16
; MIPS64-DAG: srl ${{[0-9]+}}, $[[R0]], 16
; MIPS64-DAG: srl ${{[0-9]+}}, $[[R1]], 8
; MIPS64-DAG: srl ${{[0-9]+}}, $[[R0]], 8

; MIPS64R5-DAG: sd $4
; MIPS64R5-DAG: sd $5

  %1 = add <8 x i8> %a, %b
  ret <8 x i8> %1
}

define <16 x i8> @i8_16(<16 x i8> %a, <16 x i8> %b) {
; ALL-LABEL: i8_16:
; MIPS32-DAG: lw ${{[0-9]+}}, 16($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 20($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 24($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 28($sp)
; MIPS32-DAG: srl ${{[0-9]+}}, $7, 24
; MIPS32-DAG: srl ${{[0-9]+}}, $6, 24
; MIPS32-DAG: srl ${{[0-9]+}}, $7, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $6, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $7, 8
; MIPS32-DAG: srl ${{[0-9]+}}, $6, 8
; MIPS32-DAG: srl ${{[0-9]+}}, $5, 24
; MIPS32-DAG: srl ${{[0-9]+}}, $4, 24
; MIPS32-DAG: srl ${{[0-9]+}}, $5, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $4, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $5, 8
; MIPS32-DAG: srl ${{[0-9]+}}, $4, 8

; MIPS32R5-DAG: lw ${{[0-9]+}}, 16($sp)
; MIPS32R5-DAG: lw ${{[0-9]+}}, 20($sp)
; MIPS32R5-DAG: lw ${{[0-9]+}}, 24($sp)
; MIPS32R5-DAG: lw ${{[0-9]+}}, 28($sp)
; MIPS32R5-DAG: insert.w $w[[W0:[0-9]+]][0], $4
; MIPS32R5-DAG: insert.w $w[[W0]][1], $5
; MIPS32R5-DAG: insert.w $w[[W0]][2], $6
; MIPS32R5-DAG: insert.w $w[[W0]][3], $7

; MIPS64-DAG: dsrl ${{[0-9]+}}, $7, 56
; MIPS64-DAG: dsrl ${{[0-9]+}}, $6, 56
; MIPS64-DAG: dsrl ${{[0-9]+}}, $7, 48
; MIPS64-DAG: dsrl ${{[0-9]+}}, $6, 48
; MIPS64-DAG: dsrl ${{[0-9]+}}, $7, 40
; MIPS64-DAG: dsrl ${{[0-9]+}}, $6, 40
; MIPS64-DAG: dsrl ${{[0-9]+}}, $7, 32
; MIPS64-DAG: dsrl ${{[0-9]+}}, $6, 32
; MIPS64-DAG: dsrl ${{[0-9]+}}, $5, 56
; MIPS64-DAG: dsrl ${{[0-9]+}}, $4, 56
; MIPS64-DAG: dsrl ${{[0-9]+}}, $5, 48
; MIPS64-DAG: dsrl ${{[0-9]+}}, $4, 48
; MIPS64-DAG: dsrl ${{[0-9]+}}, $5, 32
; MIPS64-DAG: dsrl ${{[0-9]+}}, $4, 32

; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][{{[0-9]}}], $4
; MIPS64R5-DAG: insert.d $w[[W0]][{{[0-9]}}], $5
; MIPS64R5-DAG: insert.d $w[[W1:[0-9]+]][{{[0-9]}}], $6
; MIPS64R5-DAG: insert.d $w[[W1]][{{[0-9]}}], $7

  %1 = add <16 x i8> %a, %b

  ret <16 x i8> %1
}

define <2 x i16> @i16_2(<2 x i16> %a, <2 x i16> %b) {
; ALL-LABEL: i16_2:
; MIPS32: addu    $[[R0:[0-9]+]], $4, $5
; MIPS32: andi    $[[R1:[0-9]+]], $[[R0]], 65535
; MIPS32: srl     $[[R2:[0-9]+]], $5, 16
; MIPS32: srl     $[[R3:[0-9]+]], $4, 16
; MIPS32: addu    $[[R4:[0-9]+]], $[[R3]], $[[R2]]
; MIPS32: sll     $2, $[[R4]], 16

; MIPS32R5-DAG: sw $4
; MIPS32R5-DAG: sw $5

; MIPS64-DAG: sll ${{[0-9]+}}, $5, 0
; MIPS64-DAG: sll ${{[0-9]+}}, $4, 0

; MIPS64R5-DAG: sll ${{[0-9]+}}, $4, 0
; MIPS64R5-DAG: sll ${{[0-9]+}}, $5, 0

  %1 = add <2 x i16> %a, %b
  ret <2 x i16> %1
}

define <4 x i16> @i16_4(<4 x i16> %a, <4 x i16> %b) {
; ALL-LABEL: i16_4:
; MIPS32-DAG: srl ${{[0-9]+}}, $7, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $6, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $5, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $4, 16

; MIPS32R5-DAG: sw $4
; MIPS32R5-DAG: sw $5
; MIPS32R5-DAG: sw $6
; MIPS32R5-DAG: sw $7

; MIPS64-DAG: dsrl ${{[0-9]+}}, $4, 48
; MIPS64-DAG: dsrl ${{[0-9]+}}, $5, 48
; MIPS64-DAG: dsrl ${{[0-9]+}}, $4, 32
; MIPS64-DAG: dsrl ${{[0-9]+}}, $5, 32

; MIPS64R5-DAG: sd $4
; MIPS64R5-DAG: sd $5

  %1 = add <4 x i16> %a, %b
  ret <4 x i16> %1
}

define <8 x i16> @i16_8(<8 x i16> %a, <8 x i16> %b) {
; ALL-LABEL: i16_8:
; MIPS32-DAG: lw ${{[0-9]+}}, 16($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 20($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 24($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 28($sp)
; MIPS32-DAG: srl ${{[0-9]+}}, $7, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $6, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $5, 16
; MIPS32-DAG: srl ${{[0-9]+}}, $4, 16

; MIPS32R5-DAG: lw ${{[0-9]+}}, 16($sp)
; MIPS32R5-DAG: lw ${{[0-9]+}}, 20($sp)
; MIPS32R5-DAG: lw ${{[0-9]+}}, 24($sp)
; MIPS32R5-DAG: lw ${{[0-9]+}}, 28($sp)
; MIPS32R5-DAG: insert.w $w[[W0:[0-9]+]][0], $4
; MIPS32R5-DAG: insert.w $w[[W0]][1], $5
; MIPS32R5-DAG: insert.w $w[[W0]][2], $6
; MIPS32R5-DAG: insert.w $w[[W0]][3], $7

; MIPS64-DAG: dsrl ${{[0-9]+}}, $6, 48
; MIPS64-DAG: dsrl ${{[0-9]+}}, $7, 48
; MIPS64-DAG: dsrl ${{[0-9]+}}, $6, 32
; MIPS64-DAG: dsrl ${{[0-9]+}}, $7, 32
; MIPS64-DAG: dsrl ${{[0-9]+}}, $4, 48
; MIPS64-DAG: dsrl ${{[0-9]+}}, $5, 48
; MIPS64-DAG: dsrl ${{[0-9]+}}, $4, 32
; MIPS64-DAG: dsrl ${{[0-9]+}}, $5, 32

; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][{{[0-9]}}], $4
; MIPS64R5-DAG: insert.d $w[[W0]][{{[0-9]}}], $5
; MIPS64R5-DAG: insert.d $w[[W1:[0-9]+]][{{[0-9]}}], $6
; MIPS64R5-DAG: insert.d $w[[W1]][{{[0-9]}}], $7

  %1 = add <8 x i16> %a, %b
  ret <8 x i16> %1
}

define <2 x i32> @i32_2(<2 x i32> %a, <2 x i32> %b) {
; ALL-LABEL: i32_2:
; MIPS32-DAG: addu    $2, $4, $6
; MIPS32-DAG: addu    $3, $5, $7

; MIPS32R5-DAG: sw $4
; MIPS32R5-DAG: sw $5
; MIPS32R5-DAG: sw $6
; MIPS32R5-DAG: sw $7

; MIPS64-DAG: sll     ${{[0-9]+}}, $4, 0
; MIPS64-DAG: sll     ${{[0-9]+}}, $5, 0

; MIPS64R5-DAG: sd $4
; MIPS64R5-DAG: sd $5

  %1 = add <2 x i32> %a, %b

  ret <2 x i32> %1
}

define <4 x i32> @i32_4(<4 x i32> %a, <4 x i32> %b) {
; ALL-LABEL: i32_4:
; MIPS32-DAG: lw ${{[0-9]+}}, 16($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 20($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 24($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 28($sp)
; MIPS32-DAG: addu $2
; MIPS32-DAG: addu $3
; MIPS32-DAG: addu $4
; MIPS32-DAG: addu $5

; MIPS32R5-DAG: lw ${{[0-9]+}}, 16($sp)
; MIPS32R5-DAG: lw ${{[0-9]+}}, 20($sp)
; MIPS32R5-DAG: lw ${{[0-9]+}}, 24($sp)
; MIPS32R5-DAG: lw ${{[0-9]+}}, 28($sp)
; MIPS32R5-DAG: insert.w $w[[W0:[0-9]+]][0], $4
; MIPS32R5-DAG: insert.w $w[[W0]][1], $5
; MIPS32R5-DAG: insert.w $w[[W0]][2], $6
; MIPS32R5-DAG: insert.w $w[[W0]][3], $7

; MIPS64-DAG: sll     ${{[0-9]+}}, $4, 0
; MIPS64-DAG: sll     ${{[0-9]+}}, $5, 0
; MIPS64-DAG: sll     ${{[0-9]+}}, $6, 0
; MIPS64-DAG: sll     ${{[0-9]+}}, $7, 0
; MIPS64-DAG: dsrl    ${{[0-9]+}}, $4, 32
; MIPS64-DAG: dsrl    ${{[0-9]+}}, $5, 32
; MIPS64-DAG: dsrl    ${{[0-9]+}}, $6, 32
; MIPS64-DAG: dsrl    ${{[0-9]+}}, $7, 32
  %1 = add <4 x i32> %a, %b
  ret <4 x i32> %1
}

define <2 x i64> @i64_2(<2 x i64> %a, <2 x i64> %b) {
; ALL-LABEL: i64_2:
; MIPS32-DAG: lw ${{[0-9]+}}, 16($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 20($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 24($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 28($sp)
; MIPS32-DAG: addu $2
; MIPS32-DAG: addu $3
; MIPS32-DAG: addu $4
; MIPS32-DAG: addu $5

; MIPS32R5-DAG: lw ${{[0-9]+}}, 16($sp)
; MIPS32R5-DAG: lw ${{[0-9]+}}, 20($sp)
; MIPS32R5-DAG: lw ${{[0-9]+}}, 24($sp)
; MIPS32R5-DAG: lw ${{[0-9]+}}, 28($sp)
; MIPS32R5-DAG: insert.w $w[[W0:[0-9]+]][0], $4
; MIPS32R5-DAG: insert.w $w[[W0]][1], $5
; MIPS32R5-DAG: insert.w $w[[W0]][2], $6
; MIPS32R5-DAG: insert.w $w[[W0]][3], $7

; MIPS64-DAG: daddu $2, $4, $6
; MIPS64-DAG: daddu $3, $5, $7

; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][{{[0-9]}}], $4
; MIPS64R5-DAG: insert.d $w[[W0]][{{[0-9]}}], $5
; MIPS64R5-DAG: insert.d $w[[W1:[0-9]+]][{{[0-9]}}], $6
; MIPS64R5-DAG: insert.d $w[[W1]][{{[0-9]}}], $7

  %1 = add <2 x i64> %a, %b
  ret <2 x i64> %1
}

; The MIPS vector ABI treats vectors of floats differently to vectors of
; integers.

; For arguments floating pointer vectors are bitcasted to integer vectors whose
; elements are of GPR width and where the element count is deduced from
; the length of the floating point vector divided by the size of the GPRs.

; For returns, integer vectors are passed via the GPR register set, but
; floating point vectors are returned via a hidden sret pointer.

; For testing purposes we skip returning values here and test them below
; instead.
@float_res_v2f32 = external global <2 x float>

define void @float_2(<2 x float> %a, <2 x float> %b) {
; ALL-LABEL: float_2:
; MIPS32: mtc1 $7, $f[[F0:[0-9]+]]
; MIPS32: mtc1 $5, $f[[F1:[0-9]+]]
; MIPS32: add.s $f[[F2:[0-9]+]], $f[[F1]], $f[[F0]]
; MIPS32: swc1 $f[[F2]]
; MIPS32: mtc1 $6, $f[[F3:[0-9]+]]
; MIPS32: mtc1 $4, $f[[F4:[0-9]+]]
; MIPS32: add.s $f[[F5:[0-9]+]], $f[[F4]], $f[[F3]]
; MIPS32: swc1 $f[[F5]]

; MIPS32R5-DAG: sw $4
; MIPS32R5-DAG: sw $5
; MIPS32R5-DAG: sw $6
; MIPS32R5-DAG: sw $7

; MIPS64-DAG: sll $[[R0:[0-9]+]], $4, 0
; MIPS64-DAG: sll $[[R1:[0-9]+]], $5, 0
; MIPS64-DAG: mtc1 $[[R0]], $f{{[0-9]+}}
; MIPS64-DAG: mtc1 $[[R1]], $f{{[0-9]+}}
; MIPS64-DAG: dsrl $[[R2:[0-9]+]], $4, 32
; MIPS64-DAG: dsrl $[[R3:[0-9]+]], $5, 32
; MIPS64-DAG: sll $[[R4:[0-9]+]], $[[R2]], 0
; MIPS64-DAG: sll $[[R5:[0-9]+]], $[[R3]], 0
; MIPS64-DAG: mtc1 $[[R4]], $f{{[0-9]+}}
; MIPS64-DAG: mtc1 $[[R5]], $f{{[0-9]+}}

; MIPS64R5-DAG: sd $4
; MIPS64R5-DAG: sd $5

  %1 = fadd <2 x float> %a, %b
  store <2 x float> %1, <2 x float> * @float_res_v2f32
  ret void
}

@float_res_v4f32 = external global <4 x float>

; For MSA this case is suboptimal, the 4 loads can be combined into a single
; ld.w.

define void @float_4(<4 x float> %a, <4 x float> %b) {
; ALL-LABEL: float_4:
; MIPS32-DAG: mtc1 $4
; MIPS32-DAG: mtc1 $5
; MIPS32-DAG: mtc1 $6
; MIPS32-DAG: mtc1 $7
; MIPS32-DAG: lwc1
; MIPS32-DAG: lwc1
; MIPS32-DAG: lwc1
; MIPS32-DAG: lwc1

; MIPS32R5-DAG: lw $[[R1:[0-9]+]], 16($sp)
; MIPS32R5-DAG: insert.w $w[[W0:[0-9]+]][0], $[[R1]]
; MIPS32R5-DAG: lw $[[R2:[0-9]+]], 20($sp)
; MIPS32R5-DAG: insert.w $w[[W0]][1], $[[R2]]
; MIPS32R5-DAG: lw $[[R3:[0-9]+]], 24($sp)
; MIPS32R5-DAG: insert.w $w[[W0]][2], $[[R3]]
; MIPS32R5-DAG: lw $[[R4:[0-9]+]], 28($sp)
; MIPS32R5-DAG: insert.w $w[[W0]][3], $[[R4]]

; MIPS32R5-DAG: insert.w $w[[W1:[0-9]+]][0], $4
; MIPS32R5-DAG: insert.w $w[[W1]][1], $5
; MIPS32R5-DAG: insert.w $w[[W1]][2], $6
; MIPS32R5-DAG: insert.w $w[[W1]][3], $7

; MIPS64-DAG: sll $[[R0:[0-9]+]], $4, 0
; MIPS64-DAG: sll $[[R1:[0-9]+]], $5, 0
; MIPS64-DAG: mtc1 $[[R0]], $f{{[0-9]+}}
; MIPS64-DAG: mtc1 $[[R1]], $f{{[0-9]+}}
; MIPS64-DAG: dsrl $[[R2:[0-9]+]], $4, 32
; MIPS64-DAG: dsrl $[[R3:[0-9]+]], $5, 32
; MIPS64-DAG: sll $[[R4:[0-9]+]], $[[R2]], 0
; MIPS64-DAG: sll $[[R5:[0-9]+]], $[[R3]], 0
; MIPS64-DAG: mtc1 $[[R4]], $f{{[0-9]+}}
; MIPS64-DAG: mtc1 $[[R5]], $f{{[0-9]+}}
; MIPS64-DAG: sll $[[R6:[0-9]+]], $6, 0
; MIPS64-DAG: sll $[[R7:[0-9]+]], $7, 0
; MIPS64-DAG: mtc1 $[[R6]], $f{{[0-9]+}}
; MIPS64-DAG: mtc1 $[[R7]], $f{{[0-9]+}}
; MIPS64-DAG: dsrl $[[R8:[0-9]+]], $6, 32
; MIPS64-DAG: dsrl $[[R9:[0-9]+]], $7, 32
; MIPS64-DAG: sll $[[R10:[0-9]+]], $[[R8]], 0
; MIPS64-DAG: sll $[[R11:[0-9]+]], $[[R9]], 0
; MIPS64-DAG: mtc1 $[[R10]], $f{{[0-9]+}}
; MIPS64-DAG: mtc1 $[[R11]], $f{{[0-9]+}}

; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][{{[0-9]}}], $4
; MIPS64R5-DAG: insert.d $w[[W0]][{{[0-9]}}], $5
; MIPS64R5-DAG: insert.d $w[[W1:[0-9]+]][{{[0-9]}}], $6
; MIPS64R5-DAG: insert.d $w[[W1]][{{[0-9]}}], $7

  %1 = fadd <4 x float> %a, %b
  store <4 x float> %1, <4 x float> * @float_res_v4f32
  ret void
}

@double_v2f64 = external global <2 x double>

define void @double_2(<2 x double> %a, <2 x double> %b) {
; ALL-LABEL: double_2:
; MIPS32-DAG: sw $7
; MIPS32-DAG: sw $6
; MIPS32-DAG: ldc1
; MIPS32-DAG: ldc1
; MIPS32:     add.d
; MIPS32-DAG: sw $5
; MIPS32-DAG: sw $4
; MIPS32-DAG: ldc1
; MIPS32-DAG: ldc1
; MIPS32:     add.d

; MIPS32R5-DAG: lw $[[R1:[0-9]+]], 16($sp)
; MIPS32R5-DAG: insert.w $w[[W0:[0-9]+]][0], $[[R1]]
; MIPS32R5-DAG: lw $[[R2:[0-9]+]], 20($sp)
; MIPS32R5-DAG: insert.w $w[[W0]][1], $[[R2]]
; MIPS32R5-DAG: lw $[[R3:[0-9]+]], 24($sp)
; MIPS32R5-DAG: insert.w $w[[W0]][2], $[[R3]]
; MIPS32R5-DAG: lw $[[R4:[0-9]+]], 28($sp)
; MIPS32R5-DAG: insert.w $w[[W0]][3], $[[R4]]

; MIPS32R5-DAG: insert.w $w[[W1:[0-9]+]][0], $4
; MIPS32R5-DAG: insert.w $w[[W1]][1], $5
; MIPS32R5-DAG: insert.w $w[[W1]][2], $6
; MIPS32R5-DAG: insert.w $w[[W1]][3], $7

; MIPS64-DAG: dmtc1 $6, $f[[R0:[0-9]+]]
; MIPS64-DAG: dmtc1 $4, $f[[R1:[0-9]+]]
; MIPS64-DAG: add.d $f[[R2:[0-9]+]], $f[[R1]], $f[[R0]]
; MIPS64-DAG: dmtc1 $7, $f[[R3:[0-9]+]]
; MIPS64-DAG: dmtc1 $5, $f[[R4:[0-9]+]]
; MIPS64-DAG: add.d $f[[R5:[0-9]+]], $f[[R4]], $f[[R3]]

; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][{{[0-9]}}], $4
; MIPS64R5-DAG: insert.d $w[[W0]][{{[0-9]}}], $5
; MIPS64R5-DAG: insert.d $w[[W1:[0-9]+]][{{[0-9]}}], $6
; MIPS64R5-DAG: insert.d $w[[W1]][{{[0-9]}}], $7

  %1 = fadd <2 x double> %a, %b
  store <2 x double> %1, <2 x double> * @double_v2f64
  ret void
}

; Return value testing.
; Integer vectors are returned in $2, $3, $4, $5 for O32, $2, $3 for N32/N64
; Floating point vectors are returned through a hidden sret pointer.

@gv2i8 = global <2 x i8> <i8 1, i8 2>
@gv4i8 = global <4 x i8> <i8 0, i8 1, i8 2, i8 3>
@gv8i8 = global <8 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>
@gv16i8 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>

@gv2i16 = global <2 x i16> <i16 1, i16 2>
@gv4i16 = global <4 x i16> <i16 0, i16 1, i16 2, i16 3>
@gv8i16 = global <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>

@gv2i32 = global <2 x i32> <i32 0, i32 1>
@gv4i32 = global <4 x i32> <i32 0, i32 1, i32 2, i32 3>

@gv2i64 = global <2 x i64> <i64 0, i64 1>

define <2 x i8> @ret_2_i8() {
; ALL-LABEL: ret_2_i8:
; MIPS32-DAG:   lhu $2
; MIPS32R5-DAG: lhu $2

; FIXME: why is this lh instead of lhu on mips64?

; MIPS64-DAG:  lh $2
; MIPS64-DAG:  lh $2
  %1 = load <2 x i8>, <2 x i8> * @gv2i8
  ret <2 x i8> %1
}

define <4 x i8> @ret_4_i8() {
; ALL-LABEL: ret_4_i8:
; MIPS32-DAG:   lw $2
; MIPS32R5-DAG: lw $2

; MIPS64-DAG:   lw $2
; MIPS64R5-DAG: lw $2

  %1 = load <4 x i8>, <4 x i8> * @gv4i8
  ret <4 x i8> %1
}

define <8 x i8> @ret_8_i8() {
; ALL-LABEL: ret_8_i8:
; MIPS32-DAG:   lw $2
; MIPS32-DAG:   lw $3

; MIPS32R5: copy_s.w $2, $w[[W0:[0-9]+]]
; MIPS32R5: copy_s.w $3, $w[[W0]]

; MIPS64-DAG:   ld $2
; MIPS64R5-DAG: ld $2
  %1 = load <8 x i8>, <8 x i8> * @gv8i8
  ret <8 x i8> %1
}

define <16 x i8> @ret_16_i8() {
; ALL-LABEL: ret_16_i8:
; MIPS32-DAG: lw $2
; MIPS32-DAG: lw $3
; MIPS32-DAG: lw $4
; MIPS32-DAG: lw $5

; MIPS32R5-DAG: copy_s.w $2, $w[[W0:[0-9]+]][0]
; MIPS32R5-DAG: copy_s.w $3, $w[[W0]][1]
; MIPS32R5-DAG: copy_s.w $4, $w[[W0]][2]
; MIPS32R5-DAG: copy_s.w $5, $w[[W0]][3]

; MIPS64-DAG: ld $2
; MIPS64-DAG: ld $3

; MIPS64R5-DAG: copy_s.d $2
; MIPS64R5-DAG: copy_s.d $3

  %1 = load <16 x i8>, <16 x i8> * @gv16i8
  ret <16 x i8> %1
}

define <2 x i16> @ret_2_i16() {
; ALL-LABEL: ret_2_i16:
; MIPS32-DAG:   lw $2

; MIPS32R5-DAG: lw $2

; MIPS64-DAG:   lw $2

; MIPS64R5-DAG: lw $2
  %1 = load <2 x i16>, <2 x i16> * @gv2i16
  ret <2 x i16> %1
}

define <4 x i16> @ret_4_i16() {
; ALL-LABEL: ret_4_i16:
; MIPS32-DAG: lw $2
; MIPS32-DAG: lw $3

; MIPS32R5-DAG: copy_s.w $2, $w[[W0:[0-9]+]]
; MIPS32R5-DAG: copy_s.w $3, $w[[W0]]

; MIPS64-DAG:   ld $2
; MIPS64R5-DAG: ld $2
  %1 = load <4 x i16>, <4 x i16> * @gv4i16
  ret <4 x i16> %1
}

define <8 x i16> @ret_8_i16() {
; ALL-LABEL: ret_8_i16:
; MIPS32-DAG: lw $2
; MIPS32-DAG: lw $3
; MIPS32-DAG: lw $4
; MIPS32-DAG: lw $5

; MIPS32R5-DAG: copy_s.w $2, $w[[W0:[0-9]+]][0]
; MIPS32R5-DAG: copy_s.w $3, $w[[W0]][1]
; MIPS32R5-DAG: copy_s.w $4, $w[[W0]][2]
; MIPS32R5-DAG: copy_s.w $5, $w[[W0]][3]

; MIPS64-DAG: ld $2
; MIPS64-DAG: ld $3

; MIPS64R5-DAG: copy_s.d $2
; MIPS64R5-DAG: copy_s.d $3

  %1 = load <8 x i16>, <8 x i16> * @gv8i16
  ret <8 x i16> %1
}

define <2 x i32> @ret_2_i32() {
; ALL-LABEL: ret_2_i32:
; MIPS32-DAG: lw $2
; MIPS32-DAG: lw $3

; MIPS32R5-DAG: copy_s.w $2, $w[[W0:[0-9]+]]
; MIPS32R5-DAG: copy_s.w $3, $w[[W0]]

; MIPS64-DAG:   ld $2
; MIPS64R5-DAG: ld $2

  %1 = load <2 x i32>, <2 x i32> * @gv2i32
  ret <2 x i32> %1
}

define <4 x i32> @ret_4_i32() {
; ALL-LABEL: ret_4_i32:
; MIPS32-DAG: lw $2
; MIPS32-DAG: lw $3
; MIPS32-DAG: lw $4
; MIPS32-DAG: lw $5

; MIPS32R5-DAG: copy_s.w $2, $w[[W0:[0-9]+]][0]
; MIPS32R5-DAG: copy_s.w $3, $w[[W0]][1]
; MIPS32R5-DAG: copy_s.w $4, $w[[W0]][2]
; MIPS32R5-DAG: copy_s.w $5, $w[[W0]][3]

; MIPS64-DAG: ld $2
; MIPS64-DAG: ld $3

; MIPS64R5-DAG: copy_s.d $2, $w[[W0:[0-9]+]]
; MIPS64R5-DAG: copy_s.d $3, $w[[W0]]

  %1 = load <4 x i32>, <4 x i32> * @gv4i32
  ret <4 x i32> %1
}

define <2 x i64> @ret_2_i64() {
; ALL-LABEL: ret_2_i64:
; MIPS32-DAG: lw $2
; MIPS32-DAG: lw $3
; MIPS32-DAG: lw $4
; MIPS32-DAG: lw $5

; MIPS32R5-DAG: copy_s.w $2, $w[[W0:[0-9]+]][0]
; MIPS32R5-DAG: copy_s.w $3, $w[[W0]][1]
; MIPS32R5-DAG: copy_s.w $4, $w[[W0]][2]
; MIPS32R5-DAG: copy_s.w $5, $w[[W0]][3]

; MIPS64-DAG: ld $2
; MIPS64-DAG: ld $3

; MIPS64R5-DAG: copy_s.d $2, $w[[W0:[0-9]+]]
; MIPS64R5-DAG: copy_s.d $3, $w[[W0]]

  %1 = load <2 x i64>, <2 x i64> * @gv2i64
  ret <2 x i64> %1
}

@gv2f32 = global <2 x float> <float 0.0, float 0.0>
@gv4f32 = global <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>

define <2 x float> @ret_float_2() {
entry:
; ALL-LABEL: ret_float_2:

; MIPS32-DAG: swc1 $f{{[0-9]+}}, 0($4)
; MIPS32-DAG: swc1 $f{{[0-9]+}}, 4($4)

; MIPS32R5-DAG: swc1 $f{{[0-9]+}}, 0($4)
; MIPS32R5-DAG: swc1 $f{{[0-9]+}}, 4($4)

; MIPS64: ld $2

; MIPS64R5: ld $2

  %0 = load <2 x float>, <2 x float> * @gv2f32
  ret <2 x float> %0
}

define <4 x float> @ret_float_4() {
entry:
; ALL-LABEL: ret_float_4:

; MIPS32-DAG: swc1 $f{{[0-9]+}}, 0($4)
; MIPS32-DAG: swc1 $f{{[0-9]+}}, 4($4)
; MIPS32-DAG: swc1 $f{{[0-9]+}}, 8($4)
; MIPS32-DAG: swc1 $f{{[0-9]+}}, 12($4)

; MIPS32R5: st.w $w{{[0-9]+}}, 0($4)

; MIPS64-DAG: ld $2
; MIPS64-DAG: ld $3

; MIPS64R5-DAG: copy_s.d $2, $w{{[0-9]+}}[0]
; MIPS64R5-DAG: copy_s.d $3, $w{{[0-9]+}}[1]

  %0 = load <4 x float>, <4 x float> * @gv4f32
  ret <4 x float> %0
}

@gv2f64 = global <2 x double> <double 0.0, double 0.0>

define <2 x double> @ret_double_2() {
entry:
; ALL-LABEL: ret_double_2:

; MIPS32-DAG: sdc1 $f{{[0-9]+}}, 8($4)
; MIPS32-DAG: sdc1 $f{{[0-9]+}}, 0($4)

; MIPS32R5: st.d $w{{[0-9]+}}, 0($4)

; MIPS64-DAG: ld $2
; MIPS64-DAG: ld $2

; MIPS64R5-DAG: copy_s.d $2, $w{{[0-9]+}}[0]
; MIPS64R5-DAG: copy_s.d $3, $w{{[0-9]+}}[1]

  %0 = load <2 x double>, <2 x double> * @gv2f64
  ret <2 x double> %0
}

; Test argument lowering and call result lowering.

define void @call_i8_2() {
entry:
; ALL-LABEL: call_i8_2:
; MIPS32EB-DAG: addiu $4
; MIPS32EB-DAG: addiu $5
; MIPS32-NOT: addiu $6
; MIPS32-NOT: addiu $7

; MIPS32R5-DAG: lhu $4, {{[0-9]+}}($sp)
; MIPS32R5-DAG: lhu $5, {{[0-9]+}}($sp)

; MIPS32R5: jal
; MIPS32R5: sw $2, {{[0-9]+}}($sp)

; MIPS32R5-DAG; sh ${{[0-9]+}}, %lo(gv2i8)(${{[0-9]+}})

; MIPS32R5-NOT: sb ${{[0-9]+}}, 1(${{[0-9]+}})
; MIPS32R5-NOT; sb ${{[0-9]+}}, %lo(gv2i8)(${{[0-9]+}})

; MIPS64EB: daddiu $4, $zero, 1543
; MIPS64EB: daddiu $5, $zero, 3080

; MIPS64EL: daddiu $4, $zero, 1798
; MIPS64EL; daddiu $5, $zero, 2060

; MIPS64R5-DAG: lh $4
; MIPS64R5-DAG: lh $5

; MIPS32: jal i8_2
; MIPS64: jalr $25

; MIPS32EB-DAG: srl $[[R0:[0-9]+]], $2, 16
; MIPS32EB-DAG: sb $[[R0]]
; MIPS32EB-DAG: srl $[[R1:[0-9]+]], $2, 24
; MIPS32EB-DAG: sb $[[R1]]

; MIPS32EL: sb $2
; MIPS32EL: srl $[[R0:[0-9]+]], $2, 8
; MIPS32EL: sb $[[R0]]

; MIPS64EB: dsrl $[[R4:[0-9]+]], $2, 48
; MIPS64EB: sb $[[R4]]
; MIPS64EB: dsrl $[[R5:[0-9]+]], $2, 56
; MIPS64EB: sb $[[R5]]

; MIPS64EL: sll $[[R6:[0-9]+]], $2, 0
; MIPS64EL: sb $[[R6]]
; MIPS64EL: srl $[[R7:[0-9]+]], $[[R6]], 8
; MIPS64EL: sb $[[R7]]

; MIPS64R5: sd $2

  %0 = call <2 x i8> @i8_2(<2 x i8> <i8 6, i8 7>, <2 x i8> <i8 12, i8 8>)
  store <2 x i8> %0, <2 x i8> * @gv2i8
  ret void
}

define void @call_i8_4() {
entry:
; ALL-LABEL: call_i8_4:
; MIPS32: ori $4
; MIPS32: ori $5
; MIPS32-NOT: ori $6
; MIPS32-NOT: ori $7

; MIPS32R5-NOT: lw $4, {{[0-9]+}}($sp)
; MIPS32R5-NOT: lw $5, {{[0-9]+}}($sp)

; MIPS64: ori $4
; MIPS64: ori $5

; MIPS64R5-NOT: lw $4
; MIPS64R5-NOT: lw $5

; MIPS32: jal i8_4
; MIPS64: jalr $25

; MIPS32: sw $2

; MIPS32R5-DAG: sw $2

; MIPS64: sw $2
; MIPS64R5: sw $2

  %0 = call <4 x i8> @i8_4(<4 x i8> <i8 6, i8 7, i8 9, i8 10>, <4 x i8> <i8 12, i8 8, i8 9, i8 10>)
  store <4 x i8> %0, <4 x i8> * @gv4i8
  ret void
}

define void @call_i8_8() {
entry:
; ALL-LABEL: call_i8_8:

; MIPS32: ori $6
; MIPS32: ori $4
; MIPS32: move  $5
; MIPS32: move  $7

; MIPS32R5-DAG: ori $6
; MIPS32R5-DAG: ori $4
; MIPS32R5-DAG: move  $5
; MIPS32R5-DAG: move  $7

; MIPS64EB: daddiu $4, ${{[0-9]+}}, 2314
; MIPS64EB: daddiu $5, ${{[0-9]+}}, 2314

; MIPS64EL: daddiu $4, ${{[0-9]+}}, 1798
; MIPS64EL: daddiu $5, ${{[0-9]+}}, 2060

; MIPS32: jal i8_8
; MIPS64: jalr $25

; MIPS32-DAG: sw $2
; MIPS32-DAG: sw $3

; MIPS32R5-DAG: sw $2
; MIPS32R5-DAG: sw $3

; MIPS64: sd $2
; MIPS64R5: sd $2

  %0 = call <8 x i8> @i8_8(<8 x i8> <i8 6, i8 7, i8 9, i8 10, i8 6, i8 7, i8 9, i8 10>, <8 x i8> <i8 12, i8 8, i8 9, i8 10, i8 6, i8 7, i8 9, i8 10>)
  store <8 x i8> %0, <8 x i8> * @gv8i8
  ret void
}

define void @calli8_16() {
entry:
; ALL-LABEL: calli8_16:
; MIPS32-DAG: sw  ${{[0-9]+}}, 28($sp)
; MIPS32-DAG: sw  ${{[0-9]+}}, 24($sp)
; MIPS32-DAG: sw  ${{[0-9]+}}, 20($sp)
; MIPS32-DAG: sw  ${{[0-9]+}}, 16($sp)

; MIPS32: ori $4, ${{[0-9]+}}, {{[0-9]+}}
; MIPS32: ori $7, ${{[0-9]+}}, {{[0-9]+}}
; MIPS32: move  $5, ${{[0-9]+}}
; MIPS32: move  $6, ${{[0-9]+}}

; MIPS32R5-DAG: copy_s.w $4, $w{{[0-9]+}}
; MIPS32R5-DAG: copy_s.w $5, $w{{[0-9]+}}
; MIPS32R5-DAG: copy_s.w $6, $w{{[0-9]+}}
; MIPS32R5-DAG: copy_s.w $7, $w{{[0-9]+}}

; MIPS32R5-DAG: sw  ${{[0-9]+}}, 28($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 24($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 20($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 16($sp)

; MIPS64-DAG: daddiu $4
; MIPS64-DAG: daddiu $5
; MIPS64-DAG: daddiu $6
; MIPS64-DAG: daddiu $7

; MIPS64R5-DAG: copy_s.d $4
; MIPS64R5-DAG: copy_s.d $5
; MIPS64R5-DAG: copy_s.d $6
; MIPS64R5-DAG: copy_s.d $7

; MIPS32: jal i8_16
; MIPS64: jalr $25

; MIPS32-DAG: sw $5, 12(${{[0-9]+}})
; MIPS32-DAG: sw $4, 8(${{[0-9]+}})
; MIPS32-DAG: sw $3, 4(${{[0-9]+}})
; MIPS32-DAG: sw $2, %lo(gv16i8)(${{[0-9]+}})

; MIPS32R5-DAG: insert.w $w[[W0:[0-9]+]][0], $2
; MIPS32R5-DAG: insert.w $w[[W0]][1], $3
; MIPS32R5-DAG: insert.w $w[[W0]][2], $4
; MIPS32R5-DAG: insert.w $w[[W0]][3], $5
; MIPS32R5-DAG: st.w $w[[W0]]

; MIPS64-DAG: sd $3
; MIPS64-DAG: sd $2

; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][0], $2
; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][1], $3

  %0 = call <16 x i8> @i8_16(<16 x i8> <i8 6, i8 7,i8 6, i8 7,i8 6, i8 7,i8 6, i8 7,i8 6, i8 7,i8 6, i8 7, i8 6, i8 7, i8 9, i8 10>, <16 x i8> <i8 7, i8 9,i8 7, i8 9,i8 7, i8 9,i8 7, i8 9,i8 7, i8 9,i8 7, i8 9,i8 12, i8 8, i8 9, i8 10>)
  store <16 x i8> %0, <16 x i8> * @gv16i8
  ret void
}

define void @calli16_2() {
entry:
; ALL-LABEL: calli16_2:

; MIPS32-DAG: ori $4
; MIPS32-DAG: ori $5

; MIPS32R5-NOT: lw $4
; MIPS32R5-NOT: lw $5

; MIPS64: ori $4
; MIPS64: ori $5

; MIPS64R5-NOT: lw $4
; MIPS64R5-NOT: lw $5

; MIPS32: jal i16_2
; MIPS64: jalr $25

; MIPS32: sw $2, %lo(gv2i16)

; MIPS32R5: sw $2, %lo(gv2i16)

; MIPS64: sw $2

; MIPS64R6: sw $2

  %0 = call <2 x i16> @i16_2(<2 x i16> <i16 6, i16 7>, <2 x i16> <i16 12, i16 8>)
  store <2 x i16> %0, <2 x i16> * @gv2i16
  ret void
}

define void @calli16_4() {
entry:
; ALL-LABEL: calli16_4:
; MIPS32-DAG: ori $4
; MIPS32-DAG: ori $5
; MIPS32-DAG: ori $6
; MIPS32-DAG: move $7

; MIPS32R5-DAG: ori $4
; MIPS32R5-DAG: ori $5
; MIPS32R5-DAG: ori $6
; MIPS32R5-DAG: move $7

; MIPS64-DAG: daddiu $4
; MIPS64-DAG: daddiu $5

; MIPS64R5-NOT: ld $4
; MIPS64R5-NOT: ld $5

; MIPS32: jal i16_4
; MIPS64: jalr $25

; MIPS32-DAG: sw $3, 4(${{[0-9]+}})
; MIPS32-DAG: sw $2, %lo(gv4i16)(${{[0-9]+}})

; MIPS32R5-DAG: sw $3, 4(${{[0-9]+}})
; MIPS32R5-DAG: sw $2, %lo(gv4i16)(${{[0-9]+}})

; MIPS64: sd $2
; MIPS64R5: sd $2

  %0 = call <4 x i16> @i16_4(<4 x i16> <i16 6, i16 7, i16 9, i16 10>, <4 x i16> <i16 12, i16 8, i16 9, i16 10>)
  store <4 x i16> %0, <4 x i16> * @gv4i16
  ret void
}

define void @calli16_8() {
entry:
; ALL-LABEL: calli16_8:

; MIPS32-DAG: sw  ${{[0-9]+}}, 28($sp)
; MIPS32-DAG: sw  ${{[0-9]+}}, 24($sp)
; MIPS32-DAG: sw  ${{[0-9]+}}, 20($sp)
; MIPS32-DAG: sw  ${{[0-9]+}}, 16($sp)

; MIPS32-DAG: ori $4, ${{[0-9]+}}, {{[0-9]+}}
; MIPS32-DAG: ori $5, ${{[0-9]+}}, {{[0-9]+}}
; MIPS32-DAG: move  $6, ${{[0-9]+}}
; MIPS32-DAG: move  $7, ${{[0-9]+}}

; MIPS32R5-DAG: copy_s.w $4, $w{{[0-9]+}}
; MIPS32R5-DAG: copy_s.w $5, $w{{[0-9]+}}
; MIPS32R5-DAG: copy_s.w $6, $w{{[0-9]+}}
; MIPS32R5-DAG: copy_s.w $7, $w{{[0-9]+}}

; MIPS32R5-DAG: sw  ${{[0-9]+}}, 28($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 24($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 20($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 16($sp)

; MIPS64-DAG: daddiu $4
; MIPS64-DAG: daddiu $7
; MIPS64-DAG: move $5
; MIPS64-DAG: move $6

; MIPS64R5-DAG: copy_s.d $4, $w[[W0:[0-9]+]][0]
; MIPS64R5-DAG: copy_s.d $5, $w[[W0]][1]
; MIPS64R5-DAG: copy_s.d $6, $w[[W1:[0-9]+]][0]
; MIPS64R5-DAG: copy_s.d $7, $w[[W1]][1]

; MIPS32: jal i16_8
; MIPS64: jalr $25

; MIPS32-DAG: sw $5, 12(${{[0-9]+}})
; MIPS32-DAG: sw $4, 8(${{[0-9]+}})
; MIPS32-DAG: sw $3, 4(${{[0-9]+}})
; MIPS32-DAG: sw $2, %lo(gv8i16)(${{[0-9]+}})

; MIPS32R5-DAG: insert.w $w[[W0:[0-9]+]][0], $2
; MIPS32R5-DAG: insert.w $w[[W0]][1], $3
; MIPS32R5-DAG: insert.w $w[[W0]][2], $4
; MIPS32R5-DAG: insert.w $w[[W0]][3], $5
; MIPS32R5-DAG: st.w $w[[W0]]

; MIPS64: sd $3
; MIPS64: sd $2

; MIPS64R5-DAG: insert.d $w[[W2:[0-9]+]][0], $2
; MIPS64R5-DAG: insert.d $w[[W2]][1], $3

  %0 = call <8 x i16> @i16_8(<8 x i16> <i16 6, i16 7, i16 9, i16 10, i16 6, i16 7, i16 9, i16 10>, <8 x i16> <i16 6, i16 7, i16 9, i16 10, i16 12, i16 8, i16 9, i16 10>)
  store <8 x i16> %0, <8 x i16> * @gv8i16
  ret void
}

define void @calli32_2() {
entry:
; ALL-LABEL: calli32_2:

; MIPS32-DAG: addiu $4
; MIPS32-DAG: addiu $5
; MIPS32-DAG: addiu $6
; MIPS32-DAG: addiu $7

; MIPS32R5-DAG: addiu $4
; MIPS32R5-DAG: addiu $5
; MIPS32R5-DAG: addiu $6
; MIPS32R5-DAG: addiu $7

; MIPS64: daddiu $4
; MIPS64: daddiu $5

; MIPS64R5-NOT ld $4
; MIPS64R5-NOT: ld $5

; MIPS32: jal i32_2
; MIPS64: jalr $25

; MIPS32-DAG: sw $2, %lo(gv2i32)(${{[0-9]+}})
; MIPS32-DAG: sw $3, 4(${{[0-9]+}})

; MIPS32R5-DAG: sw $2, %lo(gv2i32)(${{[0-9]+}})
; MIPS32R5-DAG: sw $3, 4(${{[0-9]+}})

; MIPS64: sd $2

; MIPS64R5: sd $2

  %0 = call <2 x i32> @i32_2(<2 x i32> <i32 6, i32 7>, <2 x i32> <i32 12, i32 8>)
  store <2 x i32> %0, <2 x i32> * @gv2i32
  ret void
}

define void @calli32_4() {
entry:
; ALL-LABEL: calli32_4:

; MIPS32-DAG: sw  ${{[0-9]+}}, 28($sp)
; MIPS32-DAG: sw  ${{[0-9]+}}, 24($sp)
; MIPS32-DAG: sw  ${{[0-9]+}}, 20($sp)
; MIPS32-DAG: sw  ${{[0-9]+}}, 16($sp)

; MIPS32-DAG: addiu $4
; MIPS32-DAG: addiu $5
; MIPS32-DAG: addiu $6
; MIPS32-DAG: addiu $7

; MIPS32R5-DAG: sw  ${{[0-9]+}}, 28($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 24($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 20($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 16($sp)

; MIPS32R5-DAG: addiu $4
; MIPS32R5-DAG: addiu $5
; MIPS32R5-DAG: addiu $6
; MIPS32R5-DAG: addiu $7

; MIPS64-DAG: daddiu $4
; MIPS64-DAG: daddiu $6
; MIPS64-DAG: daddiu $5
; MIPS64-DAG: move $7

; MIPS64R5-DAG: copy_s.d $4, $w[[W0:[0-9]+]][0]
; MIPS64R5-DAG: copy_s.d $5, $w[[W0]][1]
; MIPS64R5-DAG: copy_s.d $6, $w[[W1:[0-9]+]][0]
; MIPS64R5-DAG: copy_s.d $7, $w[[W1]][1]

; MIPS32: jal i32_4
; MIPS64: jalr $25

; MIPS32-DAG: sw $5, 12(${{[0-9]+}})
; MIPS32-DAG: sw $4, 8(${{[0-9]+}})
; MIPS32-DAG: sw $3, 4(${{[0-9]+}})
; MIPS32-DAG: sw $2, %lo(gv4i32)(${{[0-9]+}})

; MIPS32R5-DAG: insert.w $w[[W0:[0-9]+]][0], $2
; MIPS32R5-DAG: insert.w $w[[W0]][1], $3
; MIPS32R5-DAG: insert.w $w[[W0]][2], $4
; MIPS32R5-DAG: insert.w $w[[W0]][3], $5
; MIPS32R5-DAG: st.w $w[[W0]]

; MIPS64-DAG: sd $2
; MIPS64-DAG: sd $3

; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][0], $2
; MIPS64R6-DAG: insert.d $w[[W0:[0-9]+]][1], $3

  %0 = call <4 x i32> @i32_4(<4 x i32> <i32 6, i32 7, i32 9, i32 10>, <4 x i32> <i32 12, i32 8, i32 9, i32 10>)
  store <4 x i32> %0, <4 x i32> * @gv4i32
  ret void
}

define void @calli64_2() {
entry:
; ALL-LABEL: calli64_2:

; MIPS32-DAG: sw  ${{[0-9a-z]+}}, 28($sp)
; MIPS32-DAG: sw  ${{[0-9a-z]+}}, 24($sp)
; MIPS32-DAG: sw  ${{[0-9a-z]+}}, 20($sp)
; MIPS32-DAG: sw  ${{[0-9a-z]+}}, 16($sp)

; MIPS32-DAG: addiu $4
; MIPS32-DAG: addiu $5
; MIPS32-DAG: addiu $6
; MIPS32-DAG: addiu $7

; MIPS32R5-DAG: copy_s.w $4, $w{{[0-9]+}}
; MIPS32R5-DAG: copy_s.w $5, $w{{[0-9]+}}
; MIPS32R5-DAG: copy_s.w $6, $w{{[0-9]+}}
; MIPS32R5-DAG: copy_s.w $7, $w{{[0-9]+}}

; MIPS32R5-DAG: sw  ${{[0-9]+}}, 28($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 24($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 20($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 16($sp)

; MIPS64: daddiu $4
; MIPS64: daddiu $5
; MIPS64: daddiu $6
; MIPS64: daddiu $7

; MIPS64R5: daddiu $4
; MIPS64R5: daddiu $5
; MIPS64R5: daddiu $6
; MIPS64R5: daddiu $7

; MIPS32: jal i64_2
; MIPS64: jalr $25

; MIPS32-DAG: sw $5, 12(${{[0-9]+}})
; MIPS32-DAG: sw $4, 8(${{[0-9]+}})
; MIPS32-DAG: sw $3, 4(${{[0-9]+}})
; MIPS32-DAG: sw $2, %lo(gv2i64)(${{[0-9]+}})

; MIPS32R5-DAG: insert.w $w[[W0:[0-9]+]][0], $2
; MIPS32R5-DAG: insert.w $w[[W0]][1], $3
; MIPS32R5-DAG: insert.w $w[[W0]][2], $4
; MIPS32R5-DAG: insert.w $w[[W0]][3], $5
; MIPS32R5-DAG: st.w $w[[W0]]

; MIPS64-DAG: sd $3
; MIPS64-DAG: sd $2

; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][0], $2
; MIPS64R6-DAG: insert.d $w[[W0:[0-9]+]][1], $3

  %0 = call <2 x i64> @i64_2(<2 x i64> <i64 6, i64 7>, <2 x i64> <i64 12, i64 8>)
  store <2 x i64> %0, <2 x i64> * @gv2i64
  ret void
}

declare <2 x float> @float2_extern(<2 x float>, <2 x float>)
declare <4 x float> @float4_extern(<4 x float>, <4 x float>)
declare <2 x double> @double2_extern(<2 x double>, <2 x double>)

define void @callfloat_2() {
entry:
; ALL-LABEL: callfloat_2:

; MIPS32-DAG: addiu $4, $sp, 24
; MIPS32-DAG: addiu $6, $zero, 0
; MIPS32-DAG: lui $7

; MIPS32R5-DAG: addiu $4, $sp, 24
; MIPS32R5-DAG: addiu $6, $zero, 0
; MIPS32R5-DAG: lui $7

; MIPS64: dsll $4
; MIPS64: dsll $5

; MIPS64R5-DAG: copy_s.d $4, $w{{[0-9]+}}
; MIPS64R5-DAG: copy_s.d $5, $w{{[0-9]+}}

; MIPS32: jal float2_extern
; MIPS64: jalr $25

; MIPS32-DAG: lwc1 $f[[F0:[0-9]+]], 24($sp)
; MIPS32-DAG: lwc1 $f[[F1:[0-9]+]], 28($sp)

; MIPS32-DAG: swc1 $f[[F1]], 4(${{[0-9]+}})
; MIPS32-DAG: swc1 $f[[F0]], %lo(gv2f32)(${{[0-9]+}})

; MIPS32R5-DAG: lwc1 $f[[F0:[0-9]+]], 24($sp)
; MIPS32R5-DAG: lwc1 $f[[F1:[0-9]+]], 28($sp)

; MIPS32R5-DAG: swc1 $f[[F1]], 4(${{[0-9]+}})
; MIPS32R5-DAG: swc1 $f[[F0]], %lo(gv2f32)(${{[0-9]+}})

; MIPS64: sd $2

; MIPS64R5: sd $2

  %0 = call <2 x float> @float2_extern(<2 x float> <float 0.0, float -1.0>, <2 x float> <float 12.0, float 14.0>)
  store <2 x float> %0, <2 x float> * @gv2f32
  ret void
}

define void @callfloat_4() {
entry:
; ALL-LABEL: callfloat_4:

; MIPS32: sw ${{[0-9]+}}, 36($sp)
; MIPS32: sw ${{[0-9]+}}, 32($sp)
; MIPS32: sw ${{[0-9]+}}, 28($sp)
; MIPS32: sw ${{[0-9]+}}, 24($sp)
; MIPS32: sw ${{[0-9]+}}, 20($sp)
; MIPS32: sw ${{[0-9]+}}, 16($sp)
; MIPS32: addiu $4, $sp, 48
; MIPS32: addiu $6, $zero, 0
; MIPS32: lui $7

; MIPS32R5: copy_s.w $6, $w{{[0-9]+}}
; MIPS32R5: copy_s.w $7, $w{{[0-9]+}}
; MIPS32R5: sw ${{[0-9]+}}, 36($sp)
; MIPS32R5: sw ${{[0-9]+}}, 32($sp)
; MIPS32R5: sw ${{[0-9]+}}, 28($sp)
; MIPS32R5: sw ${{[0-9]+}}, 24($sp)
; MIPS32R5: sw ${{[0-9]+}}, 20($sp)
; MIPS32R5: sw ${{[0-9]+}}, 16($sp)
; MIPS32R5: addiu $4, $sp, 48

; MIPS64-DAG: dsll $4
; MIPS64-DAG: dsll $5
; MIPS64-DAG: dsll $6
; MIPS64-DAG: dsll $7

; MIPS64R5-DAG: copy_s.d $4, $w{{[0-9]+}}
; MIPS64R5-DAG: copy_s.d $5, $w{{[0-9]+}}
; MIPS64R5-DAG: copy_s.d $6, $w{{[0-9]+}}
; MIPS64R5-DAG: copy_s.d $7, $w{{[0-9]+}}

; MIPS64: jalr $25
; MIPS32: jal

; MIPS32-DAG: lwc1 $f{{[0-9]+}}, 48($sp)
; MIPS32-DAG: lwc1 $f{{[0-9]+}}, 52($sp)
; MIPS32-DAG: lwc1 $f{{[0-9]+}}, 56($sp)
; MIPS32-DAG: lwc1 $f{{[0-9]+}}, 60($sp)

; MIPS32R5: ld.w $w{{[0-9]+}}, 48($sp)

; MIPS64-DAG: $2
; MIPS64-DAG: $3

; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][0], $2
; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][1], $3

  %0 = call <4 x float> @float4_extern(<4 x float> <float 0.0, float -1.0, float 2.0, float 4.0>, <4 x float> <float 12.0, float 14.0, float 15.0, float 16.0>)
  store <4 x float> %0, <4 x float> * @gv4f32
  ret void
}

define void @calldouble_2() {
entry:
; ALL-LABEL: calldouble_2:

; MIPS32-DAG: sw ${{[0-9a-z]+}}, 36($sp)
; MIPS32-DAG: sw ${{[0-9a-z]+}}, 32($sp)
; MIPS32-DAG: sw ${{[0-9a-z]+}}, 28($sp)
; MIPS32-DAG: sw ${{[0-9a-z]+}}, 24($sp)
; MIPS32-DAG: sw ${{[0-9a-z]+}}, 20($sp)
; MIPS32-DAG: sw ${{[0-9a-z]+}}, 16($sp)

; MIPS32-DAG: addiu $4, $sp, [[R0:[0-9]+]]
; MIPS32-DAG: addiu $6, $zero, 0
; MIPS32-DAG: addiu $7, $zero, 0

; MIPS32R5-DAG: copy_s.w $4, $w{{[0-9]+}}
; MIPS32R5-DAG: copy_s.w $5, $w{{[0-9]+}}
; MIPS32R5-DAG: copy_s.w $6, $w{{[0-9]+}}
; MIPS32R5-DAG: copy_s.w $7, $w{{[0-9]+}}

; MIPS32R5-DAG: sw  ${{[0-9]+}}, 36($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 32($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 28($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 24($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 20($sp)
; MIPS32R5-DAG: sw  ${{[0-9]+}}, 16($sp)

; MIPS64-DAG: dsll $5
; MIPS64-DAG: dsll $6
; MIPS64-DAG: dsll $7
; MIPS64-DAG: daddiu $4

; MIPS64R5-DAG: copy_s.d $4, $w{{[0-9]+}}
; MIPS64R5-DAG: copy_s.d $5, $w{{[0-9]+}}
; MIPS64R5-DAG: copy_s.d $6, $w{{[0-9]+}}
; MIPS64R5-DAG: copy_s.d $7, $w{{[0-9]+}}

; MIPS32: jal double2_extern
; MIPS64: jalr $25

; MIPS32-DAG: ldc1 $f[[F0:[0-9]+]], 48($sp)
; MIPS32-DAG: ldc1 $f[[F1:[0-9]+]], 56($sp)

; MIPS32-DAG: sdc1 $f[[F1]], 8(${{[0-9]+}})
; MIPS32-DAG: sdc1 $f[[F0]], %lo(gv2f64)(${{[0-9]+}})

; MIPS32R5: ld.d $w[[W0:[0-9]+]], 48($sp)
; MIPS32R5: st.d $w[[W0]], 0(${{[0-9]+}})

; MIPS64-DAG: sd $2
; MIPS64-DAG: sd $3

; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][0], $2
; MIPS64R5-DAG: insert.d $w[[W0:[0-9]+]][1], $3

  %0 = call <2 x double> @double2_extern(<2 x double> <double 0.0, double -1.0>, <2 x double> <double 12.0, double 14.0>)
  store <2 x double> %0, <2 x double> * @gv2f64
  ret void
}

; The mixed tests show that due to alignment requirements, $5 is not used
; in argument passing.

define float @mixed_i8(<2 x float> %a, i8 %b, <2 x float> %c) {
entry:
; ALL-LABEL: mixed_i8:

; MIPS32-DAG: mtc1 $5, $f{{[0-9]+}}
; MIPS32: andi $[[R7:[0-9]+]], $6, 255
; MIPS32: mtc1 $[[R7]], $f[[F0:[0-9]+]]
; MIPS32: cvt.s.w $f{{[0-9]+}}, $f[[F0]]

; MIPS32-DAG: mtc1 $4, $f{{[0-9]+}}
; MIPS32-DAG: lwc1 $f{{[0-9]+}}, 16($sp)
; MIPS32-DAG: lwc1 $f{{[0-9]+}}, 20($sp)
; MIPS32-DAG: add.s $f0, $f{{[0-9]+}}, $f{{[0-9]+}}

; MIPS32R5: andi $[[R0:[0-9]+]], $6, 255
; MIPS32R5: sw $[[R0]], {{[0-9]+}}($sp)
; MIPS32R5: sw $[[R0]], {{[0-9]+}}($sp)
; MIPS32R5-DAG: sw $5, {{[0-9]+}}($sp)
; MIPS32R5-DAG: sw $4, {{[0-9]+}}($sp)

; MIPS64EB-DAG: sll $[[R0:[0-9]+]], $4, 0
; MIPS64EB-DAG: mtc1 $[[R0]], $f{{[0-9]+}}
; MIPS64EB: sll $[[R6:[0-9]+]], $5, 0
; MIPS64EB: andi $[[R7:[0-9]+]], $[[R6]], 255
; MIPS64EB: mtc1 $[[R7]], $f[[F0:[0-9]+]]
; MIPS64EB: cvt.s.w $f{{[0-9]+}}, $f[[F0]]

; MIPS64EB-DAG: dsrl $[[R1:[0-9]+]], $4, 32
; MIPS64EB-DAG: sll $[[R2:[0-9]+]], $[[R1]], 0
; MIPS64EB-DAG: mtc1 $[[R2:[0-9]+]], $f{{[0-9]+}}

; MIPS64EB-DAG: sll $[[R3:[0-9]+]], $6, 0
; MIPS64EB-DAG: mtc1 $[[R3]], $f{{[0-9]+}}
; MIPS64EB-DAG: dsrl $[[R4:[0-9]+]], $6, 32
; MIPS64EB-DAG: sll $[[R5:[0-9]+]], $[[R4]], 0
; MIPS64EB-DAG: mtc1 $[[R5:[0-9]+]], $f{{[0-9]+}}

; MIPS64EL-DAG: dsrl $[[R1:[0-9]+]], $4, 32
; MIPS64EL-DAG: sll $[[R2:[0-9]+]], $[[R1]], 0
; MIPS64EL-DAG: mtc1 $[[R2:[0-9]+]], $f{{[0-9]+}}

; MIPS64EL: sll $[[R6:[0-9]+]], $5, 0
; MIPS64EL: andi $[[R7:[0-9]+]], $[[R6]], 255
; MIPS64EL: mtc1 $[[R7]], $f[[F0:[0-9]+]]
; MIPS64EL: cvt.s.w $f{{[0-9]+}}, $f[[F0]]

; MIPS64EL-DAG: dsrl $[[R4:[0-9]+]], $6, 32
; MIPS64EL-DAG: sll $[[R5:[0-9]+]], $[[R4]], 0
; MIPS64EL-DAG: mtc1 $[[R5:[0-9]+]], $f{{[0-9]+}}

; MIPS64EL-DAG: sll $[[R0:[0-9]+]], $4, 0
; MIPS64EL-DAG: mtc1 $[[R0]], $f{{[0-9]+}}
; MIPS64EL-DAG: sll $[[R3:[0-9]+]], $6, 0
; MIPS64EL-DAG: mtc1 $[[R3]], $f{{[0-9]+}}

; MIPS64R5: sll $[[R0:[0-9]+]], $5, 0
; MIPS64R5: andi $[[R1:[0-9]+]], $[[R0]], 255
; MIPS64R5: sd $4, {{[0-9]+}}($sp)
; MIPS64R5: sd $6, {{[0-9]+}}($sp)

  %0 = zext i8 %b to i32
  %1 = uitofp i32 %0 to float
  %2 = insertelement <2 x float> undef, float %1, i32 0
  %3 = insertelement <2 x float> %2, float %1, i32 1
  %4 = fadd <2 x float> %3, %a
  %5 = fadd <2 x float> %4, %c
  %6 = extractelement <2 x float> %5, i32 0
  %7 = extractelement <2 x float> %5, i32 1
  %8 = fadd float %6, %7
  ret float %8
}

define <4 x float> @mixed_32(<4 x float> %a, i32 %b) {
entry:
; ALL-LABEL: mixed_32:

; MIPS32-DAG: mtc1 $6, $f{{[0-9]+}}
; MIPS32-DAG: mtc1 $7, $f{{[0-9]+}}
; MIPS32-DAG: lwc1 $f{{[0-9]+}}, 28($sp)
; MIPS32-DAG: lwc1 $f{{[0-9]+}}, 24($sp)
; MIPS32-DAG: swc1 $f{{[0-9]+}}, 0($4)
; MIPS32-DAG: swc1 $f{{[0-9]+}}, 4($4)
; MIPS32-DAG: swc1 $f{{[0-9]+}}, 8($4)
; MIPS32-DAG: swc1 $f{{[0-9]+}}, 12($4)

; MIPS32R5: insert.w $w[[W0:[0-9]+]][0], $6
; MIPS32R5: insert.w $w[[W0:[0-9]+]][1], $7
; MIPS32R5: lw $[[R0:[0-9]+]], 16($sp)
; MIPS32R5: insert.w $w[[W0:[0-9]+]][2], $[[R0]]
; MIPS32R5: lw $[[R1:[0-9]+]], 20($sp)
; MIPS32R5: insert.w $w[[W0:[0-9]+]][3], $[[R1]]
; MIPS32R5: lw $[[R0:[0-9]+]], 24($sp)

; MIPS64-DAG: sll ${{[0-9]+}}, $6, 0
; MIPS64-DAG: dsrl $[[R0:[0-9]+]], $4, 32
; MIPS64-DAG: sll $[[R1:[0-9]+]], $[[R0]], 0
; MIPS64-DAG: mtc1 $[[R1]], $f{{[0-9]+}}
; MIPS64-DAG: sll $[[R2:[0-9]+]], $4, 0
; MIPS64-DAG: dsrl $[[R3:[0-9]+]], $5, 32
; MIPS64-DAG: sll $[[R4:[0-9]+]], $[[R3]], 0
; MIPS64-DAG: mtc1 $[[R4]], $f{{[0-9]+}}
; MIPS64-DAG: mtc1 $[[R2]], $f{{[0-9]+}}
; MIPS64-DAG: sll	$[[R6:[0-9]+]], $5, 0
; MIPS64-DAG: mtc1 $[[R6:[0-9]+]], $f{{[0-9]+}}

; MIPS64R5: insert.d $w[[W0:[0-9]+]][0], $4
; MIPS64R5: insert.d $w[[W0]][1], $5
; MIPS64R5: sll $[[R0:[0-9]+]], $6, 0
; MIPS64R5: fill.w $w{{[0-9]+}}, $[[R0]]

  %0 = uitofp i32 %b to float
  %1 = insertelement <4 x float> undef, float %0, i32 0
  %2 = insertelement <4 x float> %1, float %0, i32 1
  %3 = insertelement <4 x float> %2, float %0, i32 2
  %4 = insertelement <4 x float> %3, float %0, i32 3
  %5 = fadd <4 x float> %4, %a
  ret <4 x float> %5
}


; This test is slightly more fragile than I'd like as the offset into the
; outgoing arguments area is dependant on the size of the stack frame for
; this function.

define <4 x float> @cast(<4 x i32> %a) {
entry:
; ALL-LABEL: cast:

; MIPS32: addiu $sp, $sp, -32
; MIPS32-DAG: sw $6, {{[0-9]+}}($sp)
; MIPS32-DAG: sw $7, {{[0-9]+}}($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 48($sp)
; MIPS32-DAG: lw ${{[0-9]+}}, 52($sp)

; MIPS32R5-DAG: insert.w  $w0[0], $6
; MIPS32R5-DAG: insert.w  $w0[1], $7
; MIPS32R5-DAG: lw  $[[R0:[0-9]+]], 16($sp)
; MIPS32R5-DAG: insert.w  $w0[2], $[[R0]]
; MIPS32R5-DAG: lw  $[[R1:[0-9]+]], 20($sp)
; MIPS32R5-DAG: insert.w  $w0[3], $[[R1]]

; MIPS64-DAG: sll ${{[0-9]+}}, $4, 0
; MIPS64-DAG: dsrl ${{[0-9]+}}, $4, 32
; MIPS64-DAG: sll ${{[0-9]+}}, $5, 0
; MIPS64-DAG: dsrl ${{[0-9]+}}, $5, 32

; MIPS64R5-DAG: insert.d  $w0[0], $4
; MIPS64R5-DAG: insert.d  $w0[1], $5

  %0 = uitofp <4 x i32> %a to <4 x float>
  ret <4 x float> %0
}

define <4 x float> @select(<4 x i32> %cond, <4 x float> %arg1, <4 x float> %arg2) {
entry:
; ALL-LABEL: select:

; MIPS32-DAG: andi ${{[0-9]+}}, $7, 1
; MIPS32-DAG: andi ${{[0-9]+}}, $6, 1
; MIPS32-DAG: lw $[[R0:[0-9]+]], 16($sp)
; MIPS32-DAG: andi ${{[0-9]+}}, $[[R0]], 1
; MIPS32-DAG: lw $[[R1:[0-9]+]], 20($sp)
; MIPS32-DAG: andi ${{[0-9]+}}, $[[R0]], 1

; MIPS32R5-DAG: insert.w $w[[W0:[0-9]+]][0], $6
; MIPS32R5-DAG: insert.w $w[[W0]][1], $7
; MIPS32R5-DAG: lw $[[R0:[0-9]+]], 16($sp)
; MIPS32R5-DAG: lw $[[R1:[0-9]+]], 20($sp)
; MIPS32R5-DAG: insert.w $w[[W0]][2], $[[R0]]
; MIPS32R5-DAG: insert.w $w[[W0]][3], $[[R1]]
; MIPS32R5-DAG: slli.w $w{{[0-9]}}, $w[[W0]]

; MIPS64-DAG: sll $[[R0:[0-9]+]], $6, 0
; MIPS64-DAG: mtc1 $[[R0]], $f{{[0-9]+}}
; MIPS64-DAG: dsrl $[[R1:[0-9]+]], $6, 32
; MIPS64-DAG: sll $[[R2:[0-9]+]], $[[R1]], 0
; MIPS64-DAG: mtc1 $[[R2]], $f{{[0-9]+}}

; MIPS64-DAG: sll $[[R3:[0-9]+]], $7, 0
; MIPS64-DAG: mtc1 $[[R3]], $f{{[0-9]+}}
; MIPS64-DAG: dsrl $[[R4:[0-9]+]], $7, 32
; MIPS64-DAG: sll $[[R5:[0-9]+]], $[[R4]], 0
; MIPS64-DAG: mtc1 $[[R5]], $f{{[0-9]+}}

; MIPS64-DAG: sll $[[R6:[0-9]+]], $8, 0
; MIPS64-DAG: mtc1 $[[R6]], $f{{[0-9]+}}
; MIPS64-DAG: dsrl $[[R7:[0-9]+]], $8, 32
; MIPS64-DAG: sll $[[R8:[0-9]+]], $[[R7]], 0
; MIPS64-DAG: mtc1 $[[R8]], $f{{[0-9]+}}

; MIPS64-DAG: sll $[[R9:[0-9]+]], $9, 0
; MIPS64-DAG: mtc1 $[[R9]], $f{{[0-9]+}}
; MIPS64-DAG: dsrl $[[R10:[0-9]+]], $9, 32
; MIPS64-DAG: sll $[[R11:[0-9]+]], $[[R10]], 0
; MIPS64-DAG: mtc1 $[[R11]], $f{{[0-9]+}}

; MIPS64-DAG: sll $[[R12:[0-9]+]], $4, 0
; MIPS64-DAG: andi ${{[0-9]+}}, $[[R12]], 1
; MIPS64-DAG: dsrl $[[R13:[0-9]+]], $4, 32
; MIPS64-DAG: sll $[[R14:[0-9]+]], $[[R13]], 0
; MIPS64-DAG: andi ${{[0-9]+}}, $[[R14]], 1

; MIPS64-DAG: sll $[[R15:[0-9]+]], $5, 0
; MIPS64-DAG: andi ${{[0-9]+}}, $[[R15]], 1
; MIPS64-DAG: dsrl $[[R16:[0-9]+]], $5, 32
; MIPS64-DAG: sll $[[R17:[0-9]+]], $[[R16]], 0
; MIPS64-DAG: andi ${{[0-9]+}}, $[[R17]], 1

; MIPS64R5-DAG: insert.d $w{{[0-9]+}}[0], $8
; MIPS64R5-DAG: insert.d $w{{[0-9]+}}[1], $9
; MIPS64R5-DAG: insert.d $w{{[0-9]+}}[0], $6
; MIPS64R5-DAG: insert.d $w{{[0-9]+}}[1], $7
; MIPS64R5-DAG: insert.d $w{{[0-9]+}}[0], $4
; MIPS64R5-DAG: insert.d $w{{[0-9]+}}[1], $5

  %cond.t = trunc <4 x i32> %cond to <4 x i1>
  %res = select <4 x i1> %cond.t, <4 x float> %arg1, <4 x float> %arg2
  ret <4 x float> %res
}
