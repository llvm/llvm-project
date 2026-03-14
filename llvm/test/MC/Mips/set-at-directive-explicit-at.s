# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 \
# RUN:     2>%t1 | FileCheck %s
# RUN: FileCheck -check-prefix=WARNINGS %s < %t1

# Check that the assembler can handle the documented syntax for ".set at" and
# will set the correct value for $at.
# Note that writing $at is always the same as writing $1.

    .text
foo:
# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
# WARNINGS: :[[@LINE+2]]:11: warning: used $at (currently $1) without ".set noat"
    .set    at=$1
    jr    $at

# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
# WARNINGS: :[[@LINE+2]]:11: warning: used $at (currently $1) without ".set noat"
    .set    at=$1
    jr    $1

# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
# WARNINGS: :[[@LINE+2]]:11: warning: used $at (currently $1) without ".set noat"
    .set    at=$at
    jr    $at

# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
# WARNINGS: :[[@LINE+2]]:11: warning: used $at (currently $1) without ".set noat"
    .set    at=$at
    jr    $1

# WARNINGS-NOT: warning: used $at (currently ${{[0-9]+}}) without ".set noat"
# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
    .set    at=$2
    jr    $at
# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
    .set    at=$3
    jr    $at
# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
    .set noat
    jr    $at
# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
    .set at=$0
    jr    $at

# CHECK:   jr    $16                     # encoding: [0x08,0x00,0x00,0x02]
# WARNINGS: :[[@LINE+2]]:11: warning: used $at (currently $16) without ".set noat"
    .set    at=$16
    jr    $s0

# CHECK:   jr    $16                     # encoding: [0x08,0x00,0x00,0x02]
# WARNINGS: :[[@LINE+2]]:11: warning: used $at (currently $16) without ".set noat"
    .set    at=$16
    jr    $16
