# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN:   FileCheck %s
# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN:   FileCheck %s
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN:   FileCheck %s

# Check that the IAS expands macro instructions in the same way as GAS

.extern sym
# imm and rs are deliberately swapped to test whitespace separated arguments.
.macro EX2 insn, rd, imm, rs
.ex\@: \insn \rd, \rs, \imm
.endm

.option pic0

EX2 addiu $2, 1 $3           # CHECK: addiu    $2, $3, 1
EX2 addiu $2, ~1 $3          # CHECK: addiu    $2, $3, -2
EX2 addiu $2, ~ 1 $3         # CHECK: addiu    $2, $3, -2
EX2 addiu $2, 1+1 $3         # CHECK: addiu    $2, $3, 2
EX2 addiu $2, 1+ 1 $3        # CHECK: addiu    $2, $3, 2
EX2 addiu $2, 1 +1 $3        # CHECK: addiu    $2, $3, 2
EX2 addiu $2, 1 + 1 $3       # CHECK: addiu    $2, $3, 2
EX2 addiu $2, 1+~1 $3        # CHECK: addiu    $2, $3, -1
EX2 addiu $2, 1+~ 1 $3       # CHECK: addiu    $2, $3, -1
EX2 addiu $2, 1+ ~1 $3       # CHECK: addiu    $2, $3, -1
EX2 addiu $2, 1 +~1 $3       # CHECK: addiu    $2, $3, -1
EX2 addiu $2, 1 +~ 1 $3      # CHECK: addiu    $2, $3, -1
EX2 addiu $2, 1 + ~1 $3      # CHECK: addiu    $2, $3, -1
EX2 addiu $2, 1 + ~ 1 $3     # CHECK: addiu    $2, $3, -1
EX2 addiu $2, 1+(1) $3       # CHECK: addiu    $2, $3, 2
EX2 addiu $2, 1 +(1) $3      # CHECK: addiu    $2, $3, 2
EX2 addiu $2, 1+ (1) $3      # CHECK: addiu    $2, $3, 2
EX2 addiu $2, 1 + (1) $3     # CHECK: addiu    $2, $3, 2
EX2 addiu $2, 1+(1)+1 $3     # CHECK: addiu    $2, $3, 3
EX2 addiu $2, 1 +(1)+1 $3    # CHECK: addiu    $2, $3, 3
EX2 addiu $2, 1+ (1)+1 $3    # CHECK: addiu    $2, $3, 3
EX2 addiu $2, 1 + (1)+1 $3   # CHECK: addiu    $2, $3, 3
nop                          # CHECK: nop
