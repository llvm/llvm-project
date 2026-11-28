# RUN: llvm-mc -triple=mips -mcpu=mips1 %s | FileCheck %s --check-prefixes=ALL,MIPS1
# RUN: llvm-mc -triple=mips -mcpu=mips2 %s | FileCheck %s --check-prefixes=ALL,MIPS2
# RUN: llvm-mc -triple=mips64 -mcpu=mips3 -target-abi=o32 %s \
# RUN:   | FileCheck %s --check-prefixes=ALL,MIPS3
# RUN: llvm-mc -triple=mips64 -mcpu=mips4 -target-abi=o32 %s \
# RUN:   | FileCheck %s --check-prefixes=ALL,MIPS4
# RUN: llvm-mc -triple=mips64 -mcpu=mips5 -target-abi=o32 %s \
# RUN:   | FileCheck %s --check-prefixes=ALL,MIPS4
# RUN: llvm-mc -triple=mips -mcpu=mips32 %s | FileCheck %s --check-prefix=ALL

# In .set reorder mode, insert the nops that hazards need, as GNU as does:
# loads on MIPS I, moves to and from coprocessors and FPU compares on MIPS I
# to MIPS III, and HI and LO accesses before MIPS32.

  .text
  .set reorder

# ALL-LABEL: load_use:
# ALL:         lw $2, 0($4)
# MIPS1-NEXT:  nop
# ALL-NEXT:    addu $3, $2, $2
# ALL-NEXT:    lw $2, 0($4)
# ALL-NEXT:    addu $3, $4, $4
load_use:
  lw $2, 0($4)
  addu $3, $2, $2
  lw $2, 0($4)
  addu $3, $4, $4

# ALL-LABEL: load_branch:
# ALL:         lb $2, 0($4)
# ALL-NEXT:    .set noreorder
# MIPS1-NEXT:  nop
# ALL-NEXT:    beqz $2, load_branch
# ALL-NEXT:    nop
# ALL-NEXT:    .set reorder
# ALL-NEXT:    lw $ra, 0($sp)
# ALL-NEXT:    .set noreorder
# MIPS1-NEXT:  nop
# ALL-NEXT:    jr $ra
# ALL-NEXT:    nop
load_branch:
  lb $2, 0($4)
  beq $2, $0, load_branch
  lw $31, 0($sp)
  jr $31

# The two halves of an unaligned load can be back to back.
# ALL-LABEL: unaligned:
# ALL:         lwl $5, 0($4)
# ALL-NEXT:    lwr $5, 3($4)
# MIPS1-NEXT:  nop
# ALL-NEXT:    addu $6, $5, $5
unaligned:
  lwl $5, 0($4)
  lwr $5, 3($4)
  addu $6, $5, $5

# ALL-LABEL: unaligned_macro:
# ALL:         lwl $5, 0($4)
# ALL-NEXT:    lwr $5, 3($4)
# MIPS1-NEXT:  nop
# ALL-NEXT:    addu $6, $5, $5
unaligned_macro:
  ulw $5, 0($4)
  addu $6, $5, $5

# ALL-LABEL: from_fpu:
# ALL:         mfc1 $2, $f0
# MIPS1-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS3-NEXT:  nop
# ALL-NEXT:    addu $3, $2, $2
# ALL-NEXT:    cfc1 $2, $31
# MIPS1-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS3-NEXT:  nop
# ALL-NEXT:    addu $3, $2, $2
from_fpu:
  mfc1 $2, $f0
  addu $3, $2, $2
  cfc1 $2, $31
  addu $3, $2, $2

# ALL-LABEL: to_fpu:
# ALL:         mtc1 $2, $f0
# MIPS1-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS3-NEXT:  nop
# ALL-NEXT:    add.s $f2, $f0, $f0
# ALL-NEXT:    mtc1 $2, $f0
# ALL-NEXT:    addu $3, $4, $4
# ALL-NEXT:    mtc1 $2, $f1
# MIPS1-NEXT:  nop
# MIPS2-NEXT:  nop
# ALL-NEXT:    add.d $f2, $f0, $f0
# $f1 is part of $d0 with 32-bit FPRs, but not with the 64-bit FPRs that
# MIPS III uses here.
to_fpu:
  mtc1 $2, $f0
  add.s $f2, $f0, $f0
  mtc1 $2, $f0
  addu $3, $4, $4
  mtc1 $2, $f1
  add.d $f2, $f0, $f0

# ALL-LABEL: fp_compare:
# ALL:         c.eq.s $f0, $f2
# ALL-NEXT:    .set noreorder
# MIPS1-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS3-NEXT:  nop
# ALL-NEXT:    bc1t fp_compare
# ALL-NEXT:    nop
fp_compare:
  c.eq.s $f0, $f2
  bc1t fp_compare

# Writing the FPU control register needs two instructions before reading the
# condition code, and one before another access to a control register.
# ALL-LABEL: to_fpu_control:
# ALL:         ctc1 $2, $31
# ALL-NEXT:    .set noreorder
# MIPS1-NEXT:  nop
# MIPS1-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS3-NEXT:  nop
# MIPS3-NEXT:  nop
# ALL-NEXT:    bc1t to_fpu_control
# ALL-NEXT:    nop
# ALL-NEXT:    .set reorder
# ALL-NEXT:    ctc1 $2, $31
# ALL-NEXT:    addu $3, $4, $4
# ALL-NEXT:    .set noreorder
# MIPS1-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS3-NEXT:  nop
# ALL-NEXT:    bc1f to_fpu_control
# ALL-NEXT:    nop
# ALL-NEXT:    .set reorder
# ALL-NEXT:    ctc1 $2, $31
# MIPS1-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS3-NEXT:  nop
# ALL-NEXT:    cfc1 $3, $31
# ALL-NEXT:    ctc1 $2, $31
# ALL-NEXT:    add.s $f0, $f2, $f2
to_fpu_control:
  ctc1 $2, $31
  bc1t to_fpu_control
  ctc1 $2, $31
  addu $3, $4, $4
  bc1f to_fpu_control
  ctc1 $2, $31
  cfc1 $3, $31
  ctc1 $2, $31
  add.s $f0, $f2, $f2

# ALL-LABEL: cop0:
# ALL:         mtc0 $2, $12, 0
# MIPS1-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS3-NEXT:  nop
# ALL-NEXT:    mfc0 $3, $12, 0
# MIPS1-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS3-NEXT:  nop
# ALL-NEXT:    addu $4, $3, $3
cop0:
  mtc0 $2, $12
  mfc0 $3, $12
  addu $4, $3, $3

# ALL-LABEL: fp_load:
# ALL:         lwc1 $f0, 0($4)
# MIPS1-NEXT:  nop
# ALL-NEXT:    add.s $f2, $f0, $f0
# ALL-NEXT:    lwc1 $f1, 4($4)
# MIPS1-NEXT:  nop
# ALL-NEXT:    add.d $f2, $f0, $f0
# ALL-NEXT:    lwc1 $f0, 0($4)
# ALL-NEXT:    add.s $f2, $f4, $f4
fp_load:
  lwc1 $f0, 0($4)
  add.s $f2, $f0, $f0
  lwc1 $f1, 4($4)
  add.d $f2, $f0, $f0
  lwc1 $f0, 0($4)
  add.s $f2, $f4, $f4

# ALL-LABEL: hilo:
# ALL:         mflo $2
# MIPS1-NEXT:  nop
# MIPS1-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS3-NEXT:  nop
# MIPS3-NEXT:  nop
# MIPS4-NEXT:  nop
# MIPS4-NEXT:  nop
# ALL-NEXT:    mult $3, $4
# ALL-NEXT:    mfhi $2
# ALL-NEXT:    addu $3, $4, $4
# MIPS1-NEXT:  nop
# MIPS2-NEXT:  nop
# MIPS3-NEXT:  nop
# MIPS4-NEXT:  nop
# ALL-NEXT:    mthi $3
# ALL-NEXT:    mfhi $2
# ALL-NEXT:    mtlo $3
hilo:
  mflo $2
  mult $3, $4
  mfhi $2
  addu $3, $4, $4
  mthi $3
  mfhi $2
  mtlo $3

# Nothing is inserted in .set noreorder mode, but the hazards from before it
# are resolved first.
# ALL-LABEL: noreorder:
# ALL:         lw $2, 0($4)
# MIPS1-NEXT:  nop
# ALL-NEXT:    .set noreorder
# ALL-NEXT:    lw $2, 0($4)
# ALL-NEXT:    addu $3, $2, $2
# ALL-NEXT:    .set reorder
noreorder:
  lw $2, 0($4)
  .set noreorder
  lw $2, 0($4)
  addu $3, $2, $2
  .set reorder

# The same with .set pop.
# ALL-LABEL: pop:
# ALL:         .set push
# ALL:         .set reorder
# ALL-NEXT:    lw $2, 0($4)
# MIPS1-NEXT:  nop
# ALL-NEXT:    .set pop
pop:
  .set noreorder
  .set push
  .set reorder
  lw $2, 0($4)
  .set pop
  .set reorder

# Hazards are tracked per section.
# ALL-LABEL: sections:
# ALL:         lw $2, 0($4)
# ALL:         .data
# ALL:         .text
# MIPS1-NEXT:  nop
# ALL-NEXT:    addu $3, $2, $2
sections:
  lw $2, 0($4)
  .data
  .word 0
  .text
  addu $3, $2, $2

# Hazards inside a macro get nops in .set noreorder mode too. The R_MIPS_JALR
# label stays on the jalr.
# ALL-LABEL: macro:
# ALL:         lw $6, %got(ext)($gp)
# MIPS1-NEXT:  nop
# ALL-NEXT:    lw $6, 0($6)
# ALL-NEXT:    addu $3, $6, $6
# ALL-NEXT:    lw $25, %call16(ext)($gp)
# MIPS1-NEXT:  nop
# ALL-NEXT:    .reloc $tmp0, R_MIPS_JALR, ext
# ALL-NEXT:  $tmp0:
# ALL-NEXT:    jalr $25
# ALL-NEXT:    nop
# ALL-NEXT:    .set reorder
macro:
  .option pic2
  .set noreorder
  lw $6, ext
  addu $3, $6, $6
  jal ext
  nop
  .set reorder
  .option pic0

# A hazard left at the end of the file is resolved.
# ALL-LABEL: end:
# ALL:         lw $2, 0($4)
# MIPS1-NEXT:  nop
# ALL-NOT:     nop
end:
  lw $2, 0($4)
