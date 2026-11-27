# RUN: llvm-mc %s -triple=mipsel-unknown-linux-gnu -mcpu=mips32 -filetype=obj | \
# RUN:   llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=O32
# RUN: llvm-mc %s -triple=mips-unknown-linux-gnu -mcpu=mips32r2 -filetype=obj | \
# RUN:   llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=O32
# RUN: llvm-mc %s -triple=mips-unknown-linux-gnu -mattr=micromips -filetype=obj | \
# RUN:   llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=MM
# RUN: llvm-mc %s -triple=mips64-unknown-linux-gnuabin32 -filetype=obj --defsym=NEWABI=1 | \
# RUN:   llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=N32
# RUN: llvm-mc %s -triple=mips64-unknown-linux-gnu -filetype=obj --defsym=NEWABI=1 | \
# RUN:   llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=N64

## In PIC mode, "la" and "jal" need to know whether the symbol is local or
## global. For a symbol that is defined later in the file without .globl, this
## is only known once the whole file has been parsed. Like GNU as, use the
## local GOT sequence (%got/%lo for O32, %got_disp for N32/N64) for such
## symbols, and keep the global sequence for symbols that end up global or
## undefined.

  .abicalls
  .option pic2
  .text
  .set noreorder
  .globl caller
  .ent caller
caller:
  .cpload $25
  .set reorder

  jal fwd_local
# O32:      lw $25, 0($gp)
# O32-NEXT:   R_MIPS_GOT16 .text
# O32-NEXT: addiu $25, $25, 140
# O32-NEXT:   R_MIPS_LO16 .text
# O32-NEXT: jalr $25
# O32-NEXT:   R_MIPS_JALR fwd_local
# O32-NEXT: nop
# MM:       lw $25, 0($gp)
# MM-NEXT:    R_MICROMIPS_GOT16 fwd_local
# MM-NEXT:  addiu $25, $25, 0
# MM-NEXT:    R_MICROMIPS_LO16 fwd_local
# MM-NEXT:  jalr $ra, $25
# MM-NEXT:    R_MICROMIPS_JALR fwd_local
# MM-NEXT:  nop
# N32:      lw $25, 0($gp)
# N32-NEXT:   R_MIPS_GOT_DISP fwd_local
# N32-NEXT: jalr $25
# N32-NEXT:   R_MIPS_JALR fwd_local
# N64:      ld $25, 0($gp)
# N64-NEXT:   R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE fwd_local
# N64-NEXT: jalr $25
# N64-NEXT:   R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE fwd_local

  ## Made global after its use.
  jal fwd_global
# O32:      lw $25, 0($gp)
# O32-NEXT:   R_MIPS_CALL16 fwd_global
# O32-NEXT: jalr $25
# O32-NEXT:   R_MIPS_JALR fwd_global
# O32-NEXT: nop
# N64:      ld $25, 0($gp)
# N64-NEXT:   R_MIPS_CALL16/R_MIPS_NONE/R_MIPS_NONE fwd_global

  jal undef_func
# O32:      lw $25, 0($gp)
# O32-NEXT:   R_MIPS_CALL16 undef_func
# O32-NEXT: jalr $25
# O32-NEXT:   R_MIPS_JALR undef_func
# O32-NEXT: nop
# N64:      ld $25, 0($gp)
# N64-NEXT:   R_MIPS_CALL16/R_MIPS_NONE/R_MIPS_NONE undef_func

.ifndef NEWABI
  la $4, fwd_data
# O32:      lw $4, 0($gp)
# O32-NEXT:   R_MIPS_GOT16 .data
# O32-NEXT: addiu $4, $4, 16
# O32-NEXT:   R_MIPS_LO16 .data
# MM:       lw $4, 0($gp)
# MM-NEXT:    R_MICROMIPS_GOT16 .data
# MM-NEXT:  addiu $4, $4, 16
# MM-NEXT:    R_MICROMIPS_LO16 .data

  la $5, fwd_data+8
# O32-NEXT: lw $5, 0($gp)
# O32-NEXT:   R_MIPS_GOT16 .data
# O32-NEXT: addiu $5, $5, 24
# O32-NEXT:   R_MIPS_LO16 .data

  la $6, fwd_data($7)
# O32-NEXT: lw $6, 0($gp)
# O32-NEXT:   R_MIPS_GOT16 .data
# O32-NEXT: addiu $6, $6, 16
# O32-NEXT:   R_MIPS_LO16 .data
# O32-NEXT: addu $6, $6, $7

  la $7, fwd_data($7)
# O32-NEXT: lw $1, 0($gp)
# O32-NEXT:   R_MIPS_GOT16 .data
# O32-NEXT: addiu $1, $1, 16
# O32-NEXT:   R_MIPS_LO16 .data
# O32-NEXT: addu $7, $1, $7

  lw $8, fwd_data+4
# O32-NEXT: lw $8, 0($gp)
# O32-NEXT:   R_MIPS_GOT16 .data
# O32-NEXT: addiu $8, $8, 16
# O32-NEXT:   R_MIPS_LO16 .data
# O32-NEXT: lw $8, 4($8)

  la $9, fwd_global_data
# O32-NEXT: lw $9, 0($gp)
# O32-NEXT:   R_MIPS_GOT16 fwd_global_data

  la $10, undef_data+8
# O32-NEXT: lw $10, 0($gp)
# O32-NEXT:   R_MIPS_GOT16 undef_data
# O32-NEXT: addiu $10, $10, 8

  ## The branch offset must account for the relaxed sequence.
  b 1f
  la $11, fwd_data
1:
# O32-NEXT: b 0x84 <caller+0x84>
# O32-NEXT: nop
# O32-NEXT: lw $11, 0($gp)
# O32-NEXT:   R_MIPS_GOT16 .data
# O32-NEXT: addiu $11, $11, 16
# O32-NEXT:   R_MIPS_LO16 .data
.endif

  jr $ra
  .end caller

  .ent fwd_local
fwd_local:
  jr $ra
  .end fwd_local

  .ent fwd_global
fwd_global:
  jr $ra
  .end fwd_global
  .globl fwd_global

  .data
  .space 16
fwd_data:
  .4byte 0, 0, 0
fwd_global_data:
  .4byte 0
  .globl fwd_global_data
