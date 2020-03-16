# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MIPS,O32

# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -mattr=+xgot -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MIPS,XO32

# RUN: llvm-mc %s -triple mips64-unknown-linux-gnuabin32 -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MIPS,N32

# RUN: llvm-mc %s -triple mips64-unknown-linux-gnuabin32 -mattr=+xgot -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MIPS,XN32

# RUN: llvm-mc %s -triple mips64-unknown-linux-gnu -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MIPS,N64

# RUN: llvm-mc %s -triple mips64-unknown-linux-gnu -mattr=+xgot -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MIPS,XN64

# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -mattr=micromips -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MM,O32-MM

# Repeat the tests but using ELF output. An initial version of this patch did
# this as the output different depending on whether it went through
# MCAsmStreamer or MCELFStreamer. This ensures that the assembly expansion and
# direct objection emission match.

# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -filetype=obj | \
# RUN:   llvm-objdump -d -r - | FileCheck %s --check-prefixes=ELF-O32
# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -mattr=+xgot -filetype=obj | \
# RUN:   llvm-objdump -d -r - | FileCheck %s --check-prefixes=ELF-XO32
# RUN: llvm-mc %s -triple mips64-unknown-linux-gnuabin32 -filetype=obj | \
# RUN:   llvm-objdump -d -r - | FileCheck %s --check-prefixes=ELF-N32
# RUN: llvm-mc %s -triple mips64-unknown-linux-gnuabin32 -mattr=+xgot -filetype=obj | \
# RUN:   llvm-objdump -d -r - | FileCheck %s --check-prefixes=ELF-XN32
# RUN: llvm-mc %s -triple mips64-unknown-linux-gnu -filetype=obj | \
# RUN:   llvm-objdump -d -r - | FileCheck %s --check-prefixes=ELF-N64
# RUN: llvm-mc %s -triple mips64-unknown-linux-gnu -mattr=+xgot -filetype=obj | \
# RUN:   llvm-objdump -d -r - | FileCheck %s --check-prefixes=ELF-XN64

  .weak weak_label

  .text
  .option pic2

  .ent local_label
local_label:
  .frame  $sp, 0, $ra
  .set noreorder

  jal local_label
  nop

# Expanding "jal local_label":
# O32: lw     $25, %got(local_label)($gp)   # encoding: [0x8f,0x99,A,A]
# O32:                                      #   fixup A - offset: 0, value: %got(local_label), kind:   fixup_Mips_GOT
# O32: addiu  $25, $25, %lo(local_label)    # encoding: [0x27,0x39,A,A]
# O32:                                      #   fixup A - offset: 0, value: %lo(local_label), kind:   fixup_Mips_LO16
# O32-NEXT: .reloc ($tmp0), R_MIPS_JALR, local_label

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 .text
# ELF-O32-NEXT: 27 39 00 00 addiu $25, $25, 0
# ELF-O32-NEXT:                 R_MIPS_LO16 .text
# ELF-O32-NEXT: 03 20 f8 09 jalr $25
# ELF-O32-NEXT:                 R_MIPS_JALR local_label

# XO32:      lw    $25, %got(local_label)($gp)  # encoding: [0x8f,0x99,A,A]
# XO32-NEXT:                                    #   fixup A - offset: 0, value: %got(local_label), kind:   fixup_Mips_GOT
# XO32-NEXT: addiu $25, $25, %lo(local_label)   # encoding: [0x27,0x39,A,A]
# XO32-NEXT:                                    #   fixup A - offset: 0, value: %lo(local_label), kind:   fixup_Mips_LO16
# XO32-NEXT: .reloc ($tmp0), R_MIPS_JALR, local_label

# ELF-XO32:      8f 99 00 00 lw $25, 0($gp)
# ELF-XO32-NEXT:                 R_MIPS_GOT16 .text
# ELF-XO32-NEXT: 27 39 00 00 addiu $25, $25, 0
# ELF-XO32-NEXT:                 R_MIPS_LO16  .text
# ELF-XO32-NEXT: 03 20 f8 09 jalr    $25
# ELF-XO32-NEXT:                 R_MIPS_JALR local_label

# N32: lw  $25, %got_disp(local_label)($gp) # encoding: [0x8f,0x99,A,A]
# N32:                                      #   fixup A - offset: 0, value: %got_disp(local_label), kind:   fixup_Mips_GOT_DISP
# N32-NEXT: .reloc .Ltmp0, R_MIPS_JALR, local_label

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP local_label
# ELF-N32-NEXT: 03 20 f8 09 jalr $25
# ELF-N32-NEXT:                 R_MIPS_JALR local_label

# XN32:      lw $25, %got_disp(local_label)($gp)  # encoding: [0x8f,0x99,A,A]
# XN32-NEXT:                                      #   fixup A - offset: 0, value: %got_disp(local_label), kind: fixup_Mips_GOT_DISP
# XN32-NEXT: .reloc .Ltmp0, R_MIPS_JALR, local_label

# ELF-XN32:      8f 99 00 00 lw $25, 0($gp)
# ELF-XN32-NEXT:                 R_MIPS_GOT_DISP local_label
# ELF-XN32-NEXT: 03 20 f8 09 jalr    $25
# ELF-XN32-NEXT:                 R_MIPS_JALR local_label

# N64: ld  $25, %got_disp(local_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64:                                      #   fixup A - offset: 0, value: %got_disp(local_label), kind:   fixup_Mips_GOT_DISP
# N64-NEXT: .reloc .Ltmp0, R_MIPS_JALR, local_label

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE local_label
# ELF-N64-NEXT: 03 20 f8 09 jalr $25
# ELF-N64-NEXT: R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE local_label

# XN64:      ld $25, %got_disp(local_label)($gp)  # encoding: [0xdf,0x99,A,A]
# XN64-NEXT:                                      #   fixup A - offset: 0, value: %got_disp(local_label), kind: fixup_Mips_GOT_DISP
# XN64-NEXT: .reloc .Ltmp0, R_MIPS_JALR, local_label

# ELF-XN64:      df 99 00 00 ld $25, 0($gp)
# ELF-XN64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE local_label
# ELF-XN64-NEXT: 03 20 f8 09 jalr $25
# ELF-XN64-NEXT:                 R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE local_label

# O32-MM: lw    $25, %got(local_label)($gp)      # encoding: [0xff,0x3c,A,A]
# O32-MM:                                        #   fixup A - offset: 0, value: %got(local_label), kind:   fixup_MICROMIPS_GOT16
# O32-MM: addiu $25, $25, %lo(local_label)       # encoding: [0x33,0x39,A,A]
# O32-MM:                                        #   fixup A - offset: 0, value: %lo(local_label), kind:   fixup_MICROMIPS_LO16
# O32-MM-NEXT: .reloc ($tmp0), R_MICROMIPS_JALR, local_label

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  jal local_label+8
  nop

# O32:      lw $25, %got(local_label+8)($gp)    # encoding: [0x8f,0x99,A,A]
# O32-NEXT:                                     #   fixup A - offset: 0, value: %got(local_label+8), kind: fixup_Mips_GOT
# O32-NEXT: addiu $25, $25, %lo(local_label+8)  # encoding: [0x27,0x39,A,A]
# O32-NEXT:                                     #   fixup A - offset: 0, value: %lo(local_label+8), kind: fixup_Mips_LO16
# O32-NOT:  .reloc

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 .text
# ELF-O32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-O32-NEXT:                 R_MIPS_LO16 .text
# ELF-O32-NEXT: 03 20 f8 09 jalr $25
# ELF-O32-NEXT: 00 00 00 00 nop

# XO32:      lw $25, %got(local_label+8)($gp)     # encoding: [0x8f,0x99,A,A]
# XO32-NEXT:                                      #   fixup A - offset: 0, value: %got(local_label+8), kind: fixup_Mips_GOT
# XO32-NEXT: addiu $25, $25, %lo(local_label+8)   # encoding: [0x27,0x39,A,A]
# XO32-NEXT:                                      #   fixup A - offset: 0, value: %lo(local_label+8), kind: fixup_Mips_LO16
# XO32-NOT:  .reloc

# ELF-XO32:      8f 99 00 00 lw $25, 0($gp)
# ELF-XO32-NEXT:                 R_MIPS_GOT16 .text
# ELF-XO32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-XO32-NEXT:                 R_MIPS_LO16  .text
# ELF-XO32-NEXT: 03 20 f8 09 jalr    $25
# ELF-XO32-NEXT: 00 00 00 00 nop

# N32:      lw $25, %got_disp(local_label)($gp)   # encoding: [0x8f,0x99,A,A]
# N32-NEXT:                                       #   fixup A - offset: 0, value: %got_disp(local_label), kind: fixup_Mips_GOT_DISP
# N32-NEXT: addiu $25, $25, 8                     # encoding: [0x27,0x39,0x00,0x08]
# N32-NEXT: .reloc .Ltmp1, R_MIPS_JALR, local_label

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP local_label
# ELF-N32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-N32-NEXT: 03 20 f8 09 jalr $25
# ELF-N32-NEXT:                 R_MIPS_JALR local_label

# XN32:      lw $25, %got_disp(local_label)($gp)  # encoding: [0x8f,0x99,A,A]
# XN32-NEXT:                                      #   fixup A - offset: 0, value: %got_disp(local_label), kind: fixup_Mips_GOT_DISP
# XN32-NEXT: addiu $25, $25, 8                    # encoding: [0x27,0x39,0x00,0x08]
# XN32-NEXT: .reloc .Ltmp1, R_MIPS_JALR, local_label

# ELF-XN32:      8f 99 00 00 lw $25, 0($gp)
# ELF-XN32-NEXT:                 R_MIPS_GOT_DISP local_label
# ELF-XN32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-XN32-NEXT: 03 20 f8 09 jalr $25
# ELF-XN32-NEXT:                 R_MIPS_JALR local_label

# N64:      ld $25, %got_disp(local_label)($gp)   # encoding: [0xdf,0x99,A,A]
# N64-NEXT:                                       #   fixup A - offset: 0, value: %got_disp(local_label), kind: fixup_Mips_GOT_DISP
# N64-NEXT: daddiu $25, $25, 8                    # encoding: [0x67,0x39,0x00,0x08]
# N64-NEXT: .reloc .Ltmp1, R_MIPS_JALR, local_label

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE local_label
# ELF-N64-NEXT: 67 39 00 08 daddiu $25, $25, 8
# ELF-N64-NEXT: 03 20 f8 09 jalr $25
# ELF-N64-NEXT: R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE local_label

# XN64:      ld $25, %got_disp(local_label)($gp)  # encoding: [0xdf,0x99,A,A]
# XN64-NEXT:                                      #   fixup A - offset: 0, value: %got_disp(local_label), kind: fixup_Mips_GOT_DISP
# XN64-NEXT: daddiu $25, $25, 8                   # encoding: [0x67,0x39,0x00,0x08]
# XN64-NEXT: .reloc .Ltmp1, R_MIPS_JALR, local_label

# ELF-XN64:      df 99 00 00 ld $25, 0($gp)
# ELF-XN64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE local_label
# ELF-XN64-NEXT: 67 39 00 08 daddiu  $25, $25, 8
# ELF-XN64-NEXT: 03 20 f8 09 jalr $25
# ELF-XN64-NEXT:                 R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE local_label

# O32-MM:      lw $25, %got(local_label+8)($gp)   # encoding: [0xff,0x3c,A,A]
# O32-MM-NEXT:                                    #   fixup A - offset: 0, value: %got(local_label+8), kind: fixup_MICROMIPS_GOT16
# O32-MM-NEXT: addiu $25, $25, %lo(local_label+8) # encoding: [0x33,0x39,A,A]
# O32-MM-NEXT:                                    #   fixup A - offset: 0, value: %lo(local_label+8), kind: fixup_MICROMIPS_LO16
# O32-MM-NOT:  .reloc

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  jal weak_label
  nop

# Expanding "jal weak_label":
# O32: lw  $25, %call16(weak_label)($gp) # encoding: [0x8f,0x99,A,A]
# O32:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind:   fixup_Mips_CALL16
# O32-NEXT: .reloc ($tmp1), R_MIPS_JALR, weak_label

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_CALL16 weak_label
# ELF-O32-NEXT: 03 20 f8 09 jalr $25
# ELF-O32-NEXT:                 R_MIPS_JALR weak_label

# XO32:      lui  $25, %call_hi(weak_label)       # encoding: [0x3c,0x19,A,A]
# XO32-NEXT:                                      #   fixup A - offset: 0, value: %call_hi(weak_label), kind: fixup_Mips_CALL_HI16
# XO32-NEXT: addu $25, $25, $gp                   # encoding: [0x03,0x3c,0xc8,0x21]
# XO32-NEXT: lw   $25, %call_lo(weak_label)($25)  # encoding: [0x8f,0x39,A,A]
# XO32-NEXT:                                      #   fixup A - offset: 0, value: %call_lo(weak_label), kind: fixup_Mips_CALL_LO16
# XO32-NEXT: .reloc ($tmp1), R_MIPS_JALR, weak_label

# ELF-XO32:      3c 19 00 00 lui $25, 0
# ELF-XO32-MEXT:                  R_MIPS_CALL_HI16 weak_label
# ELF-XO32-MEXT: 03 3c c8 21 addu $25, $25, $gp
# ELF-XO32-MEXT: 8f 39 00 00 lw $25, 0($25)
# ELF-XO32-MEXT:                  R_MIPS_CALL_LO16 weak_label
# ELF-XO32-MEXT: 03 20 f8 09 jalr $25
# ELF-XO32-MEXT:                  R_MIPS_JALR weak_label

# N32: lw  $25, %call16(weak_label)($gp) # encoding: [0x8f,0x99,A,A]
# N32:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind:   fixup_Mips_CALL16
# N32-NEXT: .reloc .Ltmp2, R_MIPS_JALR, weak_label

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_CALL16 weak_label
# ELF-N32-NEXT: 03 20 f8 09 jalr $25
# ELF-N32-NEXT:                 R_MIPS_JALR weak_label

# XN32:      lui  $25, %call_hi(weak_label)       # encoding: [0x3c,0x19,A,A]
# XN32-NEXT:                                      #   fixup A - offset: 0, value: %call_hi(weak_label), kind: fixup_Mips_CALL_HI16
# XN32-NEXT: addu $25, $25, $gp                   # encoding: [0x03,0x3c,0xc8,0x21]
# XN32-NEXT: lw   $25, %call_lo(weak_label)($25)  # encoding: [0x8f,0x39,A,A]
# XN32-NEXT:                                      #   fixup A - offset: 0, value: %call_lo(weak_label), kind: fixup_Mips_CALL_LO16
# XN32-NEXT: .reloc .Ltmp2, R_MIPS_JALR, weak_label

# ELF-XN32:      3c 19 00 00 lui $25, 0
# ELF-XN32-NEXT:                  R_MIPS_CALL_HI16 weak_label
# ELF-XN32-NEXT: 03 3c c8 21 addu $25, $25, $gp
# ELF-XN32-NEXT: 8f 39 00 00 lw $25, 0($25)
# ELF-XN32-NEXT:                  R_MIPS_CALL_LO16 weak_label
# ELF-XN32-NEXT: 03 20 f8 09 jalr    $25
# ELF-XN32-NEXT:                  R_MIPS_JALR weak_label

# N64: ld  $25, %call16(weak_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind:   fixup_Mips_CALL16
# N64-NEXT: .reloc .Ltmp2, R_MIPS_JALR, weak_label

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_CALL16/R_MIPS_NONE/R_MIPS_NONE weak_label
# ELF-N64-NEXT: 03 20 f8 09 jalr $25
# ELF-N64-NEXT:                 R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE weak_label

# XN64:      lui   $25, %call_hi(weak_label)      # encoding: [0x3c,0x19,A,A]
# XN64-NEXT:                                      #   fixup A - offset: 0, value: %call_hi(weak_label), kind: fixup_Mips_CALL_HI16
# XN64-NEXT: daddu $25, $25, $gp                  # encoding: [0x03,0x3c,0xc8,0x2d]
# XN64-NEXT: ld    $25, %call_lo(weak_label)($25) # encoding: [0xdf,0x39,A,A]
# XN64-NEXT:                                      #   fixup A - offset: 0, value: %call_lo(weak_label), kind: fixup_Mips_CALL_LO16
# XN64-NEXT: .reloc .Ltmp2, R_MIPS_JALR, weak_label

# ELF-XN64:      3c 19 00 00 lui $25, 0
# ELF-XN64-NEXT:                  R_MIPS_CALL_HI16/R_MIPS_NONE/R_MIPS_NONE weak_label
# ELF-XN64-NEXT: 03 3c c8 2d daddu $25, $25, $gp
# ELF-XN64-NEXT: df 39 00 00 ld $25, 0($25)
# ELF-XN64-NEXT:                  R_MIPS_CALL_LO16/R_MIPS_NONE/R_MIPS_NONE weak_label
# ELF-XN64-NEXT: 03 20 f8 09 jalr    $25
# ELF-XN64-NEXT:                  R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE weak_label

# O32-MM: lw  $25, %call16(weak_label)($gp) # encoding: [0xff,0x3c,A,A]
# O32-MM:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind:   fixup_MICROMIPS_CALL16
# O32-MM-NEXT: .reloc ($tmp1), R_MICROMIPS_JALR, weak_label

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  jal weak_label+8
  nop

# O32:      lw  $25, %got(weak_label)($gp)  # encoding: [0x8f,0x99,A,A]
# O32-NEXT:                                 #   fixup A - offset: 0, value: %got(weak_label), kind: fixup_Mips_GOT
# O32-NEXT: addiu $25, $25, 8               # encoding: [0x27,0x39,0x00,0x08]
# O32-NOT:  .reloc

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 weak_label
# ELF-O32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-O32-NEXT: 03 20 f8 09 jalr $25
# ELF-O32-NEXT: 00 00 00 00 nop

# XO32:      lui  $25, %got_hi(weak_label+8)    # encoding: [0x3c,0x19,A,A]
# XO32-NEXT:                                    #   fixup A - offset: 0, value: %got_hi(weak_label+8), kind: fixup_Mips_GOT_HI16
# XO32-NEXT: addu $25, $25, $gp                 # encoding: [0x03,0x3c,0xc8,0x21]
# XO32-NEXT: lw   $25, %got_lo(weak_label)($25) # encoding: [0x8f,0x39,A,A]
# XO32-NEXT:                                    #   fixup A - offset: 0, value: %got_lo(weak_label), kind: fixup_Mips_GOT_LO16
# XO32-NEXT: addiu $25, $25, 8                  # encoding: [0x27,0x39,0x00,0x08]
# XO32-NOT:  .reloc

# ELF-XO32:      3c 19 00 00 lui $25, 0
# ELF-XO32-NEXT:                  R_MIPS_GOT_HI16 weak_label
# ELF-XO32-NEXT: 03 3c c8 21 addu $25, $25, $gp
# ELF-XO32-NEXT: 8f 39 00 00 lw $25, 0($25)
# ELF-XO32-NEXT:                  R_MIPS_GOT_LO16 weak_label
# ELF-XO32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-XO32-NEXT: 03 20 f8 09 jalr $25
# ELF-XO32-NEXT: 00 00 00 00 nop

# N32:      lw  $25, %got_disp(weak_label)($gp) # encoding: [0x8f,0x99,A,A]
# N32-NEXT:                                     #   fixup A - offset: 0, value: %got_disp(weak_label), kind:   fixup_Mips_GOT_DISP
# N32-NEXT: addiu $25, $25, 8                   # encoding: [0x27,0x39,0x00,0x08]
# N32-NEXT: .reloc .Ltmp3, R_MIPS_JALR, weak_label

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP weak_label
# ELF-N32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-N32-NEXT: 03 20 f8 09 jalr $25
# ELF-N32-NEXT:                 R_MIPS_JALR weak_label

# XN32:      lui  $25, %got_hi(weak_label+8)    # encoding: [0x3c,0x19,A,A]
# XN32-NEXT:                                    #   fixup A - offset: 0, value: %got_hi(weak_label+8), kind: fixup_Mips_GOT_HI16
# XN32-NEXT: addu $25, $25, $gp                 # encoding: [0x03,0x3c,0xc8,0x21]
# XN32-NEXT: lw   $25, %got_lo(weak_label)($25) # encoding: [0x8f,0x39,A,A]
# XN32-NEXT:                                    #   fixup A - offset: 0, value: %got_lo(weak_label), kind: fixup_Mips_GOT_LO16
# XN32-NEXT: addiu $25, $25, 8                  # encoding: [0x27,0x39,0x00,0x08]
# XN32-NEXT: .reloc .Ltmp3, R_MIPS_JALR, weak_label

# ELF-XN32:      3c 19 00 00 lui $25, 0
# ELF-XN32-NEXT:                  R_MIPS_GOT_HI16 weak_label+0x8
# ELF-XN32-NEXT: 03 3c c8 21 addu $25, $25, $gp
# ELF-XN32-NEXT: 8f 39 00 00 lw $25, 0($25)
# ELF-XN32-NEXT:                  R_MIPS_GOT_LO16 weak_label
# ELF-XN32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-XN32-NEXT: 03 20 f8 09 jalr $25
# ELF-XN32-NEXT:                  R_MIPS_JALR weak_label

# N64:      ld  $25, %got_disp(weak_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64:                                          #   fixup A - offset: 0, value: %got_disp(weak_label), kind:   fixup_Mips_GOT_DISP
# N64-NEXT: daddiu $25, $25, 8                  # encoding: [0x67,0x39,0x00,0x08]
# N64-NEXT: .reloc .Ltmp3, R_MIPS_JALR, weak_label

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE weak_label
# ELF-N64-NEXT: 67 39 00 08 daddiu $25, $25, 8
# ELF-N64-NEXT: 03 20 f8 09 jalr $25
# ELF-N64-NEXT:                 R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE weak_label

# XN64:      lui   $25, %got_hi(weak_label+8)     # encoding: [0x3c,0x19,A,A]
# XN64-NEXT:                                      #   fixup A - offset: 0, value: %got_hi(weak_label+8), kind: fixup_Mips_GOT_HI16
# XN64-NEXT: daddu $25, $25, $gp                  # encoding: [0x03,0x3c,0xc8,0x2d]
# XN64-NEXT: ld    $25, %got_lo(weak_label)($25)  # encoding: [0xdf,0x39,A,A]
# XN64-NEXT:                                      #   fixup A - offset: 0, value: %got_lo(weak_label), kind: fixup_Mips_GOT_LO16
# XN64-NEXT: daddiu $25, $25, 8                   # encoding: [0x67,0x39,0x00,0x08]
# XN64-NEXT: .reloc .Ltmp3, R_MIPS_JALR, weak_label

# ELF-XN64:      3c 19 00 00 lui $25, 0
# ELF-XN64-NEXT:                  R_MIPS_GOT_HI16/R_MIPS_NONE/R_MIPS_NONE weak_label+0x8
# ELF-XN64-NEXT: 03 3c c8 2d daddu $25, $25, $gp
# ELF-XN64-NEXT: df 39 00 00 ld $25, 0($25)
# ELF-XN64-NEXT:                  R_MIPS_GOT_LO16/R_MIPS_NONE/R_MIPS_NONE weak_label
# ELF-XN64-NEXT: 67 39 00 08 daddiu $25, $25, 8
# ELF-XN64-NEXT: 03 20 f8 09 jalr $25
# ELF-XN64-NEXT:                  R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE weak_label

# O32-MM:      lw  $25, %got(weak_label)($gp)     # encoding: [0xff,0x3c,A,A]
# O32-MM-NEXT:                                    #   fixup A - offset: 0, value: %got(weak_label), kind:   fixup_MICROMIPS_GOT16
# O32-MM-NEXT: addiu $25, $25, 8                  # encoding: [0x33,0x39,0x00,0x08]
# O32-MM-NOT:  .reloc

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  jal global_label
  nop

# Expanding "jal global_label":
# O32: lw  $25, %call16(global_label)($gp)  # encoding: [0x8f,0x99,A,A]
# O32-NEXT:                                 #   fixup A - offset: 0, value: %call16(global_label), kind: fixup_Mips_CALL16
# O32-NEXT: .reloc ($tmp2), R_MIPS_JALR, global_label

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_CALL16 global_label
# ELF-O32-NEXT: 03 20 f8 09 jalr $25
# ELF-O32-NEXT:                 R_MIPS_JALR global_label

# XO32:      lui  $25, %call_hi(global_label)       # encoding: [0x3c,0x19,A,A]
# XO32-NEXT:                                        #   fixup A - offset: 0, value: %call_hi(global_label), kind: fixup_Mips_CALL_HI16
# XO32-NEXT: addu $25, $25, $gp                     # encoding: [0x03,0x3c,0xc8,0x21]
# XO32-NEXT: lw   $25, %call_lo(global_label)($25)  # encoding: [0x8f,0x39,A,A]
# XO32-NEXT:                                        #   fixup A - offset: 0, value: %call_lo(global_label), kind: fixup_Mips_CALL_LO16
# XO32-NEXT: .reloc ($tmp2), R_MIPS_JALR, global_label

# ELF-XO32:      3c 19 00 00 lui $25, 0
# ELF-XO32-NEXT:                  R_MIPS_CALL_HI16 global_label
# ELF-XO32-NEXT: 03 3c c8 21 addu $25, $25, $gp
# ELF-XO32-NEXT: 8f 39 00 00 lw $25, 0($25)
# ELF-XO32-NEXT:                  R_MIPS_CALL_LO16 global_label
# ELF-XO32-NEXT: 03 20 f8 09 jalr $25
# ELF-XO32-NEXT:                  R_MIPS_JALR global_label

# N32: lw  $25, %call16(global_label)($gp)  # encoding: [0x8f,0x99,A,A]
# N32-NEXT:                                 #   fixup A - offset: 0, value: %call16(global_label), kind: fixup_Mips_CALL16
# N32-NEXT: .reloc .Ltmp4, R_MIPS_JALR, global_label

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_CALL16 global_label
# ELF-N32-NEXT: 03 20 f8 09 jalr $25
# ELF-N32-NEXT:                 R_MIPS_JALR global_label

# XN32:      lui  $25, %call_hi(global_label)       # encoding: [0x3c,0x19,A,A]
# XN32-NEXT:                                        #   fixup A - offset: 0, value: %call_hi(global_label), kind: fixup_Mips_CALL_HI16
# XN32-NEXT: addu $25, $25, $gp                     # encoding: [0x03,0x3c,0xc8,0x21]
# XN32-NEXT: lw   $25, %call_lo(global_label)($25)  # encoding: [0x8f,0x39,A,A]
# XN32-NEXT:                                        #   fixup A - offset: 0, value: %call_lo(global_label), kind: fixup_Mips_CALL_LO16
# XN32-NEXT: .reloc .Ltmp4, R_MIPS_JALR, global_label

# ELF-XN32:      3c 19 00 00 lui $25, 0
# ELF-XN32-NEXT:                  R_MIPS_CALL_HI16 global_label
# ELF-XN32-NEXT: 03 3c c8 21 addu $25, $25, $gp
# ELF-XN32-NEXT: 8f 39 00 00 lw $25, 0($25)
# ELF-XN32-NEXT:                  R_MIPS_CALL_LO16 global_label
# ELF-XN32-NEXT: 03 20 f8 09 jalr $25
# ELF-XN32-NEXT:                  R_MIPS_JALR global_label

# N64: ld  $25, %call16(global_label)($gp)  # encoding: [0xdf,0x99,A,A]
# N64-NEXT:                                 #   fixup A - offset: 0, value: %call16(global_label), kind: fixup_Mips_CALL16
# N64-NEXT: .reloc .Ltmp4, R_MIPS_JALR, global_label

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_CALL16/R_MIPS_NONE/R_MIPS_NONE global_label
# ELF-N64-NEXT: 03 20 f8 09 jalr $25
# ELF-N64-NEXT:                 R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE global_label

# XN64:      lui   $25, %call_hi(global_label)      # encoding: [0x3c,0x19,A,A]
# XN64-NEXT:                                        #   fixup A - offset: 0, value: %call_hi(global_label), kind: fixup_Mips_CALL_HI16
# XN64-NEXT: daddu $25, $25, $gp                    # encoding: [0x03,0x3c,0xc8,0x2d]
# XN64-NEXT: ld    $25, %call_lo(global_label)($25) # encoding: [0xdf,0x39,A,A]
# XN64-NEXT:                                        #   fixup A - offset: 0, value: %call_lo(global_label), kind: fixup_Mips_CALL_LO16
# XN64-NEXT: .reloc .Ltmp4, R_MIPS_JALR, global_label

# ELF-XN64:      3c 19 00 00 lui $25, 0
# ELF-XN64-NEXT:                  R_MIPS_CALL_HI16/R_MIPS_NONE/R_MIPS_NONE global_label
# ELF-XN64-NEXT: 03 3c c8 2d daddu $25, $25, $gp
# ELF-XN64-NEXT: df 39 00 00 ld $25, 0($25)
# ELF-XN64-NEXT:                  R_MIPS_CALL_LO16/R_MIPS_NONE/R_MIPS_NONE global_label
# ELF-XN64-NEXT: 03 20 f8 09 jalr $25
# ELF-XN64-NEXT:                  R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE global_label

# O32-MM: lw  $25, %call16(global_label)($gp) # encoding: [0xff,0x3c,A,A]
# O32-MM-NEXT:                                #   fixup A - offset: 0, value: %call16(global_label), kind: fixup_MICROMIPS_CALL16
# O32-MM-NEXT: .reloc ($tmp2), R_MICROMIPS_JALR, global_label

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  jal global_label+8
  nop

# O32:      lw  $25, %got(global_label)($gp)  # encoding: [0x8f,0x99,A,A]
# O32-NEXT:                                   #   fixup A - offset: 0, value: %got(global_label), kind: fixup_Mips_GOT
# O32-NEXT: addiu $25, $25, 8                 # encoding: [0x27,0x39,0x00,0x08]
# O32-NOT:  .reloc

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 global_label
# ELF-O32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-O32-NEXT: 03 20 f8 09 jalr $25
# ELF-O32-NEXT: 00 00 00 00 nop

# XO32:      lui  $25, %got_hi(global_label+8)    # encoding: [0x3c,0x19,A,A]
# XO32-NEXT:                                      #   fixup A - offset: 0, value: %got_hi(global_label+8), kind: fixup_Mips_GOT_HI16
# XO32-NEXT: addu $25, $25, $gp                   # encoding: [0x03,0x3c,0xc8,0x21]
# XO32-NEXT: lw   $25, %got_lo(global_label)($25) # encoding: [0x8f,0x39,A,A]
# XO32-NEXT:                                      #   fixup A - offset: 0, value: %got_lo(global_label), kind: fixup_Mips_GOT_LO16
# XO32-NEXT: addiu $25, $25, 8                    # encoding: [0x27,0x39,0x00,0x08]
# XO32-NOT:  .reloc

# ELF-XO32:      3c 19 00 00 lui $25, 0
# ELF-XO32-NEXT:                  R_MIPS_GOT_HI16 global_label
# ELF-XO32-NEXT: 03 3c c8 21 addu $25, $25, $gp
# ELF-XO32-NEXT: 8f 39 00 00 lw $25, 0($25)
# ELF-XO32-NEXT:                  R_MIPS_GOT_LO16 global_label
# ELF-XO32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-XO32-NEXT: 03 20 f8 09 jalr $25
# ELF-XO32-NEXT: 00 00 00 00 nop

# N32:      lw  $25, %got_disp(global_label)($gp) # encoding: [0x8f,0x99,A,A]
# N32-NEXT:                                       #   fixup A - offset: 0, value: %got_disp(global_label), kind: fixup_Mips_GOT_DISP
# N32-NEXT: addiu $25, $25, 8                     # encoding: [0x27,0x39,0x00,0x08]
# N32-NEXT: .reloc .Ltmp5, R_MIPS_JALR, global_label

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP global_label
# ELF-N32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-N32-NEXT: 03 20 f8 09 jalr $25
# ELF-N32-NEXT:                 R_MIPS_JALR global_label

# XN32:      lui  $25, %got_hi(global_label+8)    # encoding: [0x3c,0x19,A,A]
# XN32-NEXT:                                      #   fixup A - offset: 0, value: %got_hi(global_label+8), kind: fixup_Mips_GOT_HI16
# XN32-NEXT: addu $25, $25, $gp                   # encoding: [0x03,0x3c,0xc8,0x21]
# XN32-NEXT: lw   $25, %got_lo(global_label)($25) # encoding: [0x8f,0x39,A,A]
# XN32-NEXT:                                      #   fixup A - offset: 0, value: %got_lo(global_label), kind: fixup_Mips_GOT_LO16
# XN32-NEXT: addiu $25, $25, 8                    # encoding: [0x27,0x39,0x00,0x08]
# XN32-NEXT: .reloc .Ltmp5, R_MIPS_JALR, global_label

# ELF-XN32:      3c 19 00 00 lui $25, 0
# ELF-XN32-NEXT:                  R_MIPS_GOT_HI16 global_label+0x8
# ELF-XN32-NEXT: 03 3c c8 21 addu $25, $25, $gp
# ELF-XN32-NEXT: 8f 39 00 00 lw $25, 0($25)
# ELF-XN32-NEXT:                  R_MIPS_GOT_LO16 global_label
# ELF-XN32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-XN32-NEXT: 03 20 f8 09 jalr $25
# ELF-XN32-NEXT:                  R_MIPS_JALR global_label

# N64:      ld  $25, %got_disp(global_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64-NEXT:                                       #   fixup A - offset: 0, value: %got_disp(global_label), kind: fixup_Mips_GOT_DISP
# N64-NEXT: daddiu $25, $25, 8                    # encoding: [0x67,0x39,0x00,0x08]
# N64-NEXT: .reloc .Ltmp5, R_MIPS_JALR, global_label

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE global_label
# ELF-N64-NEXT: 67 39 00 08 daddiu $25, $25, 8
# ELF-N64-NEXT: 03 20 f8 09 jalr $25
# ELF-N64-NEXT:                 R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE global_label

# XN64:      lui   $25, %got_hi(global_label+8)     # encoding: [0x3c,0x19,A,A]
# XN64-NEXT:                                        #   fixup A - offset: 0, value: %got_hi(global_label+8), kind: fixup_Mips_GOT_HI16
# XN64-NEXT: daddu $25, $25, $gp                    # encoding: [0x03,0x3c,0xc8,0x2d]
# XN64-NEXT: ld    $25, %got_lo(global_label)($25)  # encoding: [0xdf,0x39,A,A]
# XN64-NEXT:                                        #   fixup A - offset: 0, value: %got_lo(global_label), kind: fixup_Mips_GOT_LO16
# XN64-NEXT: daddiu $25, $25, 8                     # encoding: [0x67,0x39,0x00,0x08]
# XN64-NEXT: .reloc .Ltmp5, R_MIPS_JALR, global_label

# ELF-XN64:      3c 19 00 00 lui $25, 0
# ELF-XN64-NEXT:                  R_MIPS_GOT_HI16/R_MIPS_NONE/R_MIPS_NONE global_label+0x8
# ELF-XN64-NEXT: 03 3c c8 2d daddu $25, $25, $gp
# ELF-XN64-NEXT: df 39 00 00 ld $25, 0($25)
# ELF-XN64-NEXT:                  R_MIPS_GOT_LO16/R_MIPS_NONE/R_MIPS_NONE global_label
# ELF-XN64-NEXT: 67 39 00 08 daddiu $25, $25, 8
# ELF-XN64-NEXT: 03 20 f8 09 jalr $25
# ELF-XN64-NEXT:                  R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE global_label

# O32-MM:      lw  $25, %got(global_label)($gp) # encoding: [0xff,0x3c,A,A]
# O32-MM-NEXT:                                  #   fixup A - offset: 0, value: %got(global_label), kind: fixup_MICROMIPS_GOT16
# O32-MM-NEXT: addiu $25, $25, 8                # encoding: [0x33,0x39,0x00,0x08]
# O32-MM-NOT:  .reloc

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  jal .text
  nop

# Expanding "jal .text":
# O32: lw  $25, %got(.text)($gp)  # encoding: [0x8f,0x99,A,A]
# O32-NEXT:                       #   fixup A - offset: 0, value: %got(.text), kind: fixup_Mips_GOT

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 .text

# XO32:      lw    $25, %got(.text)($gp)  # encoding: [0x8f,0x99,A,A]
# XO32-NEXT:                              #   fixup A - offset: 0, value: %got(.text), kind: fixup_Mips_GOT
# XO32-NEXT: addiu $25, $25, %lo(.text)   # encoding: [0x27,0x39,A,A]
# XO32-NEXT:                              #   fixup A - offset: 0, value: %lo(.text), kind: fixup_Mips_LO16
# XO32-NEXT: .reloc ($tmp3), R_MIPS_JALR, .text

# ELF-XO32:      8f 99 00 00 lw $25, 0($gp)
# ELF-XO32-NEXT:                 R_MIPS_GOT16 .text
# ELF-XO32-NEXT: 27 39 00 00 addiu $25, $25, 0
# ELF-XO32-NEXT:                 R_MIPS_LO16  .text
# ELF-XO32-NEXT: 03 20 f8 09 jalr $25
# ELF-XO32-NEXT:                 R_MIPS_JALR  .text

# N32: lw  $25, %got_disp(.text)($gp) # encoding: [0x8f,0x99,A,A]
# N32-NEXT:                           #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_Mips_GOT_DISP

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP .text

# XN32:      lw $25, %got_disp(.text)($gp)  # encoding: [0x8f,0x99,A,A]
# XN32-NEXT:                                #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_Mips_GOT_DISP
# XN32-NEXT: .reloc .Ltmp6, R_MIPS_JALR, .text

# ELF-XN32:      8f 99 00 00 lw $25, 0($gp)
# ELF-XN32-NEXT:                 R_MIPS_GOT_DISP .text
# ELF-XN32-NEXT: 03 20 f8 09 jalr $25
# ELF-XN32-NEXT:                 R_MIPS_JALR .text

# N64: ld  $25, %got_disp(.text)($gp) # encoding: [0xdf,0x99,A,A]
# N64-NEXT:                           #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_Mips_GOT_DISP

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE  .text

# XN64:      ld $25, %got_disp(.text)($gp)  # encoding: [0xdf,0x99,A,A]
# XN64-NEXT:                                #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_Mips_GOT_DISP
# XN64-NEXT: .reloc .Ltmp6, R_MIPS_JALR, .text

# ELF-XN64:      df 99 00 00 ld $25, 0($gp)
# ELF-XN64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE .text
# ELF-XN64-NEXT: 03 20 f8 09 jalr $25
# ELF-XN64-NEXT:                 R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE .text

# O32-MM: lw    $25, %got(.text)($gp)      # encoding: [0xff,0x3c,A,A]
# O32-MM-NEXT:                                  #   fixup A - offset: 0, value: %got(.text), kind: fixup_MICROMIPS_GOT16
# O32-MM-NEXT: addiu $25, $25, %lo(.text)       # encoding: [0x33,0x39,A,A]
# O32-MM-NEXT:                                  #   fixup A - offset: 0, value: %lo(.text), kind: fixup_MICROMIPS_LO16
# O42-MM-NEXT: .reloc ($tmp3), R_MICROMIPS_JALR, .text

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  jal .text+8
  nop

# O32:      lw  $25, %got(.text+8)($gp) # encoding: [0x8f,0x99,A,A]
# O32-NEXT:                             #   fixup A - offset: 0, value: %got(.text+8), kind: fixup_Mips_GOT

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 .text

# XO32:      lw    $25, %got(.text+8)($gp)  # encoding: [0x8f,0x99,A,A]
# XO32-NEXT:                                #   fixup A - offset: 0, value: %got(.text+8), kind: fixup_Mips_GOT
# XO32-NEXT: addiu $25, $25, %lo(.text+8)   # encoding: [0x27,0x39,A,A]
# XO32-NEXT:                                #   fixup A - offset: 0, value: %lo(.text+8), kind: fixup_Mips_LO16
# XO32-NOT:  .reloc

# ELF-XO32:      8f 99 00 00 lw $25, 0($gp)
# ELF-XO32-NEXT:                 R_MIPS_GOT16 .text
# ELF-XO32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-XO32-NEXT:                 R_MIPS_LO16  .text
# ELF-XO32-NEXT: 03 20 f8 09 jalr $25
# ELF-XO32-NEXT: 00 00 00 00 nop

# N32:      lw  $25, %got_disp(.text)($gp)  # encoding: [0x8f,0x99,A,A]
# N32-NEXT:                                 #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_Mips_GOT_DISP

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP .text

# XN32:      lw $25, %got_disp(.text)($gp)  # encoding: [0x8f,0x99,A,A]
# XN32-NEXT:                                #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_Mips_GOT_DISP
# XN32-NEXT: addiu $25, $25, 8              # encoding: [0x27,0x39,0x00,0x08]
# XN32-NEXT: .reloc .Ltmp7, R_MIPS_JALR, .text

# ELF-XN32:      8f 99 00 00 lw $25, 0($gp)
# ELF-XN32-NEXT:                 R_MIPS_GOT_DISP .text
# ELF-XN32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-XN32-NEXT: 03 20 f8 09 jalr $25
# ELF-XN32-NEXT:                 R_MIPS_JALR .text

# N64:      ld  $25, %got_disp(.text)($gp)  # encoding: [0xdf,0x99,A,A]
# N64-NEXT:                                 #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_Mips_GOT_DISP

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE  .text

# XN64:      ld $25, %got_disp(.text)($gp)  # encoding: [0xdf,0x99,A,A]
# XN64-NEXT:                                #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_Mips_GOT_DISP
# XN64-NEXT: daddiu $25, $25, 8             # encoding: [0x67,0x39,0x00,0x08]
# XN64-NEXT: .reloc .Ltmp7, R_MIPS_JALR, .text

# ELF-XN64:      df 99 00 00 ld $25, 0($gp)
# ELF-XN64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE .text
# ELF-XN64-NEXT: 67 39 00 08 daddiu $25, $25, 8
# ELF-XN64-NEXT: 03 20 f8 09 jalr $25
# ELF-XN64-NEXT:                 R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE .text

# O32-MM:      lw    $25, %got(.text+8)($gp)  # encoding: [0xff,0x3c,A,A]
# O32-MM-NEXT:                                #   fixup A - offset: 0, value: %got(.text+8), kind: fixup_MICROMIPS_GOT16
# O32-MM-NEXT: addiu $25, $25, %lo(.text+8)   # encoding: [0x33,0x39,A,A]
# O32-MM-NEXT:                                #   fixup A - offset: 0, value: %lo(.text+8), kind: fixup_MICROMIPS_LO16
# O42-MM-NEXT: .reloc ($tmp4), R_MICROMIPS_JALR, .text

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  # local labels ($tmp symbols)
  jal 1f
  nop

# Expanding "jal 1f":
# O32:      lw     $25, %got($tmp4)($gp)  # encoding: [0x8f,0x99,A,A]
# O32-NEXT:                               #   fixup A - offset: 0, value: %got($tmp4), kind: fixup_Mips_GOT
# O32-NEXT: addiu  $25, $25, %lo($tmp4)   # encoding: [0x27,0x39,A,A]
# O32-NEXT:                               #   fixup A - offset: 0, value: %lo($tmp4), kind: fixup_Mips_LO16
# O32-NEXT: .reloc ($tmp5), R_MIPS_JALR, ($tmp4)

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 .text
# ELF-O32-NEXT: 27 39 00 b8 addiu $25, $25, 184
# ELF-O32-NEXT:                 R_MIPS_LO16 .text
# ELF-O32-NEXT: 03 20 f8 09 jalr $25
# ELF-O32-NEXT:                 R_MIPS_JALR $tmp0

# XO32:      lw    $25, %got($tmp4)($gp)  # encoding: [0x8f,0x99,A,A]
# XO32-NEXT:                              #   fixup A - offset: 0, value: %got($tmp4), kind: fixup_Mips_GOT
# XO32-NEXT: addiu $25, $25, %lo($tmp4)   # encoding: [0x27,0x39,A,A]
# XO32-NEXT:                              #   fixup A - offset: 0, value: %lo($tmp4), kind: fixup_Mips_LO16
# XO32-NEXT: .reloc ($tmp5), R_MIPS_JALR, ($tmp4)

# ELF-XO32:      8f 99 00 00 lw $25, 0($gp)
# ELF-XO32-NEXT:                 R_MIPS_GOT16 .text
# ELF-XO32-NEXT: 27 39 00 d8 addiu   $25, $25, 216
# ELF-XO32-NEXT:                 R_MIPS_LO16  .text
# ELF-XO32-NEXT: 03 20 f8 09 jalr    $25
# ELF-XO32-NEXT:                 R_MIPS_JALR  $tmp0

# N32: lw  $25, %got_disp(.Ltmp8)($gp)  # encoding: [0x8f,0x99,A,A]
# N32-NEXT:                                  #   fixup A - offset: 0, value: %got_disp(.Ltmp8), kind: fixup_Mips_GOT_DISP

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP .Ltmp0

# XN32:      lw $25, %got_disp(.Ltmp8)($gp) # encoding: [0x8f,0x99,A,A]
# XN32-NEXT:                                #   fixup A - offset: 0, value: %got_disp(.Ltmp8), kind: fixup_Mips_GOT_DISP
# XN32-NEXT: .reloc .Ltmp9, R_MIPS_JALR, .Ltmp8

# ELF-XN32:      8f 99 00 00 lw $25, 0($gp)
# ELF-XN32-NEXT:                 R_MIPS_GOT_DISP .Ltmp0
# ELF-XN32-NEXT: 03 20 f8 09 jalr $25
# ELF-XN32-NEXT:                 R_MIPS_JALR .Ltmp0

# N64: ld  $25, %got_disp(.Ltmp8)($gp)  # encoding: [0xdf,0x99,A,A]
# N64-NEXT:                             #   fixup A - offset: 0, value: %got_disp(.Ltmp8), kind: fixup_Mips_GOT_DISP

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE .Ltmp0

# XN64:      ld $25, %got_disp(.Ltmp8)($gp) # encoding: [0xdf,0x99,A,A]
# XN64-NEXT:                                #   fixup A - offset: 0, value: %got_disp(.Ltmp8), kind: fixup_Mips_GOT_DISP
# XN64-NEXT: .reloc .Ltmp9, R_MIPS_JALR, .Ltmp8

# ELF-XN64:      df 99 00 00 ld $25, 0($gp)
# ELF-XN64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE .Ltmp0
# ELF-XN64-NEXT: 03 20 f8 09 jalr $25
# ELF-XN64-NEXT:                 R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE .Ltmp0

# O32-MM: lw    $25, %got($tmp4)($gp)     # encoding: [0xff,0x3c,A,A]
# O32-MM-NEXT:                            #   fixup A - offset: 0, value: %got($tmp4), kind: fixup_MICROMIPS_GOT16
# O32-MM-NEXT: addiu $25, $25, %lo($tmp4) # encoding: [0x33,0x39,A,A]
# O32-MM-NEXT:                            #   fixup A - offset: 0, value: %lo($tmp4), kind: fixup_MICROMIPS_LO16
# O32-MM-NEXT: .reloc ($tmp5), R_MICROMIPS_JALR, ($tmp4)

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  jal 1f+8
  nop

# O32:      lw     $25, %got(($tmp4)+8)($gp)  # encoding: [0x8f,0x99,A,A]
# O32-NEXT:                                   #   fixup A - offset: 0, value: %got(($tmp4)+8), kind: fixup_Mips_GOT
# O32-NEXT: addiu  $25, $25, %lo(($tmp4)+8)   # encoding: [0x27,0x39,A,A]
# O32-NEXT:                                   #   fixup A - offset: 0, value: %lo(($tmp4)+8), kind: fixup_Mips_LO16
# O32-NOT:  .reloc

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 .text
# ELF-O32-NEXT: 27 39 00 c0 addiu $25, $25, 192
# ELF-O32-NEXT:                 R_MIPS_LO16 .text
# ELF-O32-NEXT: 03 20 f8 09 jalr $25
# ELF-O32-NEXT: 00 00 00 00 nop

# XO32:      lw    $25, %got(($tmp4)+8)($gp)  # encoding: [0x8f,0x99,A,A]
# XO32-NEXT:                                  #   fixup A - offset: 0, value: %got(($tmp4)+8), kind: fixup_Mips_GOT
# XO32-NEXT: addiu $25, $25, %lo(($tmp4)+8)   # encoding: [0x27,0x39,A,A]
# XO32-NEXT:                                  #   fixup A - offset: 0, value: %lo(($tmp4)+8), kind: fixup_Mips_LO16
# XO32-NOT:  .reloc

# ELF-XO32:      8f 99 00 00 lw $25, 0($gp)
# ELF-XO32-NEXT:                 R_MIPS_GOT16 .text
# ELF-XO32-NEXT: 27 39 00 e0 addiu $25, $25, 224
# ELF-XO32-NEXT:                 R_MIPS_LO16  .text
# ELF-XO32-NEXT: 03 20 f8 09 jalr $25
# ELF-XO32-NEXT: 00 00 00 00 nop

# N32:      lw  $25, %got_disp(.Ltmp8)($gp)   # encoding: [0x8f,0x99,A,A]
# N32-NEXT:                                   #   fixup A - offset: 0, value: %got_disp(.Ltmp8), kind: fixup_Mips_GOT_DISP

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP .Ltmp0

# XN32:      lw $25, %got_disp(.Ltmp8)($gp) # encoding: [0x8f,0x99,A,A]
# XN32-NEXT:                                #   fixup A - offset: 0, value: %got_disp(.Ltmp8), kind: fixup_Mips_GOT_DISP
# XN32-NEXT: addiu $25, $25, 8              # encoding: [0x27,0x39,0x00,0x08]
# XN32-NEXT: .reloc .Ltmp10, R_MIPS_JALR, .Ltmp8

# ELF-XN32:      8f 99 00 00 lw $25, 0($gp)
# ELF-XN32-NEXT:                 R_MIPS_GOT_DISP .Ltmp0
# ELF-XN32-NEXT: 27 39 00 08 addiu $25, $25, 8
# ELF-XN32-NEXT: 03 20 f8 09 jalr $25
# ELF-XN32-NEXT:                 R_MIPS_JALR .Ltmp0

# N64:     ld  $25, %got_disp(.Ltmp8)($gp)  # encoding: [0xdf,0x99,A,A]
# N64-NEXT:                                 #   fixup A - offset: 0, value: %got_disp(.Ltmp8), kind: fixup_Mips_GOT_DISP

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE .Ltmp0

# XN64:      ld $25, %got_disp(.Ltmp8)($gp) # encoding: [0xdf,0x99,A,A]
# XN64-NEXT:                                #   fixup A - offset: 0, value: %got_disp(.Ltmp8), kind: fixup_Mips_GOT_DISP
# XN64-NEXT: daddiu $25, $25, 8             # encoding: [0x67,0x39,0x00,0x08]
# XN64-NEXT: .reloc .Ltmp10, R_MIPS_JALR, .Ltmp8

# ELF-XN64:      df 99 00 00 ld $25, 0($gp)
# ELF-XN64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE .Ltmp0
# ELF-XN64-NEXT: 67 39 00 08 daddiu $25, $25, 8
# ELF-XN64-NEXT: 03 20 f8 09 jalr $25
# ELF-XN64-NEXT:                 R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE .Ltmp0

# O32-MM:      lw    $25, %got(($tmp4)+8)($gp)  # encoding: [0xff,0x3c,A,A]
# O32-MM-NEXT:                                  #   fixup A - offset: 0, value: %got(($tmp4)+8), kind: fixup_MICROMIPS_GOT16
# O32-MM-NEXT: addiu $25, $25, %lo(($tmp4)+8)   # encoding: [0x33,0x39,A,A]
# O32-MM-NEXT:                                  #   fixup A - offset: 0, value: %lo(($tmp4)+8), kind: fixup_MICROMIPS_LO16
# O32-MM-NOT:  .reloc

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  .local forward_local
  jal forward_local
  nop

# Expanding "jal forward_local":
# O32-FIXME: lw     $25, %got(forward_local)($gp)                    # encoding: [0x8f,0x99,A,A]
# O32-FIXME:                                                         #   fixup A - offset: 0, value: %got(forward_local), kind:   fixup_Mips_GOT
# O32-FIXME: addiu  $25, $25, %lo(forward_local)                     # encoding: [0x27,0x39,A,A]
# O32-FIXME::                                                         #   fixup A - offset: 0, value: %lo(forward_local), kind:   fixup_Mips_LO16
# O32-FIXME: .reloc ($tmp6), R_MIPS_JALR, forward_local

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 .text
# ELF-O32-NEXT: 27 39 00 c4 addiu $25, $25, 196
# ELF-O32-NEXT:                 R_MIPS_LO16 .text
# ELF-O32-NEXT: 03 20 f8 09 jalr $25
# ELF-O32-NEXT:                 R_MIPS_JALR forward_local

# N32-FIXME: lw  $25, %got_disp(forward_local)($gp)            # encoding: [0x8f,0x99,A,A]
# N32-FIXME:                                                   #   fixup A - offset: 0, value: %got_disp(forward_local), kind:   fixup_Mips_GOT_DISP

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP forward_local

# N64-FIXME: ld  $25, %got_disp(forward_local)($gp)            # encoding: [0xdf,0x99,A,A]
# N64-FIXME:                                                   #   fixup A - offset: 0, value: %got_disp(forward_local), kind:   fixup_Mips_GOT_DISP

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE forward_local

# O32-MM-FIXME: lw    $25, %got(forward_local)($gp)            # encoding: [0xff,0x3c,A,A]
# O32-MM-FIXME:                                                #   fixup A - offset: 0, value: %got(forward_local), kind:   fixup_MICROMIPS_GOT16
# O32-MM-FIXME: addiu $25, $25, %lo(forward_local)             # encoding: [0x33,0x39,A,A]
# O32-MM-FIXME:                                                #   fixup A - offset: 0, value: %lo(forward_local), kind:   fixup_MICROMIPS_LO16
# O32-MM-FIXME: .reloc ($tmp6), R_MIPS_JALR, forward_local

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  .local forward_local
  jal forward_local+8
  nop

# O32-FIXME: lw     $25, %got(forward_local+8)($gp) # encoding: [0x8f,0x99,A,A]
# O32-FIXME:                                        #   fixup A - offset: 0, value: %got(forward_local+8), kind:   fixup_Mips_GOT
# O32-FIXME: addiu  $25, $25, %lo(forward_local+8)  # encoding: [0x27,0x39,A,A]
# O32-FIXME::                                       #   fixup A - offset: 0, value: %lo(forward_local+8), kind:   fixup_Mips_LO16
# O32-FIXME: .reloc ($tmp7), R_MIPS_JALR, forward_local

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 .text
# ELF-O32-NEXT: 27 39 00 cc addiu $25, $25, 204
# ELF-O32-NEXT:                 R_MIPS_LO16 .text
# ELF-O32-NEXT: 03 20 f8 09 jalr $25
# ELF-O32-NEXT: 00 00 00 00 nop

# N32-FIXME: lw  $25, %got_disp(forward_local)($gp)            # encoding: [0x8f,0x99,A,A]
# N32-FIXME:                                                   #   fixup A - offset: 0, value: %got_disp(forward_local), kind:   fixup_Mips_GOT_DISP

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP forward_local

# N64-FIXME: ld  $25, %got_disp(forward_local)($gp)            # encoding: [0xdf,0x99,A,A]
# N64-FIXME:                                                   #   fixup A - offset: 0, value: %got_disp(forward_local), kind:   fixup_Mips_GOT_DISP

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE forward_local

# O32-MM-FIXME: lw    $25, %got(forward_local)($gp)            # encoding: [0xff,0x3c,A,A]
# O32-MM-FIXME:                                                #   fixup A - offset: 0, value: %got(forward_local), kind:   fixup_MICROMIPS_GOT16
# O32-MM-FIXME: addiu $25, $25, %lo(forward_local)             # encoding: [0x33,0x39,A,A]
# O32-MM-FIXME:                                                #   fixup A - offset: 0, value: %lo(forward_local), kind:   fixup_MICROMIPS_LO16
# O32-MM-FIXME: .reloc ($tmp6), R_MIPS_JALR, forward_local

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  .end local_label

1:
  nop
  add $8, $8, $8
  nop
forward_local:
