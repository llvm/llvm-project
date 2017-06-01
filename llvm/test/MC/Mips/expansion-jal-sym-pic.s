# RUN: llvm-mc %s -arch=mips -mcpu=mips32 -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MIPS,O32

# RUN: llvm-mc %s -arch=mips -mcpu=mips64 -target-abi n32 -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MIPS,N32

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n64 -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MIPS,N64

# RUN: llvm-mc %s -arch=mips -mcpu=mips32 -mattr=micromips -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MM,O32-MM

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n32 -mattr=micromips -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MM,N32-MM

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n64 -mattr=micromips -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MM,N64-MM

# Repeat the tests but using ELF output. An initial version of this patch did
# this as the output different depending on whether it went through
# MCAsmStreamer or MCELFStreamer. This ensures that the assembly expansion and
# direct objection emission match.

# RUN: llvm-mc %s -arch=mips -mcpu=mips32 -filetype=obj | \
# RUN:   llvm-objdump -d -r - | FileCheck %s -check-prefixes=ELF-O32
# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n32 -filetype=obj | \
# RUN:   llvm-objdump -d -r - | FileCheck %s -check-prefixes=ELF-N32
# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n64 -filetype=obj | \
# RUN:   llvm-objdump -d -r - | FileCheck %s -check-prefixes=ELF-N64

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
# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 .text
# ELF-O32-NEXT: 27 39 00 00 addiu $25, $25, 0
# ELF-O32-NEXT:                 R_MIPS_LO16 .text

# N32: lw  $25, %got_disp(local_label)($gp) # encoding: [0x8f,0x99,A,A]
# N32:                                      #   fixup A - offset: 0, value: %got_disp(local_label), kind:   fixup_Mips_GOT_DISP

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE local_label

# N64: ld  $25, %got_disp(local_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64:                                      #   fixup A - offset: 0, value: %got_disp(local_label), kind:   fixup_Mips_GOT_DISP

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE local_label

# O32-MM: lw    $25, %got(local_label)($gp)      # encoding: [0xff,0x3c,A,A]
# O32-MM:                                        #   fixup A - offset: 0, value: %got(local_label), kind:   fixup_MICROMIPS_GOT16
# O32-MM: addiu $25, $25, %lo(local_label)       # encoding: [0x33,0x39,A,A]
# O32-MM:                                        #   fixup A - offset: 0, value: %lo(local_label), kind:   fixup_MICROMIPS_LO16

# N32-MM: lw    $25, %got_disp(local_label)($gp) # encoding: [0xff,0x3c,A,A]
# N32-MM:                                        #   fixup A - offset: 0, value: %got_disp(local_label), kind: fixup_MICROMIPS_GOT_DISP

# N64-MM: ld    $25, %got_disp(local_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64-MM:                                        #   fixup A - offset: 0, value: %got_disp(local_label), kind: fixup_MICROMIPS_GOT_DISP

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  jal weak_label
  nop

# Expanding "jal weak_label":
# O32: lw  $25, %call16(weak_label)($gp) # encoding: [0x8f,0x99,A,A]
# O32:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind:   fixup_Mips_CALL16

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_CALL16 weak_label

# N32: lw  $25, %call16(weak_label)($gp) # encoding: [0x8f,0x99,A,A]
# N32:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind:   fixup_Mips_CALL16

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_CALL16/R_MIPS_NONE/R_MIPS_NONE weak_label

# N64: ld  $25, %call16(weak_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind:   fixup_Mips_CALL16

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_CALL16/R_MIPS_NONE/R_MIPS_NONE weak_label

# O32-MM: lw  $25, %call16(weak_label)($gp) # encoding: [0xff,0x3c,A,A]
# O32-MM:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind:   fixup_MICROMIPS_CALL16

# N32-MM: lw  $25, %call16(weak_label)($gp) # encoding: [0xff,0x3c,A,A]
# N32-MM:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind: fixup_MICROMIPS_CALL16

# N64-MM: ld  $25, %call16(weak_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64-MM:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind: fixup_MICROMIPS_CALL16

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  jal global_label
  nop

# Expanding "jal global_label":
# O32: lw  $25, %call16(global_label)($gp) # encoding: [0x8f,0x99,A,A]
# O32:                                     #   fixup A - offset: 0, value: %call16(global_label), kind:   fixup_Mips_CALL16

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_CALL16 global_label

# N32: lw  $25, %call16(global_label)($gp) # encoding: [0x8f,0x99,A,A]
# N32:                                     #   fixup A - offset: 0, value: %call16(global_label), kind:   fixup_Mips_CALL16

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_CALL16/R_MIPS_NONE/R_MIPS_NONE global_label

# N64: ld  $25, %call16(global_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64:                                     #   fixup A - offset: 0, value: %call16(global_label), kind:   fixup_Mips_CALL16

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_CALL16/R_MIPS_NONE/R_MIPS_NONE global_label

# O32-MM: lw  $25, %call16(global_label)($gp) # encoding: [0xff,0x3c,A,A]
# O32-MM:                                     #   fixup A - offset: 0, value: %call16(global_label), kind: fixup_MICROMIPS_CALL16

# N32-MM: lw  $25, %call16(global_label)($gp) # encoding: [0xff,0x3c,A,A]
# N32-MM:                                     #   fixup A - offset: 0, value: %call16(global_label), kind: fixup_MICROMIPS_CALL16

# N64-MM: ld  $25, %call16(global_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64-MM:                                     #   fixup A - offset: 0, value: %call16(global_label), kind: fixup_MICROMIPS_CALL16

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  jal .text
  nop

# Expanding "jal .text":
# O32: lw	$25, %got(.text)($gp)   # encoding: [0x8f,0x99,A,A]
# O32-NEXT:                                       #   fixup A - offset: 0, value: %got(.text), kind: fixup_Mips_GOT

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 .text

# N32: lw	$25, %got_disp(.text)($gp) # encoding: [0x8f,0x99,A,A]
# N32-NEXT:                                       #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_Mips_GOT_DISP

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE	.text

# N64: ld	$25, %got_disp(.text)($gp) # encoding: [0xdf,0x99,A,A]
# N64-NEXT:                                       #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_Mips_GOT_DISP

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE	.text

# O32-MM: lw    $25, %got(.text)($gp)      # encoding: [0xff,0x3c,A,A]
# O32-MM-NEXT:                                  #   fixup A - offset: 0, value: %got(.text), kind: fixup_MICROMIPS_GOT16
# O32-MM-NEXT: addiu $25, $25, %lo(.text)       # encoding: [0x33,0x39,A,A]
# O32-MM-NEXT:                                  #   fixup A - offset: 0, value: %lo(.text), kind: fixup_MICROMIPS_LO16

# N32-MM: lw    $25, %got_disp(.text)($gp) # encoding: [0xff,0x3c,A,A]
# N32-MM-NEXT:                                  #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_MICROMIPS_GOT_DISP

# N64-MM: ld    $25, %got_disp(.text)($gp) # encoding: [0xdf,0x99,A,A]
# N64-MM-NEXT:                                  #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_MICROMIPS_GOT_DISP

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  # local labels ($tmp symbols)
  jal 1f
  nop

# Expanding "jal 1f":
# O32: lw     $25, %got($tmp0)($gp)   # encoding: [0x8f,0x99,A,A]
# O32:                                #   fixup A - offset: 0, value: %got($tmp0), kind:   fixup_Mips_GOT
# O32: addiu  $25, $25, %lo($tmp0)    # encoding: [0x27,0x39,A,A]
# O32:                                #   fixup A - offset: 0, value: %lo($tmp0), kind:   fixup_Mips_LO16

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 .text
# ELF-O32-NEXT: 27 39 00 58 	addiu	$25, $25, 88
# ELF-O32-NEXT:                 R_MIPS_LO16 .text

# N32: lw  $25, %got_disp($tmp0)($gp) # encoding: [0x8f,0x99,A,A]
# N32:                                #   fixup A - offset: 0, value: %got_disp($tmp0), kind:   fixup_Mips_GOT_DISP

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE .Ltmp0

# N64: ld  $25, %got_disp(.Ltmp0)($gp) # encoding: [0xdf,0x99,A,A]
# N64:                                 #   fixup A - offset: 0, value: %got_disp(.Ltmp0), kind:   fixup_Mips_GOT_DISP

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE .Ltmp0

# O32-MM: lw    $25, %got($tmp0)($gp)    # encoding: [0xff,0x3c,A,A]
# O32-MM:                                #   fixup A - offset: 0, value: %got($tmp0), kind: fixup_MICROMIPS_GOT16
# O32-MM: addiu $25, $25, %lo($tmp0)     # encoding: [0x33,0x39,A,A]
# O32-MM:                                #   fixup A - offset: 0, value: %lo($tmp0), kind: fixup_MICROMIPS_LO16

# N32-MM: lw  $25, %got_disp(.Ltmp0)($gp) # encoding: [0xff,0x3c,A,A]
# N32-MM:                                 #   fixup A - offset: 0, value: %got_disp(.Ltmp0), kind: fixup_MICROMIPS_GOT_DISP

# N64-MM: ld  $25, %got_disp(.Ltmp0)($gp) # encoding: [0xdf,0x99,A,A]
# N64-MM:                                 #   fixup A - offset: 0, value: %got_disp(.Ltmp0), kind: fixup_MICROMIPS_GOT_DISP

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

# ELF-O32:      8f 99 00 00 lw $25, 0($gp)
# ELF-O32-NEXT:                 R_MIPS_GOT16 .text
# ELF-O32-NEXT: 27 39 00 64 	addiu	$25, $25, 100
# ELF-O32-NEXT:                 R_MIPS_LO16 .text

# N32-FIXME: lw  $25, %got_disp(forward_local)($gp)            # encoding: [0x8f,0x99,A,A]
# N32-FIXME:                                                   #   fixup A - offset: 0, value: %got_disp(forward_local), kind:   fixup_Mips_GOT_DISP

# ELF-N32:      8f 99 00 00 lw $25, 0($gp)
# ELF-N32-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE forward_local

# N64-FIXME: ld  $25, %got_disp(forward_local)($gp)            # encoding: [0xdf,0x99,A,A]
# N64-FIXME:                                                   #   fixup A - offset: 0, value: %got_disp(forward_local), kind:   fixup_Mips_GOT_DISP

# ELF-N64:      df 99 00 00 ld $25, 0($gp)
# ELF-N64-NEXT:                 R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE forward_local

# O32-MM-FIXME: lw    $25, %got(forward_local)($gp)            # encoding: [0xff,0x3c,A,A]
# O32-MM-FIXME:                                                #   fixup A - offset: 0, value: %got(forward_local), kind:   fixup_MICROMIPS_GOT16
# O32-MM-FIXME: addiu $25, $25, %lo(forward_local)             # encoding: [0x33,0x39,A,A]
# O32-MM-FIXME:                                                #   fixup A - offset: 0, value: %lo(forward_local), kind:   fixup_MICROMIPS_LO16

# N32-MM-FIXME: lw    $25, %got_disp(forward_local)($gp) # encoding: [0xff,0x3c,A,A]
# N32-MM-FIXME:                                          #   fixup A - offset: 0, value: %got_disp(forward_local), kind: fixup_MICROMIPS_GOT_DISP

# N64-MM-FIXME: ld    $25, %got_disp(forward_local)($gp) # encoding: [0xdf,0x99,A,A]
# N64-MM-FIXME:                                          #   fixup A - offset: 0, value: %got_disp(forward_local), kind: fixup_MICROMIPS_GOT_DISP

# MIPS: jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MM:   jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:  nop           # encoding: [0x00,0x00,0x00,0x00]

  .end local_label

1:
  nop
  add $8, $8, $8
  nop
forward_local:
