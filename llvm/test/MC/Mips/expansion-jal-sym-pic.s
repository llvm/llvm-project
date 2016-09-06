# RUN: llvm-mc %s -arch=mips -mcpu=mips32 -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,NORMAL,O32

# RUN: llvm-mc %s -arch=mips -mcpu=mips64 -target-abi n32 -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,NORMAL,N32

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n64 -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,NORMAL,N64

# RUN: llvm-mc %s -arch=mips -mcpu=mips32 -mattr=micromips -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MICROMIPS,O32-MICROMIPS

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n32 -mattr=micromips -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MICROMIPS,N32-MICROMIPS

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n64 -mattr=micromips -show-encoding |\
# RUN:   FileCheck %s -check-prefixes=ALL,MICROMIPS,N64-MICROMIPS

  .weak weak_label

  .text
  .option pic2

  .ent local_label
local_label:
  .frame  $sp, 0, $ra
  .set noreorder

  jal local_label
  nop

  jal weak_label
  nop

  jal global_label
  nop

  jal .text
  nop

  # local labels ($tmp symbols)
  jal 1f
  nop

  .end local_label

1:
  nop
  add $8, $8, $8
  nop

# Expanding "jal local_label":
# O32: lw     $25, %got(local_label)($gp)   # encoding: [0x8f,0x99,A,A]
# O32:                                      #   fixup A - offset: 0, value: %got(local_label), kind:   fixup_Mips_GOT
# O32: addiu  $25, $25, %lo(local_label)    # encoding: [0x27,0x39,A,A]
# O32:                                      #   fixup A - offset: 0, value: %lo(local_label), kind:   fixup_Mips_LO16

# N32: lw  $25, %got_disp(local_label)($gp) # encoding: [0x8f,0x99,A,A]
# N32:                                      #   fixup A - offset: 0, value: %got_disp(local_label), kind:   fixup_Mips_GOT_DISP

# N64: ld  $25, %got_disp(local_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64:                                      #   fixup A - offset: 0, value: %got_disp(local_label), kind:   fixup_Mips_GOT_DISP

# O32-MICROMIPS: lw    $25, %got(local_label)($gp)      # encoding: [0xff,0x3c,A,A]
# O32-MICROMIPS:                                        #   fixup A - offset: 0, value: %got(local_label), kind:   fixup_MICROMIPS_GOT16
# O32-MICROMIPS: addiu $25, $25, %lo(local_label)       # encoding: [0x33,0x39,A,A]
# O32-MICROMIPS:                                        #   fixup A - offset: 0, value: %lo(local_label), kind:   fixup_MICROMIPS_LO16

# N32-MICROMIPS: lw    $25, %got_disp(local_label)($gp) # encoding: [0xff,0x3c,A,A]
# N32-MICROMIPS:                                        #   fixup A - offset: 0, value: %got_disp(local_label), kind: fixup_MICROMIPS_GOT_DISP

# N64-MICROMIPS: ld    $25, %got_disp(local_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64-MICROMIPS:                                        #   fixup A - offset: 0, value: %got_disp(local_label), kind: fixup_MICROMIPS_GOT_DISP

# NORMAL:    jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MICROMIPS: jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:       nop           # encoding: [0x00,0x00,0x00,0x00]


# Expanding "jal weak_label":
# O32: lw  $25, %call16(weak_label)($gp) # encoding: [0x8f,0x99,A,A]
# O32:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind:   fixup_Mips_CALL16

# N32: lw  $25, %call16(weak_label)($gp) # encoding: [0x8f,0x99,A,A]
# N32:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind:   fixup_Mips_CALL16

# N64: ld  $25, %call16(weak_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind:   fixup_Mips_CALL16

# O32-MICROMIPS: lw  $25, %call16(weak_label)($gp) # encoding: [0xff,0x3c,A,A]
# O32-MICROMIPS:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind:   fixup_MICROMIPS_CALL16

# N32-MICROMIPS: lw  $25, %call16(weak_label)($gp) # encoding: [0xff,0x3c,A,A]
# N32-MICROMIPS:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind: fixup_MICROMIPS_CALL16

# N64-MICROMIPS: ld  $25, %call16(weak_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64-MICROMIPS:                                   #   fixup A - offset: 0, value: %call16(weak_label), kind: fixup_MICROMIPS_CALL16

# NORMAL:    jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MICROMIPS: jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:       nop           # encoding: [0x00,0x00,0x00,0x00]


# Expanding "jal global_label":
# O32: lw  $25, %call16(global_label)($gp) # encoding: [0x8f,0x99,A,A]
# O32:                                     #   fixup A - offset: 0, value: %call16(global_label), kind:   fixup_Mips_CALL16

# N32: lw  $25, %call16(global_label)($gp) # encoding: [0x8f,0x99,A,A]
# N32:                                     #   fixup A - offset: 0, value: %call16(global_label), kind:   fixup_Mips_CALL16

# N64: ld  $25, %call16(global_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64:                                     #   fixup A - offset: 0, value: %call16(global_label), kind:   fixup_Mips_CALL16

# O32-MICROMIPS: lw  $25, %call16(global_label)($gp) # encoding: [0xff,0x3c,A,A]
# O32-MICROMIPS:                                     #   fixup A - offset: 0, value: %call16(global_label), kind: fixup_MICROMIPS_CALL16

# N32-MICROMIPS: lw  $25, %call16(global_label)($gp) # encoding: [0xff,0x3c,A,A]
# N32-MICROMIPS:                                     #   fixup A - offset: 0, value: %call16(global_label), kind: fixup_MICROMIPS_CALL16

# N64-MICROMIPS: ld  $25, %call16(global_label)($gp) # encoding: [0xdf,0x99,A,A]
# N64-MICROMIPS:                                     #   fixup A - offset: 0, value: %call16(global_label), kind: fixup_MICROMIPS_CALL16

# NORMAL:    jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MICROMIPS: jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:       nop           # encoding: [0x00,0x00,0x00,0x00]


# FIXME: The .text section MCSymbol isn't created when printing assembly. However,
# it is created when generating an ELF object file.
# Expanding "jal .text":
# O32-FIXME: lw    $25, %got(.text)($gp)           # encoding: [0x8f,0x99,A,A]
# O32-FIXME:                                       #   fixup A - offset: 0, value: %got(.text), kind: fixup_Mips_GOT
# O32-FIXME: addiu $25, $25, %lo(.text)            # encoding: [0x27,0x39,A,A]
# O32-FIXME:                                       #   fixup A - offset: 0, value: %lo(.text), kind: fixup_Mips_LO16

# N32-FIXME: lw  $25, %got_disp(.text)($gp)        # encoding: [0x8f,0x99,A,A]
# N32-FIXME:                                       #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_Mips_GOT_DISP

# N64-FIXME: ld  $25, %got_disp(.text)($gp)        # encoding: [0xdf,0x99,A,A]
# N64-FIXME:                                       #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_Mips_GOT_DISP

# O32-MICROMIPS-FIXME: lw    $25, %got(.text)($gp)      # encoding: [0xff,0x3c,A,A]
# O32-MICROMIPS-FIXME:                                  #   fixup A - offset: 0, value: %got(.text), kind: fixup_MICROMIPS_GOT16
# O32-MICROMIPS-FIXME: addiu $25, $25, %lo(.text)       # encoding: [0x33,0x39,A,A]
# O32-MICROMIPS-FIXME:                                  #   fixup A - offset: 0, value: %lo(.text), kind: fixup_MICROMIPS_LO16

# N32-MICROMIPS-FIXME: lw    $25, %got_disp(.text)($gp) # encoding: [0xff,0x3c,A,A]
# N32-MICROMIPS-FIXME:                                  #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_MICROMIPS_GOT_DISP

# N64-MICROMIPS-FIXME: ld    $25, %got_disp(.text)($gp) # encoding: [0xdf,0x99,A,A]
# N64-MICROMIPS-FIXME:                                  #   fixup A - offset: 0, value: %got_disp(.text), kind: fixup_MICROMIPS_GOT_DISP

# NORMAL:    jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MICROMIPS: jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:       nop           # encoding: [0x00,0x00,0x00,0x00]


# Expanding "jal 1f":
# O32: lw     $25, %got($tmp0)($gp)   # encoding: [0x8f,0x99,A,A]
# O32:                                #   fixup A - offset: 0, value: %got($tmp0), kind:   fixup_Mips_GOT
# O32: addiu  $25, $25, %lo($tmp0)    # encoding: [0x27,0x39,A,A]
# O32:                                #   fixup A - offset: 0, value: %lo($tmp0), kind:   fixup_Mips_LO16

# N32: lw  $25, %got_disp($tmp0)($gp) # encoding: [0x8f,0x99,A,A]
# N32:                                #   fixup A - offset: 0, value: %got_disp($tmp0), kind:   fixup_Mips_GOT_DISP

# N64: ld  $25, %got_disp(.Ltmp0)($gp) # encoding: [0xdf,0x99,A,A]
# N64:                                 #   fixup A - offset: 0, value: %got_disp(.Ltmp0), kind:   fixup_Mips_GOT_DISP

# O32-MICROMIPS: lw    $25, %got($tmp0)($gp)    # encoding: [0xff,0x3c,A,A]
# O32-MICROMIPS:                                #   fixup A - offset: 0, value: %got($tmp0), kind: fixup_MICROMIPS_GOT16
# O32-MICROMIPS: addiu $25, $25, %lo($tmp0)     # encoding: [0x33,0x39,A,A]
# O32-MICROMIPS:                                #   fixup A - offset: 0, value: %lo($tmp0), kind: fixup_MICROMIPS_LO16

# N32-MICROMIPS: lw  $25, %got_disp(.Ltmp0)($gp) # encoding: [0xff,0x3c,A,A]
# N32-MICROMIPS:                                 #   fixup A - offset: 0, value: %got_disp(.Ltmp0), kind: fixup_MICROMIPS_GOT_DISP

# N64-MICROMIPS: ld  $25, %got_disp(.Ltmp0)($gp) # encoding: [0xdf,0x99,A,A]
# N64-MICROMIPS:                                 #   fixup A - offset: 0, value: %got_disp(.Ltmp0), kind: fixup_MICROMIPS_GOT_DISP

# NORMAL:    jalr $25      # encoding: [0x03,0x20,0xf8,0x09]
# MICROMIPS: jalr $ra, $25 # encoding: [0x03,0xf9,0x0f,0x3c]
# ALL:       nop           # encoding: [0x00,0x00,0x00,0x00]
