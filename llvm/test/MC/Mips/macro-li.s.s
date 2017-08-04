# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -target-abi=o32 | FileCheck %s --check-prefixes=ALL,O32-N32-NO-PIC
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -target-abi=o32 -position-independent | FileCheck %s --check-prefixes=ALL,O32-N32-PIC
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips64 -target-abi=n32 | FileCheck %s --check-prefixes=ALL,O32-N32-NO-PIC
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips64 -target-abi=n32 -position-independent | FileCheck %s --check-prefixes=ALL,O32-N32-PIC
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips64 -target-abi=n64 | FileCheck %s --check-prefixes=ALL,N64-NO-PIC
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips64 -target-abi=n64 -position-independent | FileCheck %s --check-prefixes=ALL,N64-PIC

li.s	$4, 0
# ALL:   addiu   $4, $zero, 0                # encoding: [0x00,0x00,0x04,0x24]

li.s	$4, 0.0
# ALL:   addiu   $4, $zero, 0                # encoding: [0x00,0x00,0x04,0x24]

li.s	$4, 1.12345
# ALL:   lui     $4, 16271                   # encoding: [0x8f,0x3f,0x04,0x3c]
# ALL:   ori     $4, $4, 52534               # encoding: [0x36,0xcd,0x84,0x34]

li.s	$4, 1
# ALL:   lui     $4, 16256                   # encoding: [0x80,0x3f,0x04,0x3c]

li.s	$4, 1.0
# ALL:   lui     $4, 16256                   # encoding: [0x80,0x3f,0x04,0x3c]

li.s	$4, 12345678910
# ALL:   lui     $4, 20535                   # encoding: [0x37,0x50,0x04,0x3c]
# ALL:   ori     $4, $4, 63239               # encoding: [0x07,0xf7,0x84,0x34]

li.s	$4, 12345678910.0
# ALL:   lui     $4, 20535                   # encoding: [0x37,0x50,0x04,0x3c]
# ALL:   ori     $4, $4, 63239               # encoding: [0x07,0xf7,0x84,0x34]

li.s	$4, 0.4
# ALL:   lui     $4, 16076                   # encoding: [0xcc,0x3e,0x04,0x3c]
# ALL:   ori     $4, $4, 52429               # encoding: [0xcd,0xcc,0x84,0x34]

li.s	$4, 1.5
# ALL:   lui     $4, 16320                   # encoding: [0xc0,0x3f,0x04,0x3c]

li.s	$4, 12345678910.12345678910
# ALL:   lui     $4, 20535                   # encoding: [0x37,0x50,0x04,0x3c]
# ALL:   ori     $4, $4, 63239               # encoding: [0x07,0xf7,0x84,0x34]

li.s	$4, 12345678910123456789.12345678910
# ALL:   lui     $4, 24363                   # encoding: [0x2b,0x5f,0x04,0x3c]
# ALL:   ori     $4, $4, 21674               # encoding: [0xaa,0x54,0x84,0x34]

li.s	$f4, 0
# ALL:   addiu   $1, $zero, 0                # encoding: [0x00,0x00,0x01,0x24]
# ALL:   mtc1    $1, $f4                     # encoding: [0x00,0x20,0x81,0x44]

li.s	$f4, 0.0
# ALL:   addiu   $1, $zero, 0                # encoding: [0x00,0x00,0x01,0x24]
# ALL:   mtc1    $1, $f4                     # encoding: [0x00,0x20,0x81,0x44]

li.s	$f4, 1.12345
# ALL:	.section	.rodata,"a",@progbits
# ALL:  [[LABEL:\$tmp[0-9]+]]:
# ALL:	.4byte	1066388790
# ALL:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             lwc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xc4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16

li.s	$f4, 1
# ALL:   lui     $1, 16256                   # encoding: [0x80,0x3f,0x01,0x3c]
# ALL:   mtc1    $1, $f4                     # encoding: [0x00,0x20,0x81,0x44]

li.s	$f4, 1.0
# ALL:   lui     $1, 16256                   # encoding: [0x80,0x3f,0x01,0x3c]
# ALL:   mtc1    $1, $f4                     # encoding: [0x00,0x20,0x81,0x44]

li.s	$f4, 12345678910
# ALL:	.section	.rodata,"a",@progbits
# ALL:  [[LABEL:\$tmp[0-9]+]]:
# ALL:	.4byte	1345844999
# ALL:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             lwc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xc4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16

li.s	$f4, 12345678910.0
# ALL:	.section	.rodata,"a",@progbits
# ALL:  [[LABEL:\$tmp[0-9]+]]:
# ALL:	.4byte	1345844999
# ALL:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             lwc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xc4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16


li.s	$f4, 0.4
# ALL:	.section	.rodata,"a",@progbits
# ALL:  [[LABEL:\$tmp[0-9]+]]:
# ALL:	.4byte	1053609165
# ALL:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             lwc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xc4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16

li.s	$f4, 1.5
# ALL:   lui     $1, 16320                   # encoding: [0xc0,0x3f,0x01,0x3c]
# ALL:   mtc1    $1, $f4                     # encoding: [0x00,0x20,0x81,0x44]

li.s	$f4, 12345678910.12345678910
# ALL:	.section	.rodata,"a",@progbits
# ALL:  [[LABEL:\$tmp[0-9]+]]:
# ALL:	.4byte	1345844999
# ALL:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             lwc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xc4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16

li.s	$f4, 12345678910123456789.12345678910
# ALL:	.section	.rodata,"a",@progbits
# ALL:  [[LABEL:\$tmp[0-9]+]]:
# ALL:	.4byte	1596675242
# ALL:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             lwc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xc4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
