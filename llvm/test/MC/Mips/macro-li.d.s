# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32 -target-abi=o32  | FileCheck %s --check-prefixes=ALL,O32-N32-NO-PIC,O32
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 -target-abi=o32 | FileCheck %s --check-prefixes=ALL,CHECK-MIPS32r2
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32 -target-abi=o32 -position-independent | FileCheck %s --check-prefixes=ALL,O32-N32-PIC,O32
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips64 -target-abi=n32 | FileCheck %s --check-prefixes=ALL,O32-N32-NO-PIC,N32-N64
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips64 -target-abi=n32 -position-independent | FileCheck %s --check-prefixes=ALL,O32-N32-PIC,N32-N64
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips64 -target-abi=n64 | FileCheck %s --check-prefixes=ALL,N64-NO-PIC,N32-N64
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips64 -target-abi=n64 -position-independent | FileCheck %s --check-prefixes=ALL,N64-PIC,N32-N64

li.d	$4, 0
# O32:     addiu   $4, $zero, 0                # encoding: [0x00,0x00,0x04,0x24]
# O32:     addiu   $5, $zero, 0                # encoding: [0x00,0x00,0x05,0x24]
# N32-N64: addiu   $4, $zero, 0                # encoding: [0x00,0x00,0x04,0x24]

li.d	$4, 0.0
# O32:     addiu   $4, $zero, 0                # encoding: [0x00,0x00,0x04,0x24]
# O32:     addiu   $5, $zero, 0                # encoding: [0x00,0x00,0x05,0x24]
# N32-N64: addiu   $4, $zero, 0                # encoding: [0x00,0x00,0x04,0x24]

li.d	$4, 1.12345
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4607738388174016296
# ALL-NEXT:	.text
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# O32-N32-NO-PIC:  addiu   $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x24]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-PIC:     addiu   $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x24]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         daddiu  $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-PIC:                                            #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# O32:             lw      $4, 0($1)                  # encoding: [0x00,0x00,0x24,0x8c]
# O32:             lw      $5, 4($1)                  # encoding: [0x04,0x00,0x25,0x8c]
# N32-N64:         ld      $4, 0($1)                  # encoding: [0x00,0x00,0x24,0xdc]

li.d	$4, 1
# O32:     lui     $4, 16368                          # encoding: [0xf0,0x3f,0x04,0x3c]
# O32:     addiu   $5, $zero, 0                       # encoding: [0x00,0x00,0x05,0x24]
# N32-N64: ori     $4, $zero, 65472                   # encoding: [0xc0,0xff,0x04,0x34]
# N32-N64: dsll    $4, $4, 46                         # encoding: [0xbc,0x23,0x04,0x00]

li.d	$4, 1.0
# O32:     lui     $4, 16368                          # encoding: [0xf0,0x3f,0x04,0x3c]
# O32:     addiu   $5, $zero, 0                       # encoding: [0x00,0x00,0x05,0x24]
# N32-N64: ori     $4, $zero, 65472                   # encoding: [0xc0,0xff,0x04,0x34]
# N32-N64: dsll    $4, $4, 46                         # encoding: [0xbc,0x23,0x04,0x00]

li.d	$4, 12345678910
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4757770298180239360
# ALL-NEXT:	.text
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# O32-N32-NO-PIC:  addiu   $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x24]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-PIC:     addiu   $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x24]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         daddiu  $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-PIC:                                            #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# O32:             lw      $4, 0($1)                  # encoding: [0x00,0x00,0x24,0x8c]
# O32:             lw      $5, 4($1)                  # encoding: [0x04,0x00,0x25,0x8c]
# N32-N64:         ld      $4, 0($1)                  # encoding: [0x00,0x00,0x24,0xdc]

li.d	$4, 12345678910.0
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4757770298180239360
# ALL-NEXT:	.text
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# O32-N32-NO-PIC:  addiu   $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x24]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-PIC:     addiu   $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x24]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         daddiu  $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-PIC:                                            #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# O32:             lw      $4, 0($1)                  # encoding: [0x00,0x00,0x24,0x8c]
# O32:             lw      $5, 4($1)                  # encoding: [0x04,0x00,0x25,0x8c]
# N32-N64:         ld      $4, 0($1)                  # encoding: [0x00,0x00,0x24,0xdc]

li.d	$4, 0.4
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4600877379321698714
# ALL-NEXT:	.text
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# O32-N32-NO-PIC:  addiu   $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x24]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-PIC:     addiu   $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x24]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         daddiu  $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-PIC:                                            #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# O32:             lw      $4, 0($1)                  # encoding: [0x00,0x00,0x24,0x8c]
# O32:             lw      $5, 4($1)                  # encoding: [0x04,0x00,0x25,0x8c]
# N32-N64:         ld      $4, 0($1)                  # encoding: [0x00,0x00,0x24,0xdc]

li.d	$4, 1.5
# O32:     lui     $4, 16376                          # encoding: [0xf8,0x3f,0x04,0x3c]
# O32:     addiu   $5, $zero, 0                       # encoding: [0x00,0x00,0x05,0x24]
# N32-N64: ori     $4, $zero, 65504                   # encoding: [0xe0,0xff,0x04,0x34]
# N32-N64: dsll    $4, $4, 46                         # encoding: [0xbc,0x23,0x04,0x00]

li.d	$4, 12345678910.12345678910
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4757770298180304087
# ALL-NEXT:	.text
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# O32-N32-NO-PIC:  addiu   $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x24]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-PIC:     addiu   $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x24]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         daddiu  $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-PIC:                                            #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# O32:             lw      $4, 0($1)                  # encoding: [0x00,0x00,0x24,0x8c]
# O32:             lw      $5, 4($1)                  # encoding: [0x04,0x00,0x25,0x8c]
# N32-N64:         ld      $4, 0($1)                  # encoding: [0x00,0x00,0x24,0xdc]


li.d	$4, 12345678910123456789.12345678910
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4892433759227321879
# ALL-NEXT:	.text
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# O32-N32-NO-PIC:  addiu   $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x24]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-PIC:     addiu   $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x24]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         daddiu  $1, $1, %lo([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-PIC:                                            #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
# O32:             lw      $4, 0($1)                  # encoding: [0x00,0x00,0x24,0x8c]
# O32:             lw      $5, 4($1)                  # encoding: [0x04,0x00,0x25,0x8c]
# N32-N64:         ld      $4, 0($1)                  # encoding: [0x00,0x00,0x24,0xdc]

li.d	$f4, 0
# O32:            mtc1    $zero, $f5         # encoding: [0x00,0x28,0x80,0x44]
# O32:            mtc1    $zero, $f4         # encoding: [0x00,0x20,0x80,0x44]
# CHECK-MIPS32r2: mtc1    $zero, $f4         # encoding: [0x00,0x20,0x80,0x44]
# CHECK-MIPS32r2: mthc1   $zero, $f4         # encoding: [0x00,0x20,0xe0,0x44]
# N32-N64:        dmtc1   $zero, $f4         # encoding: [0x00,0x20,0xa0,0x44]

li.d	$f4, 0.0
# O32:            mtc1    $zero, $f5         # encoding: [0x00,0x28,0x80,0x44]
# O32:            mtc1    $zero, $f4         # encoding: [0x00,0x20,0x80,0x44]
# CHECK-MIPS32r2: mtc1    $zero, $f4         # encoding: [0x00,0x20,0x80,0x44]
# CHECK-MIPS32r2: mthc1   $zero, $f4         # encoding: [0x00,0x20,0xe0,0x44]
# N32-N64:        dmtc1   $zero, $f4         # encoding: [0x00,0x20,0xa0,0x44]

li.d	$f4, 1.12345
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4607738388174016296
# ALL-NEXT:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             ldc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xd4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16

li.d	$f4, 1
# O32:            lui     $1, 16368          # encoding: [0xf0,0x3f,0x01,0x3c]
# O32:            mtc1    $1, $f5            # encoding: [0x00,0x28,0x81,0x44]
# O32:            mtc1    $zero, $f4         # encoding: [0x00,0x20,0x80,0x44]
# CHECK-MIPS32r2: lui     $1, 16368          # encoding: [0xf0,0x3f,0x01,0x3c]
# CHECK-MIPS32r2: mtc1    $zero, $f4         # encoding: [0x00,0x20,0x80,0x44]
# CHECK-MIPS32r2: mthc1   $1, $f4            # encoding: [0x00,0x20,0xe1,0x44]
# N32-N64:        ori     $1, $zero, 65472   # encoding: [0xc0,0xff,0x01,0x34]
# N32-N64:        dsll    $1, $1, 46         # encoding: [0xbc,0x0b,0x01,0x00]
# N32-N64:        dmtc1   $1, $f4            # encoding: [0x00,0x20,0xa1,0x44]

li.d	$f4, 1.0
# O32:            lui     $1, 16368          # encoding: [0xf0,0x3f,0x01,0x3c]
# O32:            mtc1    $1, $f5            # encoding: [0x00,0x28,0x81,0x44]
# O32:            mtc1    $zero, $f4         # encoding: [0x00,0x20,0x80,0x44]
# CHECK-MIPS32r2: lui     $1, 16368          # encoding: [0xf0,0x3f,0x01,0x3c]
# CHECK-MIPS32r2: mtc1    $zero, $f4         # encoding: [0x00,0x20,0x80,0x44]
# CHECK-MIPS32r2: mthc1   $1, $f4            # encoding: [0x00,0x20,0xe1,0x44]
# N32-N64:        ori     $1, $zero, 65472   # encoding: [0xc0,0xff,0x01,0x34]
# N32-N64:        dsll    $1, $1, 46         # encoding: [0xbc,0x0b,0x01,0x00]
# N32-N64:        dmtc1   $1, $f4            # encoding: [0x00,0x20,0xa1,0x44]

li.d	$f4, 12345678910
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4757770298180239360
# ALL-NEXT:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             ldc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xd4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16

li.d	$f4, 12345678910.0
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4757770298180239360
# ALL-NEXT:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             ldc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xd4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16

li.d	$f4, 0.4
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4600877379321698714
# ALL-NEXT:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             ldc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xd4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16

li.d	$f4, 1.5
# O32:            lui     $1, 16376          # encoding: [0xf8,0x3f,0x01,0x3c]
# O32:            mtc1    $1, $f5            # encoding: [0x00,0x28,0x81,0x44]
# O32:            mtc1    $zero, $f4         # encoding: [0x00,0x20,0x80,0x44]
# CHECK-MIPS32r2: lui     $1, 16376          # encoding: [0xf8,0x3f,0x01,0x3c]
# CHECK-MIPS32r2: mtc1    $zero, $f4         # encoding: [0x00,0x20,0x80,0x44]
# CHECK-MIPS32r2: mthc1   $1, $f4            # encoding: [0x00,0x20,0xe1,0x44]
# N32-N64:        ori     $1, $zero, 65504   # encoding: [0xe0,0xff,0x01,0x34]
# N32-N64:        dsll    $1, $1, 46         # encoding: [0xbc,0x0b,0x01,0x00]
# N32-N64:        dmtc1   $1, $f4            # encoding: [0x00,0x20,0xa1,0x44]

li.d	$f4, 2.5
# O32:            lui     $1, 16388          # encoding: [0x04,0x40,0x01,0x3c]
# O32:            mtc1    $1, $f5            # encoding: [0x00,0x28,0x81,0x44]
# O32:            mtc1    $zero, $f4         # encoding: [0x00,0x20,0x80,0x44]
# CHECK-MIPS32r2: lui     $1, 16388          # encoding: [0x04,0x40,0x01,0x3c]
# CHECK-MIPS32r2: mtc1    $zero, $f4         # encoding: [0x00,0x20,0x80,0x44]
# CHECK-MIPS32r2: mthc1   $1, $f4            # encoding: [0x00,0x20,0xe1,0x44]
# N32-N64:        ori     $1, $zero, 32776   # encoding: [0x08,0x80,0x01,0x34]
# N32-N64:        dsll    $1, $1, 47         # encoding: [0xfc,0x0b,0x01,0x00]
# N32-N64:        dmtc1   $1, $f4            # encoding: [0x00,0x20,0xa1,0x44]

li.d	$f4, 2.515625
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4612847102706319360
# ALL-NEXT:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             ldc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xd4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16

li.d	$f4, 12345678910.12345678910
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4757770298180304087
# ALL-NEXT:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             ldc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xd4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16

li.d	$f4, 12345678910123456789.12345678910
# ALL:      .section  .rodata,"a",@progbits
# ALL-NEXT:  [[LABEL:((\$)|(\.L))tmp[0-9]+]]:
# ALL-NEXT:	.p2align 3
# ALL-NEXT:	.8byte 4892433759227321879
# ALL-NEXT:	.text
# O32-N32-PIC:     lw      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0x8f]
# O32-N32-PIC:                                        #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# N64-PIC:         ld      $1, %got([[LABEL]])($gp)   # encoding: [A,A,0x81,0xdf]
# N64-PIC:                                            #   fixup A - offset: 0, value: %got([[LABEL]]), kind: fixup_Mips_GOT
# O32-N32-NO-PIC:  lui     $1, %hi([[LABEL]])         # encoding: [A,A,0x01,0x3c]
# O32-N32-NO-PIC:                                     #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      lui     $1, %highest([[LABEL]])    # encoding: [A,A,0x01,0x3c]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %highest([[LABEL]]), kind: fixup_Mips_HIGHEST
# N64-NO-PIC:      daddiu  $1, $1, %higher([[LABEL]]) # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %higher([[LABEL]]), kind: fixup_Mips_HIGHER
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# N64-NO-PIC:      daddiu  $1, $1, %hi([[LABEL]])     # encoding: [A,A,0x21,0x64]
# N64-NO-PIC:                                         #   fixup A - offset: 0, value: %hi([[LABEL]]), kind: fixup_Mips_HI16
# N64-NO-PIC:      dsll    $1, $1, 16                 # encoding: [0x38,0x0c,0x01,0x00]
# ALL:             ldc1    $f4, %lo([[LABEL]])($1)    # encoding: [A,A,0x24,0xd4]
# ALL:                                                #   fixup A - offset: 0, value: %lo([[LABEL]]), kind: fixup_Mips_LO16
