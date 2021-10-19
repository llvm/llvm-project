# RUN: llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r2 | \
# RUN:   FileCheck %s --check-prefix=CHECK-NOTRAP
# RUN: llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips64r2 \
# RUN:  -mattr=+use-tcc-in-div | FileCheck %s --check-prefix=CHECK-TRAP

  ddiv $25,$11
# CHECK-NOTRAP: bne $11, $zero, .Ltmp0    # encoding: [0x15,0x60,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: .Ltmp0-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: ddiv $zero, $25, $11      # encoding: [0x03,0x2b,0x00,0x1e]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: .Ltmp0
# CHECK-NOTRAP: addiu $1, $zero, -1       # encoding: [0x24,0x01,0xff,0xff]
# CHECK-NOTRAP: bne $11, $1, .Ltmp1       # encoding: [0x15,0x61,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: .Ltmp1-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: addiu $1, $zero, 1        # encoding: [0x24,0x01,0x00,0x01]
# CHECK-NOTRAP: dsll32 $1, $1, 31         # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-NOTRAP: bne $25, $1, .Ltmp1       # encoding: [0x17,0x21,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: .Ltmp1-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: nop                       # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NOTRAP: break 6                   # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-NOTRAP: .Ltmp1
# CHECK-NOTRAP: mflo $25                  # encoding: [0x00,0x00,0xc8,0x12]

# CHECK-TRAP: teq $11, $zero, 7           # encoding: [0x01,0x60,0x01,0xf4]
# CHECK-TRAP: ddiv $zero, $25, $11        # encoding: [0x03,0x2b,0x00,0x1e]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0x24,0x01,0xff,0xff]
# CHECK-TRAP: bne $11, $1, .Ltmp0         # encoding: [0x15,0x61,A,A]
# CHECK-TRAP:                             # fixup A - offset: 0, value: .Ltmp0-4, kind: fixup_Mips_PC16
# CHECK-TRAP: addiu $1, $zero, 1          # encoding: [0x24,0x01,0x00,0x01]
# CHECK-TRAP: dsll32 $1, $1, 31           # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-TRAP: teq $25, $1, 6              # encoding: [0x03,0x21,0x01,0xb4]
# CHECK-TRAP: .Ltmp0:
# CHECK-TRAP: mflo $25                    # encoding: [0x00,0x00,0xc8,0x12]

  ddiv $24,$12
# CHECK-NOTRAP: bne $12, $zero, .Ltmp2    # encoding: [0x15,0x80,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: .Ltmp2-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: ddiv $zero, $24, $12      # encoding: [0x03,0x0c,0x00,0x1e]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: .Ltmp2:
# CHECK-NOTRAP: addiu $1, $zero, -1       # encoding: [0x24,0x01,0xff,0xff]
# CHECK-NOTRAP: bne $12, $1, .Ltmp3       # encoding: [0x15,0x81,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: .Ltmp3-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: addiu $1, $zero, 1        # encoding: [0x24,0x01,0x00,0x01]
# CHECK-NOTRAP: dsll32 $1, $1, 31         # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-NOTRAP: bne $24, $1, .Ltmp3       # encoding: [0x17,0x01,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: .Ltmp3-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: nop                       # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NOTRAP: break 6                   # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-NOTRAP: .Ltmp3
# CHECK-NOTRAP: mflo $24                  # encoding: [0x00,0x00,0xc0,0x12]

# CHECK-TRAP: teq $12, $zero, 7           # encoding: [0x01,0x80,0x01,0xf4]
# CHECK-TRAP: ddiv $zero, $24, $12        # encoding: [0x03,0x0c,0x00,0x1e]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0x24,0x01,0xff,0xff]
# CHECK-TRAP: bne $12, $1, .Ltmp1         # encoding: [0x15,0x81,A,A]
# CHECK-TRAP:                             # fixup A - offset: 0, value: .Ltmp1-4, kind: fixup_Mips_PC16
# CHECK-TRAP: addiu $1, $zero, 1          # encoding: [0x24,0x01,0x00,0x01]
# CHECK-TRAP: dsll32 $1, $1, 31           # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-TRAP: teq $24, $1, 6              # encoding: [0x03,0x01,0x01,0xb4]
# CHECK-TRAP: .Ltmp1:
# CHECK-TRAP: mflo $24                    # encoding: [0x00,0x00,0xc0,0x12]

  ddiv $25,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  ddiv $0,$9
# CHECK-NOTRAP: ddiv $zero, $zero, $9     # encoding: [0x00,0x09,0x00,0x1e]

# CHECK-TRAP: teq $9, $zero, 7            # encoding: [0x01,0x20,0x01,0xf4]
# CHECK-TRAP: ddiv $zero, $zero, $9       # encoding: [0x00,0x09,0x00,0x1e]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0x24,0x01,0xff,0xff]
# CHECK-TRAP: bne $9, $1, .Ltmp2          # encoding: [0x15,0x21,A,A]
# CHECK-TRAP:                             # fixup A - offset: 0, value: .Ltmp2-4, kind: fixup_Mips_PC16
# CHECK-TRAP: addiu $1, $zero, 1          # encoding: [0x24,0x01,0x00,0x01]
# CHECK-TRAP: dsll32 $1, $1, 31           # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-TRAP: teq $zero, $1, 6            # encoding: [0x00,0x01,0x01,0xb4]
# CHECK-TRAP: .Ltmp2:
# CHECK-TRAP: mflo $zero                  # encoding: [0x00,0x00,0x00,0x12]

  ddiv $0,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  ddiv $4,0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  ddiv $0,0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  ddiv $4,1
# CHECK-NOTRAP: move $4, $4               # encoding: [0x00,0x80,0x20,0x25]
# CHECK-TRAP: move $4, $4                 # encoding: [0x00,0x80,0x20,0x25]

  ddiv $4,-1
# CHECK-NOTRAP: dneg $4, $4               # encoding: [0x00,0x04,0x20,0x2e]
# CHECK-TRAP: dneg $4, $4                 # encoding: [0x00,0x04,0x20,0x2e]

  ddiv $4,2
# CHECK-NOTRAP: addiu $1, $zero, 2        # encoding: [0x24,0x01,0x00,0x02]
# CHECK-NOTRAP: ddiv $zero, $4, $1        # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: addiu $1, $zero, 2          # encoding: [0x24,0x01,0x00,0x02]
# CHECK-TRAP: ddiv $zero, $4, $1          # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,0x8000
# CHECK-NOTRAP: ori $1, $zero, 32768      # encoding: [0x34,0x01,0x80,0x00]
# CHECK-NOTRAP: ddiv $zero, $4, $1        # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: ori $1, $zero, 32768        # encoding: [0x34,0x01,0x80,0x00]
# CHECK-TRAP: ddiv $zero, $4, $1          # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,-0x8000
# CHECK-NOTRAP: addiu $1, $zero, -32768   # encoding: [0x24,0x01,0x80,0x00]
# CHECK-NOTRAP: ddiv $zero, $4, $1        # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: addiu $1, $zero, -32768     # encoding: [0x24,0x01,0x80,0x00]
# CHECK-TRAP: ddiv $zero, $4, $1          # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,0x10000
# CHECK-NOTRAP: lui $1, 1                 # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-NOTRAP: ddiv $zero, $4, $1        # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-TRAP: ddiv $zero, $4, $1          # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,0x1a5a5
# CHECK-NOTRAP: lui $1, 1                 # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-NOTRAP: ori $1, $1, 42405         # encoding: [0x34,0x21,0xa5,0xa5]
# CHECK-NOTRAP: ddiv $zero, $4, $1        # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-TRAP: ori $1, $1, 42405           # encoding: [0x34,0x21,0xa5,0xa5]
# CHECK-TRAP: ddiv $zero, $4, $1          # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,0xfffffff
# CHECK-NOTRAP: lui $1, 4095              # encoding: [0x3c,0x01,0x0f,0xff]
# CHECK-NOTRAP: ori $1, $1, 65535         # encoding: [0x34,0x21,0xff,0xff]
# CHECK-NOTRAP: ddiv $zero, $4, $1        # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 4095                # encoding: [0x3c,0x01,0x0f,0xff]
# CHECK-TRAP: ori $1, $1, 65535           # encoding: [0x34,0x21,0xff,0xff]
# CHECK-TRAP: ddiv $zero, $4, $1          # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,0x10000000
# CHECK-NOTRAP: lui $1, 4096              # encoding: [0x3c,0x01,0x10,0x00]
# CHECK-NOTRAP: ddiv $zero, $4, $1        # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 4096                # encoding: [0x3c,0x01,0x10,0x00]
# CHECK-TRAP: ddiv $zero, $4, $1          # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,0xfffffffe
# CHECK-NOTRAP: ori $1, $zero, 65535      # encoding: [0x34,0x01,0xff,0xff]
# CHECK-NOTRAP: dsll $1, $1, 16           # encoding: [0x00,0x01,0x0c,0x38]
# CHECK-NOTRAP: ori $1, $1, 65534         # encoding: [0x34,0x21,0xff,0xfe]
# CHECK-NOTRAP: ddiv $zero, $4, $1        # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: ori $1, $zero, 65535        # encoding: [0x34,0x01,0xff,0xff]
# CHECK-TRAP: dsll $1, $1, 16             # encoding: [0x00,0x01,0x0c,0x38]
# CHECK-TRAP: ori $1, $1, 65534           # encoding: [0x34,0x21,0xff,0xfe]
# CHECK-TRAP: ddiv $zero, $4, $1          # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]


  ddiv $4,0xffffffff
# CHECK-NOTRAP: lui $1, 65535             # encoding: [0x3c,0x01,0xff,0xff]
# CHECK-NOTRAP: dsrl32 $1, $1, 0          # encoding: [0x00,0x01,0x08,0x3e]
# CHECK-NOTRAP: ddiv $zero, $4, $1        # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 65535               # encoding: [0x3c,0x01,0xff,0xff]
# CHECK-TRAP: dsrl32 $1, $1, 0            # encoding: [0x00,0x01,0x08,0x3e]
# CHECK-TRAP: ddiv $zero, $4, $1          # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,0xfffffffff
# CHECK-NOTRAP: addiu  $1, $zero, 15      # encoding: [0x24,0x01,0x00,0x0f]
# CHECK-NOTRAP: dsll $1, $1, 16           # encoding: [0x00,0x01,0x0c,0x38]
# CHECK-NOTRAP: ori $1, $1, 65535         # encoding: [0x34,0x21,0xff,0xff]
# CHECK-NOTRAP: dsll $1, $1, 16           # encoding: [0x00,0x01,0x0c,0x38]
# CHECK-NOTRAP: ori $1, $1, 65535         # encoding: [0x34,0x21,0xff,0xff]
# CHECK-NOTRAP: ddiv $zero, $4, $1        # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: addiu  $1, $zero, 15        # encoding: [0x24,0x01,0x00,0x0f]
# CHECK-TRAP: dsll $1, $1, 16             # encoding: [0x00,0x01,0x0c,0x38]
# CHECK-TRAP: ori $1, $1, 65535           # encoding: [0x34,0x21,0xff,0xff]
# CHECK-TRAP: dsll $1, $1, 16             # encoding: [0x00,0x01,0x0c,0x38]
# CHECK-TRAP: ori $1, $1, 65535           # encoding: [0x34,0x21,0xff,0xff]
# CHECK-TRAP: ddiv $zero, $4, $1          # encoding: [0x00,0x81,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,$5,$6
# CHECK-NOTRAP: bne $6, $zero, .Ltmp6     # encoding: [0x14,0xc0,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: .Ltmp6-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: ddiv $zero, $5, $6        # encoding: [0x00,0xa6,0x00,0x1e]
# CHECK-NOTRAP: break  7                  # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: .Ltmp6:
# CHECK-NOTRAP: addiu $1, $zero, -1       # encoding: [0x24,0x01,0xff,0xff]
# CHECK-NOTRAP: bne $6, $1, .Ltmp7        # encoding: [0x14,0xc1,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: .Ltmp7-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: addiu $1, $zero, 1        # encoding: [0x24,0x01,0x00,0x01]
# CHECK-NOTRAP: dsll32 $1, $1, 31         # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-NOTRAP: bne $5, $1, .Ltmp7        # encoding: [0x14,0xa1,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: .Ltmp7-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: nop                       # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NOTRAP: break  6                  # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-NOTRAP: .Ltmp7:
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]

# CHECK-TRAP: teq $6, $zero, 7            # encoding: [0x00,0xc0,0x01,0xf4]
# CHECK-TRAP: ddiv $zero, $5, $6          # encoding: [0x00,0xa6,0x00,0x1e]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0x24,0x01,0xff,0xff]
# CHECK-TRAP: bne $6, $1, .Ltmp3          # encoding: [0x14,0xc1,A,A]
# CHECK-TRAP:                             # fixup A - offset: 0, value: .Ltmp3-4, kind: fixup_Mips_PC16
# CHECK-TRAP: addiu $1, $zero, 1          # encoding: [0x24,0x01,0x00,0x01]
# CHECK-TRAP: dsll32 $1, $1, 31           # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-TRAP: teq $5, $1, 6               # encoding: [0x00,0xa1,0x01,0xb4]
# CHECK-TRAP: .Ltmp3:
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,$5,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  ddiv $4,$0,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  ddiv $0,$4,$5
# CHECK-NOTRAP: ddiv $zero, $4, $5        # encoding: [0x00,0x85,0x00,0x1e]
# CHECK-TRAP: ddiv $zero, $4, $5          # encoding: [0x00,0x85,0x00,0x1e]

  ddiv $4,$0,0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  ddiv $4,$5,0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  ddiv $4,$5,1
# CHECK-NOTRAP: move $4, $5               # encoding: [0x00,0xa0,0x20,0x25]
# CHECK-TRAP: move     $4, $5             # encoding: [0x00,0xa0,0x20,0x25]

  ddiv $4,$5,-1
# CHECK-NOTRAP: dneg $4, $5               # encoding: [0x00,0x05,0x20,0x2e]
# CHECK-TRAP: dneg    $4, $5              # encoding: [0x00,0x05,0x20,0x2e]

  ddiv $4,$5,2
# CHECK-NOTRAP: addiu $1, $zero, 2        # encoding: [0x24,0x01,0x00,0x02]
# CHECK-NOTRAP: ddiv $zero, $5, $1        # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: addiu $1, $zero, 2          # encoding: [0x24,0x01,0x00,0x02]
# CHECK-TRAP: ddiv $zero, $5, $1          # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,$5,0x8000
# CHECK-NOTRAP: ori $1, $zero, 32768      # encoding: [0x34,0x01,0x80,0x00]
# CHECK-NOTRAP: ddiv $zero, $5, $1        # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: ori $1, $zero, 32768        # encoding: [0x34,0x01,0x80,0x00]
# CHECK-TRAP: ddiv $zero, $5, $1          # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,$5,-0x8000
# CHECK-NOTRAP: addiu $1, $zero, -32768   # encoding: [0x24,0x01,0x80,0x00]
# CHECK-NOTRAP: ddiv $zero, $5, $1        # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: addiu $1, $zero, -32768     # encoding: [0x24,0x01,0x80,0x00]
# CHECK-TRAP: ddiv $zero, $5, $1          # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,$5,0x10000
# CHECK-NOTRAP: lui $1, 1                 # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-NOTRAP: ddiv $zero, $5, $1        # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-TRAP: ddiv $zero, $5, $1          # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,$5,0x1a5a5
# CHECK-NOTRAP: lui $1, 1                 # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-NOTRAP: ori $1, $1, 42405         # encoding: [0x34,0x21,0xa5,0xa5]
# CHECK-NOTRAP: ddiv $zero, $5, $1        # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-TRAP: ori $1, $1, 42405           # encoding: [0x34,0x21,0xa5,0xa5]
# CHECK-TRAP: ddiv $zero, $5, $1          # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,$5,0xfffffff
# CHECK-NOTRAP: lui $1, 4095              # encoding: [0x3c,0x01,0x0f,0xff]
# CHECK-NOTRAP: ori $1, $1, 65535         # encoding: [0x34,0x21,0xff,0xff]
# CHECK-NOTRAP: ddiv $zero, $5, $1        # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 4095                # encoding: [0x3c,0x01,0x0f,0xff]
# CHECK-TRAP: ori $1, $1, 65535           # encoding: [0x34,0x21,0xff,0xff]
# CHECK-TRAP: ddiv $zero, $5, $1          # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,$5,0x10000000
# CHECK-NOTRAP: lui $1, 4096              # encoding: [0x3c,0x01,0x10,0x00]
# CHECK-NOTRAP: ddiv $zero, $5, $1        # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 4096                # encoding: [0x3c,0x01,0x10,0x00]
# CHECK-TRAP: ddiv $zero, $5, $1          # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,$5,0xfffffffe
# CHECK-NOTRAP: ori $1, $zero, 65535      # encoding: [0x34,0x01,0xff,0xff]
# CHECK-NOTRAP: dsll $1, $1, 16           # encoding: [0x00,0x01,0x0c,0x38]
# CHECK-NOTRAP: ori $1, $1, 65534         # encoding: [0x34,0x21,0xff,0xfe]
# CHECK-NOTRAP: ddiv $zero, $5, $1        # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: ori $1, $zero, 65535        # encoding: [0x34,0x01,0xff,0xff]
# CHECK-TRAP: dsll $1, $1, 16             # encoding: [0x00,0x01,0x0c,0x38]
# CHECK-TRAP: ori $1, $1, 65534           # encoding: [0x34,0x21,0xff,0xfe]
# CHECK-TRAP: ddiv $zero, $5, $1          # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,$5,0xffffffff
# CHECK-NOTRAP: lui $1, 65535             # encoding: [0x3c,0x01,0xff,0xff]
# CHECK-NOTRAP: dsrl32 $1, $1, 0          # encoding: [0x00,0x01,0x08,0x3e]
# CHECK-NOTRAP: ddiv $zero, $5, $1        # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 65535               # encoding: [0x3c,0x01,0xff,0xff]
# CHECK-TRAP: dsrl32 $1, $1, 0            # encoding: [0x00,0x01,0x08,0x3e]
# CHECK-TRAP: ddiv $zero, $5, $1          # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv $4,$5,0xfffffffff
# CHECK-NOTRAP: addiu $1, $zero, 15       # encoding: [0x24,0x01,0x00,0x0f]
# CHECK-NOTRAP: dsll $1, $1, 16           # encoding: [0x00,0x01,0x0c,0x38]
# CHECK-NOTRAP: ori $1, $1, 65535         # encoding: [0x34,0x21,0xff,0xff]
# CHECK-NOTRAP: dsll $1, $1, 16           # encoding: [0x00,0x01,0x0c,0x38]
# CHECK-NOTRAP: ori $1, $1, 65535         # encoding: [0x34,0x21,0xff,0xff]
# CHECK-NOTRAP: ddiv $zero, $5, $1        # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: addiu $1, $zero, 15         # encoding: [0x24,0x01,0x00,0x0f]
# CHECK-TRAP: dsll $1, $1, 16             # encoding: [0x00,0x01,0x0c,0x38]
# CHECK-TRAP: ori $1, $1, 65535           # encoding: [0x34,0x21,0xff,0xff]
# CHECK-TRAP: dsll $1, $1, 16             # encoding: [0x00,0x01,0x0c,0x38]
# CHECK-TRAP: ori $1, $1, 65535           # encoding: [0x34,0x21,0xff,0xff]
# CHECK-TRAP: ddiv $zero, $5, $1          # encoding: [0x00,0xa1,0x00,0x1e]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]
