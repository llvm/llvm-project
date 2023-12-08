# RUN: llvm-mc  %s -triple mips-unknown-linux -show-encoding -mcpu=mips64r2 | FileCheck %s
# RUN: llvm-mc  %s -triple mips-unknown-linux -show-encoding -mcpu=mips64r3 | FileCheck %s
# RUN: llvm-mc  %s -triple mips-unknown-linux -show-encoding -mcpu=mips64r5 | FileCheck %s

# RUN: llvm-mc  %s -triple mips-unknown-linux -show-encoding -mattr=use-tcc-in-div -mcpu=mips64 | FileCheck %s --check-prefix=CHECK-TRAP
# RUN: llvm-mc  %s -triple mips-unknown-linux -show-encoding -mattr=use-tcc-in-div -mcpu=mips64r2 | FileCheck %s --check-prefix=CHECK-TRAP
# RUN: llvm-mc  %s -triple mips-unknown-linux -show-encoding -mattr=use-tcc-in-div -mcpu=mips64r3 | FileCheck %s --check-prefix=CHECK-TRAP
# RUN: llvm-mc  %s -triple mips-unknown-linux -show-encoding -mattr=use-tcc-in-div -mcpu=mips64r5 | FileCheck %s --check-prefix=CHECK-TRAP

.text
text_label:

  mul  $4, $5
# CHECK:        mul     $4, $4, $5              # encoding: [0x70,0x85,0x20,0x02]
# CHECK-TRAP:   mul     $4, $4, $5              # encoding: [0x70,0x85,0x20,0x02]
  mul   $4, $5, $6
# CHECK:        mul     $4, $5, $6              # encoding: [0x70,0xa6,0x20,0x02]
# CHECK-TRAP:   mul     $4, $5, $6              # encoding: [0x70,0xa6,0x20,0x02]
  mul  $4, $5, 0
# CHECK:        addiu   $1, $zero, 0            # encoding: [0x24,0x01,0x00,0x00]
# CHECK:        mult    $5, $1                  # encoding: [0x00,0xa1,0x00,0x18]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   addiu   $1, $zero, 0            # encoding: [0x24,0x01,0x00,0x00]
# CHECK-TRAP:   mult    $5, $1                  # encoding: [0x00,0xa1,0x00,0x18]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
  mul   $4, $5, 1
# CHECK:        addiu   $1, $zero, 1            # encoding: [0x24,0x01,0x00,0x01]
# CHECK:        mult    $5, $1                  # encoding: [0x00,0xa1,0x00,0x18]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   addiu   $1, $zero, 1            # encoding: [0x24,0x01,0x00,0x01]
# CHECK-TRAP:   mult    $5, $1                  # encoding: [0x00,0xa1,0x00,0x18]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
  mul  $4, $5, 0x8000
# CHECK:        ori     $1, $zero, 32768        # encoding: [0x34,0x01,0x80,0x00]
# CHECK:        mult    $5, $1                  # encoding: [0x00,0xa1,0x00,0x18]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   ori     $1, $zero, 32768        # encoding: [0x34,0x01,0x80,0x00]
# CHECK-TRAP:   mult    $5, $1                  # encoding: [0x00,0xa1,0x00,0x18]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
  mul  $4, $5, -0x8000
# CHECK:        addiu   $1, $zero, -32768       # encoding: [0x24,0x01,0x80,0x00]
# CHECK:        mult    $5, $1                  # encoding: [0x00,0xa1,0x00,0x18]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   addiu   $1, $zero, -32768       # encoding: [0x24,0x01,0x80,0x00]
# CHECK-TRAP:   mult    $5, $1                  # encoding: [0x00,0xa1,0x00,0x18]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
  mul  $4, $5, 0x10000
# CHECK:        lui     $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
# CHECK:        mult    $5, $1                  # encoding: [0x00,0xa1,0x00,0x18]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   lui     $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-TRAP:   mult    $5, $1                  # encoding: [0x00,0xa1,0x00,0x18]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
  mul  $4, $5, 0x1a5a5
# CHECK:        lui     $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
# CHECK:        ori     $1, $1, 42405           # encoding: [0x34,0x21,0xa5,0xa5]
# CHECK:        mult    $5, $1                  # encoding: [0x00,0xa1,0x00,0x18]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   lui     $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-TRAP:   ori     $1, $1, 42405           # encoding: [0x34,0x21,0xa5,0xa5]
# CHECK-TRAP:   mult    $5, $1                  # encoding: [0x00,0xa1,0x00,0x18]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
  mulo  $4, $5
# CHECK:        mult    $4, $5                  # encoding: [0x00,0x85,0x00,0x18]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK:        sra     $4, $4, 31              # encoding: [0x00,0x04,0x27,0xc3]
# CHECK:        mfhi    $1                      # encoding: [0x00,0x00,0x08,0x10]
# CHECK:        beq     $4, $1, $tmp0           # encoding: [0x10,0x81,A,A]
# CHECK:        nop                             # encoding: [0x00,0x00,0x00,0x00]
# CHECK:        break   6                       # encoding: [0x00,0x06,0x00,0x0d]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   mult    $4, $5                  # encoding: [0x00,0x85,0x00,0x18]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   sra     $4, $4, 31              # encoding: [0x00,0x04,0x27,0xc3]
# CHECK-TRAP:   mfhi    $1                      # encoding: [0x00,0x00,0x08,0x10]
# CHECK-TRAP:   tne     $4, $1, 6               # encoding: [0x00,0x81,0x01,0xb6]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]

  mulo  $4, $5, $6
# CHECK:        mult    $5, $6                  # encoding: [0x00,0xa6,0x00,0x18]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK:        sra     $4, $4, 31              # encoding: [0x00,0x04,0x27,0xc3]
# CHECK:        mfhi    $1                      # encoding: [0x00,0x00,0x08,0x10]
# CHECK:        beq     $4, $1, $tmp1           # encoding: [0x10,0x81,A,A]
# CHECK:        nop                             # encoding: [0x00,0x00,0x00,0x00]
# CHECK:        break   6                       # encoding: [0x00,0x06,0x00,0x0d]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   mult    $5, $6                  # encoding: [0x00,0xa6,0x00,0x18]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   sra     $4, $4, 31              # encoding: [0x00,0x04,0x27,0xc3]
# CHECK-TRAP:   mfhi    $1                      # encoding: [0x00,0x00,0x08,0x10]
# CHECK-TRAP:   tne     $4, $1, 6               # encoding: [0x00,0x81,0x01,0xb6]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
 mulou  $4,$5
# CHECK:        multu   $4, $5                  # encoding: [0x00,0x85,0x00,0x19]
# CHECK:        mfhi    $1                      # encoding: [0x00,0x00,0x08,0x10]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK:        beqz    $1, $tmp2               # encoding: [0x10,0x20,A,A]
# CHECK:        nop                             # encoding: [0x00,0x00,0x00,0x00]
# CHECK:        break   6                       # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-TRAP:   multu   $4, $5                  # encoding: [0x00,0x85,0x00,0x19]
# CHECK-TRAP:   mfhi    $1                      # encoding: [0x00,0x00,0x08,0x10]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   tne     $1, $zero, 6            # encoding: [0x00,0x20,0x01,0xb6]
 mulou $4, $5, $6
# CHECK:        multu   $5, $6                  # encoding: [0x00,0xa6,0x00,0x19]
# CHECK:        mfhi    $1                      # encoding: [0x00,0x00,0x08,0x10]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK:        beqz    $1, $tmp3               # encoding: [0x10,0x20,A,A]
# CHECK:        nop                             # encoding: [0x00,0x00,0x00,0x00]
# CHECK:        break   6                       # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-TRAP:   multu   $5, $6                  # encoding: [0x00,0xa6,0x00,0x19]
# CHECK-TRAP:   mfhi    $1                      # encoding: [0x00,0x00,0x08,0x10]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   tne     $1, $zero, 6            # encoding: [0x00,0x20,0x01,0xb6]

 dmul $4, $5, $6
# CHECK:        dmultu  $5, $6                  # encoding: [0x00,0xa6,0x00,0x1d]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   dmultu  $5, $6                  # encoding: [0x00,0xa6,0x00,0x1d]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
 dmul $4, $5, 1
# CHECK:        addiu   $1, $zero, 1            # encoding: [0x24,0x01,0x00,0x01]
# CHECK:        dmult   $5, $1                  # encoding: [0x00,0xa1,0x00,0x1c]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   addiu   $1, $zero, 1            # encoding: [0x24,0x01,0x00,0x01]
# CHECK-TRAP:   dmult   $5, $1                  # encoding: [0x00,0xa1,0x00,0x1c]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
 dmulo $4, $5, $6
# CHECK:        dmult   $5, $6                  # encoding: [0x00,0xa6,0x00,0x1c]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK:        dsra32  $4, $4, 31              # encoding: [0x00,0x04,0x27,0xff]
# CHECK:        mfhi    $1                      # encoding: [0x00,0x00,0x08,0x10]
# CHECK:        beq     $4, $1, $tmp4           # encoding: [0x10,0x81,A,A]
# CHECK:        nop                             # encoding: [0x00,0x00,0x00,0x00]
# CHECK:        break   6                       # encoding: [0x00,0x06,0x00,0x0d]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   dmult   $5, $6                  # encoding: [0x00,0xa6,0x00,0x1c]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   dsra32  $4, $4, 31              # encoding: [0x00,0x04,0x27,0xff]
# CHECK-TRAP:   mfhi    $1                      # encoding: [0x00,0x00,0x08,0x10]
# CHECK-TRAP:   tne     $4, $1, 6               # encoding: [0x00,0x81,0x01,0xb6]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
 dmulou  $4,$5,$6
# CHECK:        dmultu  $5, $6                  # encoding: [0x00,0xa6,0x00,0x1d]
# CHECK:        mfhi    $1                      # encoding: [0x00,0x00,0x08,0x10]
# CHECK:        mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK:        beqz    $1, $tmp5               # encoding: [0x10,0x20,A,A]
# CHECK:        nop                             # encoding: [0x00,0x00,0x00,0x00]
# CHECK:        break   6                       # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-TRAP:   dmultu  $5, $6                  # encoding: [0x00,0xa6,0x00,0x1d]
# CHECK-TRAP:   mfhi    $1                      # encoding: [0x00,0x00,0x08,0x10]
# CHECK-TRAP:   mflo    $4                      # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP:   tne     $1, $zero, 6            # encoding: [0x00,0x20,0x01,0xb6]
