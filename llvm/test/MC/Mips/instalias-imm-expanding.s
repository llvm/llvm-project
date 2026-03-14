# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -show-encoding -show-inst | FileCheck --check-prefix=MIPS %s
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -show-encoding -show-inst -mattr=+micromips | FileCheck --check-prefix=MICROMIPS %s


  .text
text_label:

  add $4, -0x80000000
# MIPS: lui   $1, 32768               # encoding: [0x00,0x80,0x01,0x3c]
# MIPS: add   $4, $4, $1              # encoding: [0x20,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADD
# MICROMIPS: lui $1, 32768            # encoding: [0xa1,0x41,0x00,0x80]
# MICROMIPS: add $4, $4, $1           # encoding: [0x24,0x00,0x10,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADD_MM
  add $4, -0x8001
# MIPS: lui   $1, 65535               # encoding: [0xff,0xff,0x01,0x3c]
# MIPS: ori   $1, $1, 32767           # encoding: [0xff,0x7f,0x21,0x34]
# MIPS: add   $4, $4, $1              # encoding: [0x20,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADD
# MICROMIPS: lui $1, 65535            # encoding: [0xa1,0x41,0xff,0xff]
# MICROMIPS: ori $1, $1, 32767        # encoding: [0x21,0x50,0xff,0x7f]
# MICROMIPS: add $4, $4, $1           # encoding: [0x24,0x00,0x10,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADD_MM
  add $4, -0x8000
# MIPS: addi  $4, $4, -32768          # encoding: [0x00,0x80,0x84,0x20]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDi
# MICROMIPS: addi $4, $4, -32768      # encoding: [0x84,0x10,0x00,0x80]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDi_MM
  add $4, 0
# MIPS: addi  $4, $4, 0               # encoding: [0x00,0x00,0x84,0x20]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDi
# MICROMIPS: addi $4, $4, 0           # encoding: [0x84,0x10,0x00,0x00]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDi_MM
  add $4, 0xFFFF
# MIPS: ori   $1, $zero, 65535        # encoding: [0xff,0xff,0x01,0x34]
# MIPS: add   $4, $4, $1              # encoding: [0x20,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADD
# MICROMIPS: ori $1, $zero, 65535     # encoding: [0x20,0x50,0xff,0xff]
# MICROMIPS: add $4, $4, $1           # encoding: [0x24,0x00,0x10,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADD_MM
  add $4, 0x10000
# MIPS: lui   $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
# MIPS: add   $4, $4, $1              # encoding: [0x20,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADD
# MICROMIPS: lui $1, 1                # encoding: [0xa1,0x41,0x01,0x00]
# MICROMIPS: add $4, $4, $1           # encoding: [0x24,0x00,0x10,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADD_MM
  add $4, 0xFFFFFFFF
# MIPS: addi  $4, $4, -1              # encoding: [0xff,0xff,0x84,0x20]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADD
# MICROMIPS: addi $4, $4, -1          # encoding: [0x84,0x10,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDi_MM
  add $5, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $1, 255                 # encoding: [0xff,0x00,0x01,0x3c]
# MIPS: ori   $1, $1, 65295           # encoding: [0x0f,0xff,0x21,0x34]
# MIPS: add   $5, $5, $1              # encoding: [0x20,0x28,0xa1,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADD
# MICROMIPS: lui $1, 255              # encoding: [0xa1,0x41,0xff,0x00]
# MICROMIPS: ori $1, $1, 65295        # encoding: [0x21,0x50,0x0f,0xff]
# MICROMIPS: add $5, $5, $1           # encoding: [0x25,0x00,0x10,0x29]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADD_MM
  add $4, $5, -0x80000000
# MIPS: lui   $4, 32768               # encoding: [0x00,0x80,0x04,0x3c]
# MIPS: add   $4, $4, $5              # encoding: [0x20,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADD
# MICROMIPS: lui $4, 32768            # encoding: [0xa4,0x41,0x00,0x80]
# MICROMIPS: add $4, $4, $5           # encoding: [0xa4,0x00,0x10,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADD_MM
  add $4, $5, -0x8001
# MIPS: lui   $4, 65535               # encoding: [0xff,0xff,0x04,0x3c]
# MIPS: ori   $4, $4, 32767           # encoding: [0xff,0x7f,0x84,0x34]
# MIPS: add   $4, $4, $5              # encoding: [0x20,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADD
# MICROMIPS: lui $4, 65535            # encoding: [0xa4,0x41,0xff,0xff]
# MICROMIPS: ori $4, $4, 32767        # encoding: [0x84,0x50,0xff,0x7f]
# MICROMIPS: add $4, $4, $5           # encoding: [0xa4,0x00,0x10,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADD_MM
  add $4, $5, -0x8000
# MIPS: addi  $4, $5, -32768          # encoding: [0x00,0x80,0xa4,0x20]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDi
# MICROMIPS: addi $4, $5, -32768      # encoding: [0x85,0x10,0x00,0x80]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDi_MM
  add $4, $5, 0
# MIPS: addi  $4, $5, 0               # encoding: [0x00,0x00,0xa4,0x20]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDi
# MICROMIPS: addi $4, $5, 0           # encoding: [0x85,0x10,0x00,0x00]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDi_MM
  add $4, $5, 0xFFFF
# MIPS: ori   $4, $zero, 65535        # encoding: [0xff,0xff,0x04,0x34]
# MIPS: add   $4, $4, $5              # encoding: [0x20,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADD
# MICROMIPS: ori $4, $zero, 65535     # encoding: [0x80,0x50,0xff,0xff]
# MICROMIPS: add $4, $4, $5           # encoding: [0xa4,0x00,0x10,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADD_MM
  add $4, $5, 0x10000
# MIPS: lui   $4, 1                   # encoding: [0x01,0x00,0x04,0x3c]
# MIPS: add   $4, $4, $5              # encoding: [0x20,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADD
# MICROMIPS: lui $4, 1                # encoding: [0xa4,0x41,0x01,0x00]
# MICROMIPS: add $4, $4, $5           # encoding: [0xa4,0x00,0x10,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADD_MM
  add $4, $5, 0xFFFFFFFF
# MIPS: addi  $4, $5, -1              # encoding: [0xff,0xff,0xa4,0x20]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDi
# MICROMIPS: addi $4, $5, -1          # encoding: [0x85,0x10,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDi_MM
  add $4, $5, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $4, 255                 # encoding: [0xff,0x00,0x04,0x3c]
# MIPS: ori   $4, $4, 65295           # encoding: [0x0f,0xff,0x84,0x34]
# MIPS: add   $4, $4, $5              # encoding: [0x20,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADD
# MICROMIPS: lui $4, 255              # encoding: [0xa4,0x41,0xff,0x00]
# MICROMIPS: ori $4, $4, 65295        # encoding: [0x84,0x50,0x0f,0xff]
# MICROMIPS: add $4, $4, $5           # encoding: [0xa4,0x00,0x10,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADD_MM

  addu $4, -0x80000000
# MIPS: lui   $1, 32768               # encoding: [0x00,0x80,0x01,0x3c]
# MIPS: addu  $4, $4, $1              # encoding: [0x21,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDu
# MICROMIPS: lui $1, 32768            # encoding: [0xa1,0x41,0x00,0x80]
# MICROMIPS: addu $4, $4, $1          # encoding: [0x24,0x00,0x50,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDu_MM
  addu $4, -0x8001
# MIPS: lui   $1, 65535               # encoding: [0xff,0xff,0x01,0x3c]
# MIPS: ori   $1, $1, 32767           # encoding: [0xff,0x7f,0x21,0x34]
# MIPS: addu  $4, $4, $1              # encoding: [0x21,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDu
# MICROMIPS: lui $1, 65535            # encoding: [0xa1,0x41,0xff,0xff]
# MICROMIPS: ori $1, $1, 32767        # encoding: [0x21,0x50,0xff,0x7f]
# MICROMIPS: addu $4, $4, $1          # encoding: [0x24,0x00,0x50,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDu_MM
  addu $4, -0x8000
# MIPS: addiu $4, $4, -32768          # encoding: [0x00,0x80,0x84,0x24]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDiu
# MICROMIPS: addiu $4, $4, -32768     # encoding: [0x84,0x30,0x00,0x80]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDiu_MM
  addu $4, 0
# MIPS: addiu $4, $4, 0               # encoding: [0x00,0x00,0x84,0x24]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDiu
# MICROMIPS: addiu $4, $4, 0          # encoding: [0x84,0x30,0x00,0x00]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDiu_MM
  addu $4, 0xFFFF
# MIPS: ori   $1, $zero, 65535        # encoding: [0xff,0xff,0x01,0x34]
# MIPS: addu  $4, $4, $1              # encoding: [0x21,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDu
# MICROMIPS: ori $1, $zero, 65535     # encoding: [0x20,0x50,0xff,0xff]
# MICROMIPS: addu $4, $4, $1          # encoding: [0x24,0x00,0x50,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDu_MM
  addu $4, 0x10000
# MIPS: lui   $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
# MIPS: addu  $4, $4, $1              # encoding: [0x21,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDu
# MICROMIPS: lui $1, 1                # encoding: [0xa1,0x41,0x01,0x00]
# MICROMIPS: addu $4, $4, $1          # encoding: [0x24,0x00,0x50,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDu_MM
  addu $4, 0xFFFFFFFF
# MIPS: addiu $4, $4, -1              # encoding: [0xff,0xff,0x84,0x24]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDiu
# MICROMIPS: addiu $4, $4, -1         # encoding: [0x84,0x30,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDiu_MM
  addu $5, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $1, 255                 # encoding: [0xff,0x00,0x01,0x3c]
# MIPS: ori   $1, $1, 65295           # encoding: [0x0f,0xff,0x21,0x34]
# MIPS: addu  $5, $5, $1              # encoding: [0x21,0x28,0xa1,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDu
# MICROMIPS: lui $1, 255              # encoding: [0xa1,0x41,0xff,0x00]
# MICROMIPS: ori $1, $1, 65295        # encoding: [0x21,0x50,0x0f,0xff]
# MICROMIPS: addu $5, $5, $1          # encoding: [0x25,0x00,0x50,0x29]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDu_MM

  addu $4, $5, -0x80000000
# MIPS: lui   $4, 32768               # encoding: [0x00,0x80,0x04,0x3c]
# MIPS: addu  $4, $4, $5              # encoding: [0x21,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDu
# MICROMIPS: lui $4, 32768            # encoding: [0xa4,0x41,0x00,0x80]
# MICROMIPS: addu $4, $4, $5          # encoding: [0xa4,0x00,0x50,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDu_MM
  addu $4, $5, -0x8001
# MIPS: lui   $4, 65535               # encoding: [0xff,0xff,0x04,0x3c]
# MIPS: ori   $4, $4, 32767           # encoding: [0xff,0x7f,0x84,0x34]
# MIPS: addu  $4, $4, $5              # encoding: [0x21,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDu
# MICROMIPS: lui $4, 65535            # encoding: [0xa4,0x41,0xff,0xff]
# MICROMIPS: ori $4, $4, 32767        # encoding: [0x84,0x50,0xff,0x7f]
# MICROMIPS: addu $4, $4, $5          # encoding: [0xa4,0x00,0x50,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDu_MM
  addu $4, $5, -0x8000
# MIPS: addiu $4, $5, -32768          # encoding: [0x00,0x80,0xa4,0x24]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDiu
# MICROMIPS: addiu $4, $5, -32768     # encoding: [0x85,0x30,0x00,0x80]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDiu_MM
  addu $4, $5, 0
# MIPS: addiu $4, $5, 0               # encoding: [0x00,0x00,0xa4,0x24]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDiu
# MICROMIPS: addiu $4, $5, 0          # encoding: [0x85,0x30,0x00,0x00]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDiu_MM
  addu $4, $5, 0xFFFF
# MIPS: ori   $4, $zero, 65535        # encoding: [0xff,0xff,0x04,0x34]
# MIPS: addu  $4, $4, $5              # encoding: [0x21,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDu
# MICROMIPS: ori $4, $zero, 65535     # encoding: [0x80,0x50,0xff,0xff]
# MICROMIPS: addu $4, $4, $5          # encoding: [0xa4,0x00,0x50,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDu_MM
  addu $4, $5, 0x10000
# MIPS: lui   $4, 1                   # encoding: [0x01,0x00,0x04,0x3c]
# MIPS: addu  $4, $4, $5              # encoding: [0x21,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDu
# MICROMIPS: lui $4, 1                # encoding: [0xa4,0x41,0x01,0x00]
# MICROMIPS: addu $4, $4, $5          # encoding: [0xa4,0x00,0x50,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDu_MM
  addu $4, $5, 0xFFFFFFFF
# MIPS: addiu  $4, $5, -1              # encoding: [0xff,0xff,0xa4,0x24]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDiu
# MICROMIPS: addiu $4, $5, -1         # encoding: [0x85,0x30,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDiu_MM
  addu $4, $5, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $4, 255                 # encoding: [0xff,0x00,0x04,0x3c]
# MIPS: ori   $4, $4, 65295           # encoding: [0x0f,0xff,0x84,0x34]
# MIPS: addu  $4, $4, $5              # encoding: [0x21,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ADDu
# MICROMIPS: lui $4, 255              # encoding: [0xa4,0x41,0xff,0x00]
# MICROMIPS: ori $4, $4, 65295        # encoding: [0x84,0x50,0x0f,0xff]
# MICROMIPS: addu $4, $4, $5          # encoding: [0xa4,0x00,0x50,0x21]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDu_MM

  and $4, -0x80000000
# MIPS: lui   $1, 32768               # encoding: [0x00,0x80,0x01,0x3c]
# MIPS: and   $4, $4, $1              # encoding: [0x24,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} AND
# MICROMIPS: lui $1, 32768            # encoding: [0xa1,0x41,0x00,0x80]
# MICROMIPS: and $4, $4, $1           # encoding: [0x24,0x00,0x50,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} AND_MM
  and $4, -0x8001
# MIPS: lui   $1, 65535               # encoding: [0xff,0xff,0x01,0x3c]
# MIPS: ori   $1, $1, 32767           # encoding: [0xff,0x7f,0x21,0x34]
# MIPS: and   $4, $4, $1              # encoding: [0x24,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} AND
# MICROMIPS: lui $1, 65535            # encoding: [0xa1,0x41,0xff,0xff]
# MICROMIPS: ori $1, $1, 32767        # encoding: [0x21,0x50,0xff,0x7f]
# MICROMIPS: and $4, $4, $1           # encoding: [0x24,0x00,0x50,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} AND_MM
  and $4, -0x8000
# MIPS: addiu $1, $zero, -32768       # encoding: [0x00,0x80,0x01,0x24]
# MIPS: and   $4, $4, $1              # encoding: [0x24,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} AND
# MICROMIPS: addiu $1, $zero, -32768  # encoding: [0x20,0x30,0x00,0x80]
# MICROMIPS: and $4, $4, $1           # encoding: [0x24,0x00,0x50,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} AND_MM
  and $4, 0
# MIPS: andi  $4, $4, 0               # encoding: [0x00,0x00,0x84,0x30]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ANDi
# MICROMIPS: andi $4, $4, 0           # encoding: [0x84,0xd0,0x00,0x00]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ANDi_MM
  and $4, 0xFFFF
# MIPS: andi  $4, $4, 65535           # encoding: [0xff,0xff,0x84,0x30]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ANDi
# MICROMIPS: andi $4, $4, 65535       # encoding: [0x84,0xd0,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ANDi_MM
  and $4, 0x10000
# MIPS: lui   $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
# MIPS: and   $4, $4, $1              # encoding: [0x24,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} AND
# MICROMIPS: lui $1, 1                # encoding: [0xa1,0x41,0x01,0x00]
# MICROMIPS: and $4, $4, $1           # encoding: [0x24,0x00,0x50,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} AND_MM
  and $4, 0xFFFFFFFF
# MIPS: addiu $1, $zero, -1           # encoding: [0xff,0xff,0x01,0x24]
# MIPS: and   $4, $4, $1              # encoding: [0x24,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} AND
# MICROMIPS: addiu $1, $zero, -1      # encoding: [0x20,0x30,0xff,0xff]
# MICROMIPS: and $4, $4, $1           # encoding: [0x24,0x00,0x50,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} AND_MM
  and $5, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $1, 255                 # encoding: [0xff,0x00,0x01,0x3c]
# MIPS: ori   $1, $1, 65295           # encoding: [0x0f,0xff,0x21,0x34]
# MIPS: and   $5, $5, $1              # encoding: [0x24,0x28,0xa1,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} AND
# MICROMIPS: lui $1, 255              # encoding: [0xa1,0x41,0xff,0x00]
# MICROMIPS: ori $1, $1, 65295        # encoding: [0x21,0x50,0x0f,0xff]
# MICROMIPS: and $5, $5, $1           # encoding: [0x25,0x00,0x50,0x2a]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} AND_MM

  and $4, $5, -0x80000000
# MIPS: lui   $4, 32768               # encoding: [0x00,0x80,0x04,0x3c]
# MIPS: and   $4, $4, $5              # encoding: [0x24,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} AND
# MICROMIPS: lui $4, 32768            # encoding: [0xa4,0x41,0x00,0x80]
# MICROMIPS: and $4, $4, $5           # encoding: [0xa4,0x00,0x50,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} AND_MM
  and $4, $5, -0x8001
# MIPS: lui   $4, 65535               # encoding: [0xff,0xff,0x04,0x3c]
# MIPS: ori   $4, $4, 32767           # encoding: [0xff,0x7f,0x84,0x34]
# MIPS: and   $4, $4, $5              # encoding: [0x24,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} AND
# MICROMIPS: lui $4, 65535            # encoding: [0xa4,0x41,0xff,0xff]
# MICROMIPS: ori $4, $4, 32767        # encoding: [0x84,0x50,0xff,0x7f]
# MICROMIPS: and $4, $4, $5           # encoding: [0xa4,0x00,0x50,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} AND_MM
  and $4, $5, -0x8000
# MIPS: addiu $4, $zero, -32768       # encoding: [0x00,0x80,0x04,0x24]
# MIPS: and   $4, $4, $5              # encoding: [0x24,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} AND
# MICROMIPS: addiu $4, $zero, -32768  # encoding: [0x80,0x30,0x00,0x80]
# MICROMIPS: and $4, $4, $5           # encoding: [0xa4,0x00,0x50,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} AND_MM
  and $4, $5, 0
# MIPS: andi  $4, $5, 0               # encoding: [0x00,0x00,0xa4,0x30]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ANDi
# MICROMIPS: andi $4, $5, 0           # encoding: [0x85,0xd0,0x00,0x00]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ANDi_MM
  and $4, $5, 0xFFFF
# MIPS: andi  $4, $5, 65535           # encoding: [0xff,0xff,0xa4,0x30]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ANDi
# MICROMIPS: andi $4, $5, 65535       # encoding: [0x85,0xd0,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ANDi_MM
  and $4, $5, 0x10000
# MIPS: lui   $4, 1                   # encoding: [0x01,0x00,0x04,0x3c]
# MIPS: and   $4, $4, $5              # encoding: [0x24,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} AND
# MICROMIPS: lui $4, 1                # encoding: [0xa4,0x41,0x01,0x00]
# MICROMIPS: and $4, $4, $5           # encoding: [0xa4,0x00,0x50,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} AND_MM
  and $4, $5, 0xFFFFFFFF
# MIPS: addiu $4, $zero, -1           # encoding: [0xff,0xff,0x04,0x24]
# MIPS: and   $4, $4, $5              # encoding: [0x24,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} AND
# MICROMIPS: addiu $4, $zero, -1      # encoding: [0x80,0x30,0xff,0xff]
# MICROMIPS: and $4, $4, $5           # encoding: [0xa4,0x00,0x50,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} AND_MM
  and $4, $5, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $4, 255                 # encoding: [0xff,0x00,0x04,0x3c]
# MIPS: ori   $4, $4, 65295           # encoding: [0x0f,0xff,0x84,0x34]
# MIPS: and   $4, $4, $5              # encoding: [0x24,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} AND
# MICROMIPS: lui $4, 255              # encoding: [0xa4,0x41,0xff,0x00]
# MICROMIPS: ori $4, $4, 65295        # encoding: [0x84,0x50,0x0f,0xff]
# MICROMIPS: and $4, $4, $5           # encoding: [0xa4,0x00,0x50,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} AND_MM

  nor $4, $5, 0
# MIPS: addiu $4, $zero, 0            # encoding: [0x00,0x00,0x04,0x24]
# MIPS: nor   $4, $4, $5              # encoding: [0x27,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: addiu $4, $zero, 0       # encoding: [0x80,0x30,0x00,0x00]
# MICROMIPS: nor $4, $4, $5           # encoding: [0xa4,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR
  nor $4, $5, 1
# MIPS: addiu $4, $zero, 1            # encoding: [0x01,0x00,0x04,0x24]
# MIPS: nor   $4, $4, $5              # encoding: [0x27,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: addiu $4, $zero, 1       # encoding: [0x80,0x30,0x01,0x00]
# MICROMIPS: nor $4, $4, $5           # encoding: [0xa4,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR
  nor $4, $5, 0x8000
# MIPS: ori   $4, $zero, 32768        # encoding: [0x00,0x80,0x04,0x34]
# MIPS: nor   $4, $4, $5              # encoding: [0x27,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: ori $4, $zero, 32768     # encoding: [0x80,0x50,0x00,0x80]
# MICROMIPS: nor $4, $4, $5           # encoding: [0xa4,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR
  nor $4, $5, -0x8000
# MIPS: addiu $4, $zero, -32768       # encoding: [0x00,0x80,0x04,0x24]
# MIPS: nor   $4, $4, $5              # encoding: [0x27,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: addiu $4, $zero, -32768  # encoding: [0x80,0x30,0x00,0x80]
# MICROMIPS: nor $4, $4, $5           # encoding: [0xa4,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR
  nor $4, $5, 0x10000
# MIPS: lui   $4, 1                   # encoding: [0x01,0x00,0x04,0x3c]
# MIPS: nor   $4, $4, $5              # encoding: [0x27,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: lui $4, 1                # encoding: [0xa4,0x41,0x01,0x00]
# MICROMIPS: nor $4, $4, $5           # encoding: [0xa4,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR
  nor $4, $5, 0x1a5a5
# MIPS: lui   $4, 1                   # encoding: [0x01,0x00,0x04,0x3c]
# MIPS: ori   $4, $4, 42405           # encoding: [0xa5,0xa5,0x84,0x34]
# MIPS: nor   $4, $4, $5              # encoding: [0x27,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: lui $4, 1                # encoding: [0xa4,0x41,0x01,0x00]
# MICROMIPS: ori $4, $4, 42405        # encoding: [0x84,0x50,0xa5,0xa5]
# MICROMIPS: nor $4, $4, $5           # encoding: [0xa4,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR
  nor $4, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $1, 255                 # encoding: [0xff,0x00,0x01,0x3c]
# MIPS: ori   $1, $1, 65295           # encoding: [0x0f,0xff,0x21,0x34]
# MIPS: nor   $4, $4, $1              # encoding: [0x27,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: lui $1, 255              # encoding: [0xa1,0x41,0xff,0x00]
# MICROMIPS: ori $1, $1, 65295        # encoding: [0x21,0x50,0x0f,0xff]
# MICROMIPS: nor $4, $4, $1           # encoding: [0x24,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR

  nor $4, 0
# MIPS: addiu $1, $zero, 0            # encoding: [0x00,0x00,0x01,0x24]
# MIPS: nor   $4, $4, $1              # encoding: [0x27,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: addiu $1, $zero, 0       # encoding: [0x20,0x30,0x00,0x00]
# MICROMIPS: nor $4, $4, $1           # encoding: [0x24,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR
  nor $4, 1
# MIPS: addiu $1, $zero, 1            # encoding: [0x01,0x00,0x01,0x24]
# MIPS: nor   $4, $4, $1              # encoding: [0x27,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: addiu $1, $zero, 1       # encoding: [0x20,0x30,0x01,0x00]
# MICROMIPS: nor $4, $4, $1           # encoding: [0x24,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR
  nor $4, 0x8000
# MIPS: ori   $1, $zero, 32768        # encoding: [0x00,0x80,0x01,0x34]
# MIPS: nor   $4, $4, $1              # encoding: [0x27,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: ori $1, $zero, 32768     # encoding: [0x20,0x50,0x00,0x80]
# MICROMIPS: nor $4, $4, $1           # encoding: [0x24,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR
  nor $4, -0x8000
# MIPS: addiu $1, $zero, -32768       # encoding: [0x00,0x80,0x01,0x24]
# MIPS: nor   $4, $4, $1              # encoding: [0x27,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: addiu $1, $zero, -32768  # encoding: [0x20,0x30,0x00,0x80]
# MICROMIPS: nor $4, $4, $1           # encoding: [0x24,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR
  nor $4, 0x10000
# MIPS: lui   $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
# MIPS: nor   $4, $4, $1              # encoding: [0x27,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: lui $1, 1                # encoding: [0xa1,0x41,0x01,0x00]
# MICROMIPS: nor $4, $4, $1           # encoding: [0x24,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR
  nor $4, 0x1a5a5
# MIPS: lui   $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
# MIPS: ori   $1, $1, 42405           # encoding: [0xa5,0xa5,0x21,0x34]
# MIPS: nor   $4, $4, $1              # encoding: [0x27,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: lui $1, 1                # encoding: [0xa1,0x41,0x01,0x00]
# MICROMIPS: ori $1, $1, 42405        # encoding: [0x21,0x50,0xa5,0xa5]
# MICROMIPS: nor $4, $4, $1           # encoding: [0x24,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR
  nor $4, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $1, 255                 # encoding: [0xff,0x00,0x01,0x3c]
# MIPS: ori   $1, $1, 65295           # encoding: [0x0f,0xff,0x21,0x34]
# MIPS: nor   $4, $4, $1              # encoding: [0x27,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} NOR
# MICROMIPS: lui $1, 255              # encoding: [0xa1,0x41,0xff,0x00]
# MICROMIPS: ori $1, $1, 65295        # encoding: [0x21,0x50,0x0f,0xff]
# MICROMIPS: nor $4, $4, $1           # encoding: [0x24,0x00,0xd0,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} NOR

  or $4, -0x80000000
# MIPS: lui   $1, 32768               # encoding: [0x00,0x80,0x01,0x3c]
# MIPS: or    $4, $4, $1              # encoding: [0x25,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} OR
# MICROMIPS: lui $1, 32768            # encoding: [0xa1,0x41,0x00,0x80]
# MICROMIPS: or $4, $4, $1            # encoding: [0x24,0x00,0x90,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} OR_MM
  or $4, -0x8001
# MIPS: lui   $1, 65535               # encoding: [0xff,0xff,0x01,0x3c]
# MIPS: ori   $1, $1, 32767           # encoding: [0xff,0x7f,0x21,0x34]
# MIPS: or    $4, $4, $1              # encoding: [0x25,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} OR
# MICROMIPS: lui $1, 65535            # encoding: [0xa1,0x41,0xff,0xff]
# MICROMIPS: ori $1, $1, 32767        # encoding: [0x21,0x50,0xff,0x7f]
# MICROMIPS: or $4, $4, $1            # encoding: [0x24,0x00,0x90,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} OR_MM
  or $4, -0x8000
# MIPS: addiu $1, $zero, -32768       # encoding: [0x00,0x80,0x01,0x24]
# MIPS: or    $4, $4, $1              # encoding: [0x25,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} OR
# MICROMIPS: addiu $1, $zero, -32768  # encoding: [0x20,0x30,0x00,0x80]
# MICROMIPS: or $4, $4, $1            # encoding: [0x24,0x00,0x90,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} OR_MM
  or $4, 0
# MIPS: ori   $4, $4, 0               # encoding: [0x00,0x00,0x84,0x34]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ORi
# MICROMIPS: ori $4, $4, 0            # encoding: [0x84,0x50,0x00,0x00]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ORi_MM
  or $4, 0xFFFF
# MIPS: ori   $4, $4, 65535           # encoding: [0xff,0xff,0x84,0x34]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ORi
# MICROMIPS: ori $4, $4, 65535        # encoding: [0x84,0x50,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ORi_MM
  or $4, 0x10000
# MIPS: lui   $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
# MIPS: or    $4, $4, $1              # encoding: [0x25,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} OR
# MICROMIPS: lui $1, 1                # encoding: [0xa1,0x41,0x01,0x00]
# MICROMIPS: or $4, $4, $1            # encoding: [0x24,0x00,0x90,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} OR_MM
  or $4, 0xFFFFFFFF
# MIPS: addiu $1, $zero, -1           # encoding: [0xff,0xff,0x01,0x24]
# MIPS: or    $4, $4, $1              # encoding: [0x25,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} OR
# MICROMIPS: addiu $1, $zero, -1      # encoding: [0x20,0x30,0xff,0xff]
# MICROMIPS: or $4, $4, $1            # encoding: [0x24,0x00,0x90,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} OR_MM
  or $5, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $1, 255                 # encoding: [0xff,0x00,0x01,0x3c]
# MIPS: ori   $1, $1, 65295           # encoding: [0x0f,0xff,0x21,0x34]
# MIPS: or    $5, $5, $1              # encoding: [0x25,0x28,0xa1,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} OR
# MICROMIPS: lui $1, 255              # encoding: [0xa1,0x41,0xff,0x00]
# MICROMIPS: ori $1, $1, 65295        # encoding: [0x21,0x50,0x0f,0xff]
# MICROMIPS: or $5, $5, $1            # encoding: [0x25,0x00,0x90,0x2a]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} OR_MM

  or $4, $5, -0x80000000
# MIPS: lui   $4, 32768               # encoding: [0x00,0x80,0x04,0x3c]
# MIPS: or    $4, $4, $5              # encoding: [0x25,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} OR
# MICROMIPS: lui $4, 32768            # encoding: [0xa4,0x41,0x00,0x80]
# MICROMIPS: or $4, $4, $5            # encoding: [0xa4,0x00,0x90,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} OR_MM
  or $4, $5, -0x8001
# MIPS: lui   $4, 65535               # encoding: [0xff,0xff,0x04,0x3c]
# MIPS: ori   $4, $4, 32767           # encoding: [0xff,0x7f,0x84,0x34]
# MIPS: or    $4, $4, $5              # encoding: [0x25,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} OR
# MICROMIPS: lui $4, 65535            # encoding: [0xa4,0x41,0xff,0xff]
# MICROMIPS: ori $4, $4, 32767        # encoding: [0x84,0x50,0xff,0x7f]
# MICROMIPS: or $4, $4, $5            # encoding: [0xa4,0x00,0x90,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} OR_MM
  or $4, $5, -0x8000
# MIPS: addiu $4, $zero, -32768       # encoding: [0x00,0x80,0x04,0x24]
# MIPS: or    $4, $4, $5              # encoding: [0x25,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} OR
# MICROMIPS: addiu $4, $zero, -32768  # encoding: [0x80,0x30,0x00,0x80]
# MICROMIPS: or $4, $4, $5            # encoding: [0xa4,0x00,0x90,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} OR_MM
  or $4, $5, 0
# MIPS: ori   $4, $5, 0               # encoding: [0x00,0x00,0xa4,0x34]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ORi
# MICROMIPS: ori $4, $5, 0            # encoding: [0x85,0x50,0x00,0x00]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ORi_MM
  or $4, $5, 0xFFFF
# MIPS: ori   $4, $5, 65535           # encoding: [0xff,0xff,0xa4,0x34]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} ORi
# MICROMIPS: ori $4, $5, 65535        # encoding: [0x85,0x50,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ORi_MM
  or $4, $5, 0x10000
# MIPS: lui   $4, 1                   # encoding: [0x01,0x00,0x04,0x3c]
# MIPS: or    $4, $4, $5              # encoding: [0x25,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} OR
# MICROMIPS: lui $4, 1                # encoding: [0xa4,0x41,0x01,0x00]
# MICROMIPS: or $4, $4, $5            # encoding: [0xa4,0x00,0x90,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} OR_MM
  or $4, $5, 0xFFFFFFFF
# MIPS: addiu  $4, $zero, -1          # encoding: [0xff,0xff,0x04,0x24]
# MIPS: or    $4, $4, $5              # encoding: [0x25,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} OR
# MICROMIPS: addiu $4, $zero, -1      # encoding: [0x80,0x30,0xff,0xff]
# MICROMIPS: or $4, $4, $5            # encoding: [0xa4,0x00,0x90,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} OR_MM
  or $4, $5, ~(0xF0000000|0x0F000000|0x000000F0)
# MIPS: lui   $4, 255                 # encoding: [0xff,0x00,0x04,0x3c]
# MIPS: ori   $4, $4, 65295           # encoding: [0x0f,0xff,0x84,0x34]
# MIPS: or    $4, $4, $5              # encoding: [0x25,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} OR
# MICROMIPS: lui $4, 255              # encoding: [0xa4,0x41,0xff,0x00]
# MICROMIPS: ori $4, $4, 65295        # encoding: [0x84,0x50,0x0f,0xff]
# MICROMIPS: or $4, $4, $5            # encoding: [0xa4,0x00,0x90,0x22]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} OR_MM

  slt $4, $5, -0x80000000
# MIPS: lui   $4, 32768               # encoding: [0x00,0x80,0x04,0x3c]
# MIPS: slt   $4, $4, $5              # encoding: [0x2a,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLT
# MICROMIPS: lui $4, 32768            # encoding: [0xa4,0x41,0x00,0x80]
# MICROMIPS: slt $4, $4, $5           # encoding: [0xa4,0x00,0x50,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLT_MM
  slt $4, $5, -0x8001
# MIPS: lui   $4, 65535               # encoding: [0xff,0xff,0x04,0x3c]
# MIPS: ori   $4, $4, 32767           # encoding: [0xff,0x7f,0x84,0x34]
# MIPS: slt   $4, $4, $5              # encoding: [0x2a,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLT
# MICROMIPS: lui $4, 65535            # encoding: [0xa4,0x41,0xff,0xff]
# MICROMIPS: ori $4, $4, 32767        # encoding: [0x84,0x50,0xff,0x7f]
# MICROMIPS: slt $4, $4, $5           # encoding: [0xa4,0x00,0x50,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLT_MM
  slt $4, $5, -0x8000
# MIPS: slti  $4, $5, -32768          # encoding: [0x00,0x80,0xa4,0x28]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLTi
# MICROMIPS: slti $4, $5, -32768      # encoding: [0x85,0x90,0x00,0x80]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLTi_MM
  slt $4, $5, 0
# MIPS: slti  $4, $5, 0               # encoding: [0x00,0x00,0xa4,0x28]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLTi
# MICROMIPS: slti $4, $5, 0           # encoding: [0x85,0x90,0x00,0x00]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLTi_MM
  slt $4, $5, 0xFFFF
# MIPS: ori   $4, $zero, 65535        # encoding: [0xff,0xff,0x04,0x34]
# MIPS: slt   $4, $4, $5              # encoding: [0x2a,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLT
# MICROMIPS: ori $4, $zero, 65535     # encoding: [0x80,0x50,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ORi
# MICROMIPS: slt $4, $4, $5           # encoding: [0xa4,0x00,0x50,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLT_MM
  slt $4, $5, 0x10000
# MIPS: lui   $4, 1                   # encoding: [0x01,0x00,0x04,0x3c]
# MIPS: slt   $4, $4, $5              # encoding: [0x2a,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLT
# MICROMIPS: lui $4, 1                # encoding: [0xa4,0x41,0x01,0x00]
# MICROMIPS: slt $4, $4, $5           # encoding: [0xa4,0x00,0x50,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLT_MM
  slt $4, $5, 0xFFFFFFFF
# MIPS: slti   $4, $5, -1             # encoding: [0xff,0xff,0xa4,0x28]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLT
# MICROMIPS: slti $4, $5, -1          # encoding: [0x85,0x90,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLTi_MM
  slt $4, $5, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $4, 255                 # encoding: [0xff,0x00,0x04,0x3c]
# MIPS: ori   $4, $4, 65295           # encoding: [0x0f,0xff,0x84,0x34]
# MIPS: slt   $4, $4, $5              # encoding: [0x2a,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLT
# MICROMIPS: lui $4, 255              # encoding: [0xa4,0x41,0xff,0x00]
# MICROMIPS: ori $4, $4, 65295        # encoding: [0x84,0x50,0x0f,0xff]
# MICROMIPS: slt $4, $4, $5           # encoding: [0xa4,0x00,0x50,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLT_MM

  sltu $4, $5, -0x80000000
# MIPS: lui   $4, 32768               # encoding: [0x00,0x80,0x04,0x3c]
# MIPS: sltu  $4, $4, $5              # encoding: [0x2b,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLTu
# MICROMIPS: lui $4, 32768            # encoding: [0xa4,0x41,0x00,0x80]
# MICROMIPS: sltu $4, $4, $5          # encoding: [0xa4,0x00,0x90,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLTu_MM
  sltu $4, $5, -0x8001
# MIPS: lui   $4, 65535               # encoding: [0xff,0xff,0x04,0x3c]
# MIPS: ori   $4, $4, 32767           # encoding: [0xff,0x7f,0x84,0x34]
# MIPS: sltu  $4, $4, $5              # encoding: [0x2b,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLTu
# MICROMIPS: lui $4, 65535            # encoding: [0xa4,0x41,0xff,0xff]
# MICROMIPS: ori $4, $4, 32767        # encoding: [0x84,0x50,0xff,0x7f]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ORi
# MICROMIPS: sltu $4, $4, $5          # encoding: [0xa4,0x00,0x90,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLTu_MM
  sltu $4, $5, -0x8000
# MIPS: sltiu  $4, $5, -32768         # encoding: [0x00,0x80,0xa4,0x2c]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLTiu
# MICROMIPS: sltiu $4, $5, -32768     # encoding: [0x85,0xb0,0x00,0x80]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLTiu_MM
  sltu $4, $5, 0
# MIPS: sltiu  $4, $5, 0              # encoding: [0x00,0x00,0xa4,0x2c]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLTiu
# MICROMIPS: sltiu $4, $5, 0          # encoding: [0x85,0xb0,0x00,0x00]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLTiu_MM
  sltu $4, $5, 0xFFFF
# MIPS: ori   $4, $zero, 65535        # encoding: [0xff,0xff,0x04,0x34]
# MIPS: sltu  $4, $4, $5              # encoding: [0x2b,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLTu
# MICROMIPS: ori $4, $zero, 65535     # encoding: [0x80,0x50,0xff,0xff]
# MICROMIPS: sltu $4, $4, $5          # encoding: [0xa4,0x00,0x90,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLTu_MM
  sltu $4, $5, 0x10000
# MIPS: lui   $4, 1                   # encoding: [0x01,0x00,0x04,0x3c]
# MIPS: sltu  $4, $4, $5              # encoding: [0x2b,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLTu
# MICROMIPS: lui $4, 1                # encoding: [0xa4,0x41,0x01,0x00]
# MICROMIPS: sltu $4, $4, $5          # encoding: [0xa4,0x00,0x90,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLTu_MM
  sltu $4, $5, 0xFFFFFFFF
# MIPS: sltiu $4, $5, -1              # encoding: [0xff,0xff,0xa4,0x2c]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLTiu
# MICROMIPS: sltiu $4, $5, -1         # encoding: [0x85,0xb0,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLTiu_MM
  sltu $4, $5, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $4, 255                 # encoding: [0xff,0x00,0x04,0x3c]
# MIPS: ori   $4, $4, 65295           # encoding: [0x0f,0xff,0x84,0x34]
# MIPS: sltu  $4, $4, $5              # encoding: [0x2b,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} SLTu
# MICROMIPS: lui $4, 255              # encoding: [0xa4,0x41,0xff,0x00]
# MICROMIPS: ori $4, $4, 65295        # encoding: [0x84,0x50,0x0f,0xff]
# MICROMIPS: sltu $4, $4, $5          # encoding: [0xa4,0x00,0x90,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} SLTu_MM

  xor $4, -0x80000000
# MIPS: lui   $1, 32768               # encoding: [0x00,0x80,0x01,0x3c]
# MIPS: xor   $4, $4, $1              # encoding: [0x26,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XOR
# MICROMIPS: lui $1, 32768            # encoding: [0xa1,0x41,0x00,0x80]
# MICROMIPS: xor $4, $4, $1           # encoding: [0x24,0x00,0x10,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XOR_MM
  xor $4, -0x8001
# MIPS: lui   $1, 65535               # encoding: [0xff,0xff,0x01,0x3c]
# MIPS: ori   $1, $1, 32767           # encoding: [0xff,0x7f,0x21,0x34]
# MIPS: xor   $4, $4, $1              # encoding: [0x26,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XOR
# MICROMIPS: lui $1, 65535            # encoding: [0xa1,0x41,0xff,0xff]
# MICROMIPS: ori $1, $1, 32767        # encoding: [0x21,0x50,0xff,0x7f]
# MICROMIPS: xor $4, $4, $1           # encoding: [0x24,0x00,0x10,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XOR_MM
  xor $4, -0x8000
# MIPS: addiu $1, $zero, -32768       # encoding: [0x00,0x80,0x01,0x24]
# MIPS: xor   $4, $4, $1              # encoding: [0x26,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XOR
# MICROMIPS: addiu $1, $zero, -32768  # encoding: [0x20,0x30,0x00,0x80]
# MICROMIPS: xor $4, $4, $1           # encoding: [0x24,0x00,0x10,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XOR_MM
  xor $4, 0
# MIPS: xori  $4, $4, 0               # encoding: [0x00,0x00,0x84,0x38]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XORi
# MICROMIPS: xori $4, $4, 0           # encoding: [0x84,0x70,0x00,0x00]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XORi_MM
  xor $4, 0xFFFF
# MIPS: xori  $4, $4, 65535           # encoding: [0xff,0xff,0x84,0x38]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XORi
# MICROMIPS: xori $4, $4, 65535       # encoding: [0x84,0x70,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XORi_MM
  xor $4, 0x10000
# MIPS: lui   $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
# MIPS: xor   $4, $4, $1              # encoding: [0x26,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XOR
# MICROMIPS: lui $1, 1                # encoding: [0xa1,0x41,0x01,0x00]
# MICROMIPS: xor $4, $4, $1           # encoding: [0x24,0x00,0x10,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XOR_MM
  xor $4, 0xFFFFFFFF
# MIPS: addiu $1, $zero, -1           # encoding: [0xff,0xff,0x01,0x24]
# MIPS: xor   $4, $4, $1              # encoding: [0x26,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XOR
# MICROMIPS: addiu $1, $zero, -1      # encoding: [0x20,0x30,0xff,0xff]
# MICROMIPS: xor $4, $4, $1           # encoding: [0x24,0x00,0x10,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XOR_MM
  xor $4, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $1, 255                 # encoding: [0xff,0x00,0x01,0x3c]
# MIPS: ori   $1, $1, 65295           # encoding: [0x0f,0xff,0x21,0x34]
# MIPS: xor   $4, $4, $1              # encoding: [0x26,0x20,0x81,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XOR
# MICROMIPS: lui $1, 255              # encoding: [0xa1,0x41,0xff,0x00]
# MICROMIPS: ori $1, $1, 65295        # encoding: [0x21,0x50,0x0f,0xff]
# MICROMIPS: xor $4, $4, $1           # encoding: [0x24,0x00,0x10,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XOR_MM

  xor $4, $5, -0x80000000
# MIPS: lui   $4, 32768               # encoding: [0x00,0x80,0x04,0x3c]
# MIPS: xor   $4, $4, $5              # encoding: [0x26,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XOR
# MICROMIPS: lui $4, 32768            # encoding: [0xa4,0x41,0x00,0x80]
# MICROMIPS: xor $4, $4, $5           # encoding: [0xa4,0x00,0x10,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XOR_MM
  xor $4, $5, -0x8001
# MIPS: lui   $4, 65535               # encoding: [0xff,0xff,0x04,0x3c]
# MIPS: ori   $4, $4, 32767           # encoding: [0xff,0x7f,0x84,0x34]
# MIPS: xor   $4, $4, $5              # encoding: [0x26,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XOR
# MICROMIPS: lui $4, 65535            # encoding: [0xa4,0x41,0xff,0xff]
# MICROMIPS: ori $4, $4, 32767        # encoding: [0x84,0x50,0xff,0x7f]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ORi
# MICROMIPS: xor $4, $4, $5           # encoding: [0xa4,0x00,0x10,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XOR_MM
  xor $4, $5, -0x8000
# MIPS: addiu $4, $zero, -32768       # encoding: [0x00,0x80,0x04,0x24]
# MIPS: xor   $4, $4, $5              # encoding: [0x26,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XOR
# MICROMIPS: addiu $4, $zero, -32768  # encoding: [0x80,0x30,0x00,0x80]
# MICROMIPS: xor $4, $4, $5           # encoding: [0xa4,0x00,0x10,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XOR_MM
  xor $4, $5, 0
# MIPS: xori  $4, $5, 0               # encoding: [0x00,0x00,0xa4,0x38]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XORi
# MICROMIPS: xori $4, $5, 0           # encoding: [0x85,0x70,0x00,0x00]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XORi_MM
  xor $4, $5, 0xFFFF
# MIPS: xori  $4, $5, 65535           # encoding: [0xff,0xff,0xa4,0x38]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XORi
# MICROMIPS: xori $4, $5, 65535       # encoding: [0x85,0x70,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XORi_MM
  xor $4, $5, 0x10000
# MIPS: lui   $4, 1                   # encoding: [0x01,0x00,0x04,0x3c]
# MIPS: xor   $4, $4, $5              # encoding: [0x26,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XOR
# MICROMIPS: lui $4, 1                # encoding: [0xa4,0x41,0x01,0x00]
# MICROMIPS: xor $4, $4, $5           # encoding: [0xa4,0x00,0x10,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XOR_MM
  xor $4, $5, 0xFFFFFFFF
# MIPS: addiu $4, $zero, -1           # encoding: [0xff,0xff,0x04,0x24]
# MIPS: xor   $4, $4, $5              # encoding: [0x26,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XOR
# MICROMIPS: addiu $4, $zero, -1      # encoding: [0x80,0x30,0xff,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ADDiu
# MICROMIPS: xor $4, $4, $5           # encoding: [0xa4,0x00,0x10,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XOR_MM
  xor $4, $5, ~(0xf0000000|0x0f000000|0x000000f0)
# MIPS: lui   $4, 255                 # encoding: [0xff,0x00,0x04,0x3c]
# MIPS: ori   $4, $4, 65295           # encoding: [0x0f,0xff,0x84,0x34]
# MIPS: xor   $4, $4, $5              # encoding: [0x26,0x20,0x85,0x00]
# MIPS-NEXT:                          # <MCInst #{{[0-9]+}} XOR
# MICROMIPS: lui $4, 255              # encoding: [0xa4,0x41,0xff,0x00]
# MICROMIPS: ori $4, $4, 65295        # encoding: [0x84,0x50,0x0f,0xff]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} ORi
# MICROMIPS: xor $4, $4, $5           # encoding: [0xa4,0x00,0x10,0x23]
# MICROMIPS-NEXT:                     # <MCInst #{{[0-9]+}} XOR_MM
