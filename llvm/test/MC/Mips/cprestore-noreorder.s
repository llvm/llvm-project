# RUN: llvm-mc %s -arch=mips -mcpu=mips32 --position-independent -show-encoding | \
# RUN:  FileCheck %s

# RUN: llvm-mc %s -arch=mips -mcpu=mips32 --position-independent -filetype=obj -o -| \
# RUN:  llvm-objdump -d -r - | FileCheck %s -check-prefix=CHECK-FOR-STORE

# RUN: llvm-mc %s -arch=mips -mcpu=mips32 -mattr=+micromips --position-independent -show-encoding | \
# RUN:  FileCheck %s -check-prefix=MICROMIPS

# RUN: llvm-mc %s -arch=mips -mcpu=mips32 -show-encoding | \
# RUN:  FileCheck %s -check-prefix=NO-PIC

# RUN: llvm-mc %s -arch=mips -mcpu=mips64 -target-abi n32 --position-independent -show-encoding | \
# RUN:  FileCheck %s -check-prefixes=BAD-ABI,BAD-ABI-N32

# RUN: llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n64 --position-independent -show-encoding | \
# RUN:  FileCheck %s -check-prefixes=BAD-ABI,BAD-ABI-N64

  .text
  .ent foo
foo:
  .frame  $sp, 0, $ra
  .set noreorder

  .cpload $25
  .cprestore 8

  jal $25
  jal $4, $25
  jal foo

  .end foo

# CHECK-FOR-STORE: sw  $gp, 8($sp)

# CHECK: .cprestore 8
# CHECK: jalr  $25                 # encoding: [0x03,0x20,0xf8,0x09]
# CHECK: nop                       # encoding: [0x00,0x00,0x00,0x00]
# CHECK: lw    $gp, 8($sp)         # encoding: [0x8f,0xbc,0x00,0x08]

# CHECK: jalr  $4,  $25            # encoding: [0x03,0x20,0x20,0x09]
# CHECK: nop                       # encoding: [0x00,0x00,0x00,0x00]
# CHECK: lw    $gp, 8($sp)         # encoding: [0x8f,0xbc,0x00,0x08]

# CHECK: lw    $25, %got(foo)($gp) # encoding: [0x8f,0x99,A,A]
# CHECK:                           #   fixup A - offset: 0, value: %got(foo), kind: fixup_Mips_GOT
# CHECK: addiu $25, $25, %lo(foo)  # encoding: [0x27,0x39,A,A]
# CHECK:                           #   fixup A - offset: 0, value: %lo(foo), kind: fixup_Mips_LO16
# CHECK: jalr  $25                 # encoding: [0x03,0x20,0xf8,0x09]
# CHECK: nop                       # encoding: [0x00,0x00,0x00,0x00]
# CHECK: lw    $gp, 8($sp)         # encoding: [0x8f,0xbc,0x00,0x08]
# CHECK: .end  foo

# MICROMIPS: .cprestore 8
# MICROMIPS: jalrs16 $25                 # encoding: [0x45,0xf9]
# MICROMIPS: nop                         # encoding: [0x00,0x00,0x00,0x00]
# MICROMIPS: lw      $gp, 8($sp)         # encoding: [0xff,0x9d,0x00,0x08]

# MICROMIPS: jalrs   $4,  $25            # encoding: [0x00,0x99,0x4f,0x3c]
# MICROMIPS: nop                         # encoding: [0x00,0x00,0x00,0x00]
# MICROMIPS: lw      $gp, 8($sp)         # encoding: [0xff,0x9d,0x00,0x08]

# MICROMIPS: lw      $25, %got(foo)($gp) # encoding: [0xff,0x3c,A,A]
# MICROMIPS:                             #   fixup A - offset: 0, value: %got(foo), kind: fixup_MICROMIPS_GOT16
# MICROMIPS: addiu   $25, $25, %lo(foo)  # encoding: [0x33,0x39,A,A]
# MICROMIPS:                             #   fixup A - offset: 0, value: %lo(foo), kind: fixup_MICROMIPS_LO16
# MICROMIPS: jalrs   $ra, $25            # encoding: [0x03,0xf9,0x4f,0x3c]
# MICROMIPS: nop                         # encoding: [0x0c,0x00]
# MICROMIPS: lw      $gp, 8($sp)         # encoding: [0xff,0x9d,0x00,0x08]
# MICROMIPS: .end  foo

# NO-PIC:     .cprestore  8
# NO-PIC:     jalr  $25         # encoding: [0x03,0x20,0xf8,0x09]
# NO-PIC-NOT: lw    $gp, 8($sp) # encoding: [0x8f,0xbc,0x00,0x08]

# NO-PIC:     jalr  $4,  $25    # encoding: [0x03,0x20,0x20,0x09]
# NO-PIC-NOT: lw    $gp, 8($sp) # encoding: [0x8f,0xbc,0x00,0x08]

# NO-PIC:     jal   foo         # encoding: [0b000011AA,A,A,A]
# NO-PIC:                       #   fixup A - offset: 0, value: foo, kind: fixup_Mips_26
# NO-PIC-NOT: lw    $gp, 8($sp) # encoding: [0x8f,0xbc,0x00,0x08]
# NO-PIC:     .end  foo

# BAD-ABI:     .cprestore  8
# BAD-ABI:     jalr  $25                      # encoding: [0x03,0x20,0xf8,0x09]
# BAD-ABI-NOT: lw    $gp, 8($sp)              # encoding: [0x8f,0xbc,0x00,0x08]

# BAD-ABI:     jalr  $4,  $25                 # encoding: [0x03,0x20,0x20,0x09]
# BAD-ABI-NOT: lw    $gp, 8($sp)              # encoding: [0x8f,0xbc,0x00,0x08]

# BAD-ABI-N32: lw    $25, %got_disp(foo)($gp) # encoding: [0x8f,0x99,A,A]
# BAD-ABI-N64: ld    $25, %got_disp(foo)($gp) # encoding: [0xdf,0x99,A,A]
# BAD-ABI:                                    #   fixup A - offset: 0, value: %got_disp(foo), kind: fixup_Mips_GOT_DISP
# BAD-ABI:     jalr  $25                      # encoding: [0x03,0x20,0xf8,0x09]
# BAD-ABI-NOT: lw    $gp, 8($sp)              # encoding: [0x8f,0xbc,0x00,0x08]
# BAD-ABI:  .end  foo
