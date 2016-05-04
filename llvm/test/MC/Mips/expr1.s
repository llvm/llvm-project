# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -show-encoding | \
# RUN:   FileCheck %s --check-prefix=32R2-EL
# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -mattr=micromips -show-encoding | \
# RUN:   FileCheck %s --check-prefix=MM-32R2-EL

# Check that the assembler can handle the expressions as operands.
# 32R2-EL: .text
# 32R2-EL: .globl  foo
# 32R2-EL: foo:
# 32R2-EL: lw   $4, %lo(foo)($4)       # encoding: [A,A,0x84,0x8c]
# 32R2-EL:                             #   fixup A - offset: 0, value: %lo(foo), kind: fixup_Mips_LO16
# 32R2-EL: lw   $4, 56($4)             # encoding: [0x38,0x00,0x84,0x8c]
# 32R2-EL: lw   $4, %lo(8+foo)($4)     # encoding: [A,A,0x84,0x8c]
# 32R2-EL:                             #   fixup A - offset: 0, value: %lo(8+foo), kind: fixup_Mips_LO16
# 32R2-EL: lw   $4, %lo(8+foo)($4)     # encoding: [A,A,0x84,0x8c]
# 32R2-EL:                             #   fixup A - offset: 0, value: %lo(8+foo), kind: fixup_Mips_LO16
# 32R2-EL: lw   $4, %lo(8+foo)($4)     # encoding: [A,A,0x84,0x8c]
# 32R2-EL:                             #   fixup A - offset: 0, value: %lo(8+foo), kind: fixup_Mips_LO16
# 32R2-EL: lw   $4, 10($4)             # encoding: [0x0a,0x00,0x84,0x8c]
# 32R2-EL: lw   $4, 15($4)             # encoding: [0x0f,0x00,0x84,0x8c]
# 32R2-EL: lw   $4, 21($4)             # encoding: [0x15,0x00,0x84,0x8c]
# 32R2-EL: lw   $4, 28($4)             # encoding: [0x1c,0x00,0x84,0x8c]
# 32R2-EL: lw   $4, %lo(65542)($4)     # encoding: [0x06,0x00,0x84,0x8c]
# 32R2-EL: .space  64

# MM-32R2-EL: .text
# MM-32R2-EL: .globl  foo
# MM-32R2-EL: foo:
# MM-32R2-EL: lw   $4, %lo(foo)($4)     # encoding: [0x84'A',0xfc'A',0x00,0x00]
# MM-32R2-EL:                           #   fixup A - offset: 0, value: %lo(foo), kind: fixup_MICROMIPS_LO16
# MM-32R2-EL: lw   $4, 56($4)           # encoding: [0x84,0xfc,0x38,0x00]
# MM-32R2-EL: lw   $4, %lo(8+foo)($4)   # encoding: [0x84'A',0xfc'A',0x00,0x00]
# MM-32R2-EL:                           #   fixup A - offset: 0, value: %lo(8+foo), kind: fixup_MICROMIPS_LO16
# MM-32R2-EL: lw   $4, %lo(8+foo)($4)   # encoding: [0x84'A',0xfc'A',0x00,0x00]
# MM-32R2-EL:                           #   fixup A - offset: 0, value: %lo(8+foo), kind: fixup_MICROMIPS_LO16
# MM-32R2-EL: lw   $4, %lo(8+foo)($4)   # encoding: [0x84'A',0xfc'A',0x00,0x00]
# MM-32R2-EL:                           #   fixup A - offset: 0, value: %lo(8+foo), kind: fixup_MICROMIPS_LO16
# MM-32R2-EL: lw   $4, 10($4)           # encoding: [0x84,0xfc,0x0a,0x00]
# MM-32R2-EL: lw   $4, 15($4)           # encoding: [0x84,0xfc,0x0f,0x00]
# MM-32R2-EL: lw   $4, 21($4)           # encoding: [0x84,0xfc,0x15,0x00]
# MM-32R2-EL: lw   $4, 28($4)           # encoding: [0x84,0xfc,0x1c,0x00]
# MM-32R2-EL: lw   $4, %lo(65542)($4)   # encoding: [0x84,0xfc,0x06,0x00]
# MM-32R2-EL: .space  64

  .globl  foo
  .ent  foo
foo:
  lw  $4,%lo(foo)($4)
  lw  $4,((10 + 4) * 4)($4)
  lw  $4,%lo (2 * 4) + foo($4)
  lw  $4,%lo((2 * 4) + foo)($4)
  lw  $4,(((%lo ((2 * 4) + foo))))($4)
  lw  $4, (((1+2)+3)+4)($4)
  lw  $4, ((((1+2)+3)+4)+5)($4)
  lw  $4, (((((1+2)+3)+4)+5)+6)($4)
  lw  $4, ((((((1+2)+3)+4)+5)+6)+7)($4)
  lw  $4, (%lo((1+2)+65536)+3)($4)
  .space  64
  .end  foo
