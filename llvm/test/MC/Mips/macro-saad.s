# RUN: llvm-mc -triple=mips64 -show-encoding -mcpu=octeon+ %s \
# RUN:   | FileCheck -check-prefix=MIPS64 %s

saad  $2, 8($5)

# MIPS64:      daddiu  $1, $5, 8  # encoding: [0x64,0xa1,0x00,0x08]
# MIPS64-NEXT: saad    $2, ($1)   # encoding: [0x70,0x22,0x00,0x19]

saad  $2, foo

# MIPS64:      lui    $1, %highest(foo)     # encoding: [0x3c,0x01,A,A]
# MIPS64-NEXT:                              #   fixup A - offset: 0, value: %highest(foo), kind: fixup_Mips_HIGHEST
# MIPS64-NEXT: daddiu $1, $1, %higher(foo)  # encoding: [0x64,0x21,A,A]
# MIPS64-NEXT:                              #   fixup A - offset: 0, value: %higher(foo), kind: fixup_Mips_HIGHER
# MIPS64-NEXT: dsll   $1, $1, 16            # encoding: [0x00,0x01,0x0c,0x38]
# MIPS64-NEXT: daddiu $1, $1, %hi(foo)      # encoding: [0x64,0x21,A,A]
# MIPS64-NEXT:                              #   fixup A - offset: 0, value: %hi(foo), kind: fixup_Mips_HI16
# MIPS64-NEXT: dsll   $1, $1, 16            # encoding: [0x00,0x01,0x0c,0x38]
# MIPS64-NEXT: daddiu $1, $1, %lo(foo)      # encoding: [0x64,0x21,A,A]
# MIPS64-NEXT:                              #   fixup A - offset: 0, value: %lo(foo), kind: fixup_Mips_LO16
# MIPS64-NEXT: saad   $2, ($1)              # encoding: [0x70,0x22,0x00,0x19]

.option pic2
saad  $2, foo

# MIPS64:      ld      $1, %got_disp(foo)($gp)  # encoding: [0xdf,0x81,A,A]
# MIPS64-NEXT:                                  #   fixup A - offset: 0, value: %got_disp(foo), kind: fixup_Mips_GOT_DISP
# MIPS64-NEXT: saad    $2, ($1)                 # encoding: [0x70,0x22,0x00,0x19]
