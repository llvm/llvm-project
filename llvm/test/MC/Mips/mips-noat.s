# RUN: not llvm-mc %s -triple=mips-unknown-linux 2>%t0 | FileCheck %s
# RUN: FileCheck -check-prefix=ERROR %s < %t0
# Check that using the assembler temporary when .set noat is in effect is an error.

# We start with the assembler temporary enabled
# CHECK-LABEL: test1:
# CHECK:  lui   $1, 1
# CHECK:  addu  $1, $1, $2
# CHECK:  lw    $2, 0($1)
# CHECK-LABEL: test2:
# CHECK:  .set noat
test1:
        lw      $2, 65536($2)

test2:
        .set noat
        lw      $2, 65536($2) # ERROR: mips-noat.s:[[@LINE]]:9: error: pseudo-instruction requires $at, which is not available


# Can we switch it back on successfully?
# CHECK-LABEL: test3:
# CHECK:  lui   $1, 1
# CHECK:  addu  $1, $1, $2
# CHECK:  lw    $2, 0($1)
# CHECK-LABEL: test4:
# CHECK:  .set  at=$0
test3:
        .set at
        lw      $2, 65536($2)

test4:
        .set at=$0
        lw      $2, 65536($2) # ERROR: mips-noat.s:[[@LINE]]:9: error: pseudo-instruction requires $at, which is not available
