# RUN: llvm-mc %s -triple=mipsel-unknown-linux -mcpu=mips32r2 -mattr=+msa | \
# RUN:   FileCheck %s
# .set push creates a copy of the current environment.
# .set pop restores the previous environment.
# FIXME: Also test resetting of .set macro/nomacro option.

    .text
    # The first environment on the stack (with initial values).
    lw       $1, 65536($1)
    b        1336
    addvi.b  $w15, $w13, 18
    
    # Create a new environment.
    .set push
    .set at=$ra           # Test the ATReg option.
    lw       $1, 65536($1)
    .set noreorder        # Test the Reorder option.
    b        1336

    # Switch back to the first environment.
    .set pop
    lw       $1, 65536($1)
    b        1336
    addvi.b  $w15, $w13, 18

# CHECK:  lui      $1, 1
# CHECK:  addu     $1, $1, $1
# CHECK:  lw       $1, 0($1)
# CHECK:  b        1336
# CHECK:  nop
# CHECK:  addvi.b  $w15, $w13, 18

# CHECK:  .set push
# CHECK:  lui      $ra, 1
# CHECK:  addu     $ra, $ra, $1
# CHECK:  lw       $1, 0($ra)
# CHECK:  .set noreorder   
# CHECK:  b        1336
# CHECK-NOT:  nop

# CHECK:  .set pop
# CHECK:  lui      $1, 1
# CHECK:  addu     $1, $1, $1
# CHECK:  lw       $1, 0($1)
# CHECK:  b        1336
# CHECK:  nop
# CHECK:  addvi.b  $w15, $w13, 18
