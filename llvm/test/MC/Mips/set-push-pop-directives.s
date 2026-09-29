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
    .set nomsa            # Test the Features option (ASE).
    .set mips32r6         # Test the Features option (ISA).
    mod      $2, $4, $6

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
# CHECK:  .set nomsa       
# CHECK:  .set mips32r6    
# CHECK:  mod      $2, $4, $6

# CHECK:  .set pop
# CHECK:  lui      $1, 1
# CHECK:  addu     $1, $1, $1
# CHECK:  lw       $1, 0($1)
# CHECK:  b        1336
# CHECK:  nop
# CHECK:  addvi.b  $w15, $w13, 18

    .set push
    .set dsp
    lbux    $7, $10($11)
    .set pop

    .set push
    .set dsp
    lbux    $7, $10($11)
# CHECK-NOT: :[[@LINE-1]]:5: error: instruction requires a CPU feature not currently enabled
    .set pop

    .set push
    .set dsp
    lbux    $7, $10($11)
# CHECK-NOT: :[[@LINE-1]]:5: error: instruction requires a CPU feature not currently enabled
    .set pop
