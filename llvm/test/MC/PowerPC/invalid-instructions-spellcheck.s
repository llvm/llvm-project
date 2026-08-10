# RUN: not llvm-mc -triple powerpc-unknown-unknown < %s 2>&1 | FileCheck %s

# This tests the mnemonic spell checker.

# First check what happens when an instruction is omitted:

%r1, %r2, %r3

# CHECK:      error: unexpected token at start of statement
# CHECK-NEXT: %r1, %r2, %r3
# CHECK-NEXT:   ^

# We don't want to see a suggestion here; the edit distance is too large to
# give sensible suggestions:

aaaaaaaaaaaaaaa %r1, %r2, %r3

# CHECK:      error: invalid instruction
# CHECK-NEXT: aaaaaaaaaaaaaaa %r1, %r2, %r3
# CHECK-NEXT: ^

# Check that we get one suggestion: 'vmaxfpg' is 1 edit away, i.e. an deletion.

vmaxfpg %r1, %r2

# CHECK:      error: invalid instruction, did you mean: vmaxfp?
# CHECK-NEXT: vmaxfpg %r1, %r2
# CHECK-NEXT: ^

# Check edit distance 1 and 2, just insertions:

xsnmsubad %r1, %r2

# CHECK:      error: invalid instruction, did you mean: xsmsubadp, xsnmsubadp?
# CHECK-NEXT: xsnmsubad %r1, %r2
# CHECK-NEXT: ^

# Check an instruction that is 2 edits away, and also has a lot of candidates:

adXd %r1, %r2, %r3

# CHECK:      error: invalid instruction, did you mean: add, addc, adde, addi, addo, dadd, fadd?
# CHECK-NEXT: adXd %r1, %r2, %r3
# CHECK-NEXT: ^
