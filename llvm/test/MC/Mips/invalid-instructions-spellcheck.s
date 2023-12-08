# RUN: not llvm-mc -triple mips-unknown-linux-gnu -mcpu=mips32r2 \
# RUN:             -mattr=+micromips %s 2>&1 \
# RUN:   | FileCheck --check-prefixes=ALL,MMR2 %s
# RUN: not llvm-mc -triple mips-unknown-linux-gnu -mcpu=mips32r6 \
# RUN:             -mattr=+micromips %s 2>&1 \
# RUN:   | FileCheck --check-prefixes=ALL,MMR6 %s
# RUN: not llvm-mc -triple mips-unknown-linux-gnu -mcpu=mips32r6 %s 2>&1 \
# RUN:   | FileCheck --check-prefixes=ALL,MIPS32R6 %s

# This tests the mnemonic spell checker.

# First check what happens when an instruction is omitted:

$2, $1, $25

# ALL:      error: unknown instruction
# ALL-NEXT: $2, $1, $25
# ALL-NEXT:  ^

# We don't want to see a suggestion here; the edit distance is too large to
# give sensible suggestions:

aaaaaaaaaaaaaaa $2, $1, $25

# ALL:      error: unknown instruction
# ALL-NEXT: aaaaaaaaaaaaaaa $2, $1, $25
# ALL-NEXT: ^

# Check that we get one suggestion: 'addiuspi' is 1 edit away, i.e. an deletion.

addiuspi -16

# MMR2:     error: unknown instruction, did you mean: addiusp?
# MMR6:     error: unknown instruction, did you mean: addiusp?
# MIPS32R6: error: unknown instruction{{$}}
# ALL:      addiuspi -16
# ALL-NEXT: ^

# Check edit distance 1 and 2, just insertions:

addru $9, $6, 17767

# MMR2:      error: unknown instruction, did you mean: add, addiu, addu, maddu?
# MMR6:      error: unknown instruction, did you mean: add, addiu, addu?
# MIPS32R6:  error: unknown instruction, did you mean: add, addiu, addu?
# ALL:       addru $9, $6, 17767
# ALL-NEXT:  ^

# Check an instruction that is 2 edits away, and also has a lot of candidates:

culE.d  $fcc7, $f24, $f18

# MMR2:     error: unknown instruction, did you mean: c.le.d, c.ule.d?
# MMR6:     error: unknown instruction{{$}}
# MIPS32R6: error: unknown instruction{{$}}
# ALL:      culE.d  $fcc7, $f24, $f18
# ALL-NEXT: ^

# Check that candidates list includes only instructions valid for target CPU.

swk $3, $4

# MMR2: error: unknown instruction, did you mean: sw, swl, swm, swp, swr, usw?
# MMR6: error: unknown instruction, did you mean: sw, swm, swp, usw?
# MIPS32R6: error: unknown instruction, did you mean: sw, usw?
# ALL:      swk $3, $4
# ALL-NEXT: ^
