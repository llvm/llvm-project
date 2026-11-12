# RUN: not llvm-mc -triple=systemz -mcpu=z13 -show-encoding < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=systemz -mcpu=zEC12 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ZEC12

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

# Check that we get one suggestion: 'cpdt' is 1 edit away, i.e. an deletion.

  cpdtX %r1, 0(4, %r15), 0

#CHECK:      error: invalid instruction, did you mean: cpdt
#CHECK-NEXT: cpdtX %r1, 0(4, %r15), 0
#CHECK-NEXT: ^

# Check edit distance 1 and 2

  ltTr %r1, %r2

# CHECK:      error: invalid instruction, did you mean: lr, lt, ltdr, ltdtr, lter, ltgr, ltr, ltxr, ltxtr, tr, trtr?
# CHECK-NEXT: ltTr %r1, %r2
# CHECK-NEXT: ^

# Check edit distance 1 and 2, just insertions:

  begin 0, 65292

# CHECK:      error: invalid instruction, did you mean: tbegin, tbeginc?
# CHECK-NEXT: begin 0, 65292
# CHECK-NEXT: ^

# Check an instruction that is 2 edits away, and also has a lot of candidates:

  adt %r1, 244(%r15)

# CHECK:      error: invalid instruction, did you mean: a, ad, adb, adr, adtr, adtra, d, lat, mad, qadtr?
# CHECK-NEXT: adt %r1, 244(%r15)
# CHECK-NEXT: ^

# Here it is checked that we don't suggest instructions that are not supported.
# For example, in pre-z13 mode we don't want to see suggestions for vector instructions.

  vlvggp %v1, %r2, %r3

# CHECK-ZEC12: error: invalid instruction
# CHECK-ZEC12: vlvggp
# CHECK-ZEC12: ^

# CHECK:      error: invalid instruction, did you mean: vlvg, vlvgg, vlvgp?
# CHECK-NEXT: vlvggp %v1, %r2, %r3
# CHECK-NEXT: ^
