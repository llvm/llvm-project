@ RUN: not llvm-mc -triple=arm -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple=thumb -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-THUMB

@ This tests the mnemonic spell checker.

@ First check what happens when an instruction is omitted:

  r1, r2, r3

@ CHECK:      error: unexpected token in operand
@ CHECK-NEXT: r1, r2, r3
@ CHECK-NEXT:   ^

@ We don't want to see a suggestion here; the edit distance is too large to
@ give sensible suggestions:

  aaaaaaaaaaaaaaa r1, r2, r3

@ CHECK:      error: invalid instruction
@ CHECK-NEXT: aaaaaaaaaaaaaaa r1, r2, r3
@ CHECK-NEXT: ^

@ Check that we get one suggestion: 'pushh' is 1 edit away, i.e. an deletion.

  pushh r1, r2, r3

@CHECK:      error: invalid instruction, did you mean: push?
@CHECK-NEXT: pushh r1, r2, r3
@CHECK-NEXT: ^

  adXd r1, r2, r3

@ Check edit distance 1 and 2: 'add' has edit distance of 1 (a deletion),
@ and 'qadd' a distance of 2 (a deletion and an insertion)

@ CHECK:      error: invalid instruction, did you mean: add, qadd?
@ CHECK-NEXT: adXd r1, r2, r3
@ CHECK-NEXT: ^

@ Check edit distance 1 and 2, just insertions:

  ad r1, r2, r3

@ CHECK:      error: invalid instruction, did you mean: adc, add, adr, and, qadd?
@ CHECK-NEXT: ad r1, r2, r3
@ CHECK-NEXT: ^

@ Check an instruction that is 2 edits away, and also has a lot of candidates:

  ldre r1, r2, r3

@ CHECK:      error: invalid instruction, did you mean: ldr, ldrb, ldrd, ldrex, ldrexb, ldrexd, ldrexh, ldrh, ldrt?
@ CHECK-NEXT: ldre r1, r2, r3
@ CHECK-NEXT: ^

@ Here it is checked that we don't suggest instructions that are not supported.
@ For example, in Thumb mode we don't want to see suggestions 'faddd' of 'qadd'
@ because they are not supported.

  fadd r1, r2, r3

@ CHECK-THUMB: error: invalid instruction, did you mean: add?
@ CHECK-THUMB: fadd r1, r2, r3
@ CHECK-THUMB: ^

@ CHECK:      error: invalid instruction, did you mean: add, qadd?
@ CHECK-NEXT: fadd r1, r2, r3
@ CHECK-NEXT: ^
