@ RUN: llvm-mc < %s --triple=armv8a   -mattr=+mve,+lob -filetype=obj | llvm-objdump -dr - --triple armv8a --mattr=+mve,+lob --no-show-raw-insn | FileCheck %s
@ RUN: llvm-mc < %s --triple=thumbv8a -mattr=+mve,+lob -filetype=obj | llvm-objdump -dr - --triple armv8a --mattr=+mve,+lob --no-show-raw-insn | FileCheck %s

foo:

  // Branches
  .arm
  b foo
  ble foo
@ CHECK:       0: b       #-8 <foo>
@ CHECK:       4: ble     #-12 <foo>

  .thumb
  b foo
  b.w foo
  ble foo
  ble.w foo
  le foo
  le lr, foo
  cbz r0, bar
  cbnz r0, bar
@ CHECK:       8: b       #-12 <foo>
@ CHECK:       a: b.w     #-14 <foo>
@ CHECK:       e: ble     #-18 <foo>
@ CHECK:      10: ble.w   #-20 <foo>
@ CHECK:      14: le      #-24 <foo>
@ CHECK:      18: le      lr, #-28 <foo>
@ CHECK:      1c: cbz     r0, #40 <bar>
@ CHECK:      1e: cbnz    r0, #38 <bar>

  // Calls without relocations (these offsets al correspond to label foo).
  .arm
  bl #-40
  blx #-44
  bleq #-48
@ CHECK:      20:   bl      #-40 <foo>
@ CHECK:      24:   blx     #-44 <foo>
@ CHECK:      28:   bleq    #-48 <foo>

  .thumb
  bl #-48
  blx #-52
@ CHECK:      2c:   bl      #-48 <foo>
@ CHECK:      30:   blx     #-52 <foo>

  // Calls with relocations. These currently emit a reference to their own
  // location, because we don't take relocations into account when printing
  // branch targets.
  .arm
  bl baz
  blx baz
  bleq baz
@ CHECK:      34:   bl      #-8 <$a.4>
@ CHECK:            00000034:  R_ARM_CALL   baz
@ CHECK:      38:   blx     #-8 <$a.4+0x4>
@ CHECK:            00000038:  R_ARM_CALL   baz
@ CHECK:      3c:   bleq    #-8 <$a.4+0x8>
@ CHECK:            0000003c:  R_ARM_JUMP24 baz

  .thumb
  bl baz
  blx baz
@ CHECK:      40:   bl      #-4 <$t.5>
@ CHECK:            00000040:  R_ARM_THM_CALL       baz
@ CHECK:      44:   blx     #-4 <$t.5+0x4>
@ CHECK:            00000044:  R_ARM_THM_CALL       baz

bar:


