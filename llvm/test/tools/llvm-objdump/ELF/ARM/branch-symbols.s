@ RUN: llvm-mc < %s --triple=armv8a   -mattr=+mve,+lob -filetype=obj | llvm-objdump --no-print-imm-hex -dr - --triple armv8a --mattr=+mve,+lob --no-show-raw-insn | FileCheck %s
@ RUN: llvm-mc < %s --triple=thumbv8a -mattr=+mve,+lob -filetype=obj | llvm-objdump --no-print-imm-hex -dr - --triple armv8a --mattr=+mve,+lob --no-show-raw-insn | FileCheck %s

foo:

  // Branches
  .arm
  b foo
  ble foo
@ CHECK:       0: b       0x0 <foo>                 @ imm = #-8
@ CHECK:       4: ble     0x0 <foo>                 @ imm = #-12

  .thumb
  b foo
  b.w foo
  ble foo
  ble.w foo
  le foo
  le lr, foo
  cbz r0, bar
  cbnz r0, bar
@ CHECK:       8: b       0x0 <foo>                 @ imm = #-12
@ CHECK:       a: b.w     0x0 <foo>                 @ imm = #-14
@ CHECK:       e: ble     0x0 <foo>                 @ imm = #-18
@ CHECK:      10: ble.w   0x0 <foo>                 @ imm = #-20
@ CHECK:      14: le      0x0 <foo>                 @ imm = #-24
@ CHECK:      18: le      lr, 0x0 <foo>             @ imm = #-28
@ CHECK:      1c: cbz     r0, 0x48 <bar>            @ imm = #40
@ CHECK:      1e: cbnz    r0, 0x48 <bar>            @ imm = #38

  // Calls without relocations (these offsets al correspond to label foo).
  .arm
  bl #-40
  blx #-44
  bleq #-48
@ CHECK:      20:   bl      0x0 <foo>               @ imm = #-40
@ CHECK:      24:   blx     0x0 <foo>               @ imm = #-44
@ CHECK:      28:   bleq    0x0 <foo>               @ imm = #-48

  .thumb
  bl #-48
  blx #-52
@ CHECK:      2c:   bl      0x0 <foo>               @ imm = #-48
@ CHECK:      30:   blx     0x0 <foo>               @ imm = #-52

  // Calls with relocations. These currently emit a reference to their own
  // location, because we don't take relocations into account when printing
  // branch targets.
  .arm
  bl baz
  blx baz
  bleq baz
@ CHECK:      34:   bl      {{.+}}                  @ imm = #-8
@ CHECK:            00000034:  R_ARM_CALL   baz
@ CHECK:      38:   blx     {{.+}}                  @ imm = #-8
@ CHECK:            00000038:  R_ARM_CALL   baz
@ CHECK:      3c:   bleq    {{.+}}                  @ imm = #-8
@ CHECK:            0000003c:  R_ARM_JUMP24 baz

  .thumb
  bl baz
  blx baz
@ CHECK:      40:   bl      {{.+}}                  @ imm = #-4
@ CHECK:            00000040:  R_ARM_THM_CALL baz
@ CHECK:      44:   blx     {{.+}}                  @ imm = #-4
@ CHECK:            00000044:  R_ARM_THM_CALL baz

bar:
