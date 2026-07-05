@@ Check that PC-relative memory addressing is annotated

@ RUN: llvm-mc %s -triple=thumbv7 -filetype=obj | \
@ RUN:   llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=thumbv7 - | \
@ RUN:   FileCheck %s

.text
foo:
@ CHECK:      00000000 <foo>:
  .word 0x01020304

_start:
@ CHECK:      00000004 <_start>:

@@ Check a special case immediate for AddrModeT2_pc
  .balign 4
  ldr r0, [pc, #-0]
@ CHECK:         4: ldr.w   r0, [pc, #-0]           @ 0x8 <_start+0x4>

@@ Same instruction, but the address is not 4-byte aligned
  nop
  ldr r0, [pc, #-0]
@ CHECK:         a: ldr.w   r0, [pc, #-0]           @ 0xc <_start+0x8>

@@ Check a special case immediate for AddrModeT2_i8s4
  .balign 4
  ldrd r0, r1, [pc, #-0]
@ CHECK:        10: ldrd    r0, r1, [pc, #-0]       @ 0x14 <_start+0x10>

@@ Same instruction, but the address is not 4-byte aligned
  nop
  ldrd r0, r1, [pc, #-0]
@ CHECK:        16: ldrd    r0, r1, [pc, #-0]       @ 0x18 <_start+0x14>

@@ Check AddrModeT2_pc instructions, with positive and negative immediates
  .balign 4
  ldr r0, foo
  ldrb r0, bar
  ldrsb r0, foo
  ldrsh r0, bar
  pli _start
  pld bar
@ CHECK:        1c: ldr.w   r0, [pc, #-32]          @ 0x0 <foo>
@ CHECK-NEXT:   20: ldrb.w  r0, [pc, #112]          @ 0x94 <bar>
@ CHECK-NEXT:   24: ldrsb.w r0, [pc, #-40]          @ 0x0 <foo>
@ CHECK-NEXT:   28: ldrsh.w r0, [pc, #104]          @ 0x94 <bar>
@ CHECK-NEXT:   2c: pli     [pc, #-44]              @ 0x4 <_start>
@ CHECK-NEXT:   30: pld     [pc, #96]               @ 0x94 <bar>

@@ Same instructions, but the addresses are not 4-byte aligned
  nop
  ldr r0, foo
  ldrb r0, bar
  ldrsb r0, foo
  ldrsh r0, bar
  pli _start
  pld bar
@ CHECK:        36: ldr.w   r0, [pc, #-56]          @ 0x0 <foo>
@ CHECK-NEXT:   3a: ldrb.w  r0, [pc, #88]           @ 0x94 <bar>
@ CHECK-NEXT:   3e: ldrsb.w r0, [pc, #-64]          @ 0x0 <foo>
@ CHECK-NEXT:   42: ldrsh.w r0, [pc, #80]           @ 0x94 <bar>
@ CHECK-NEXT:   46: pli     [pc, #-68]              @ 0x4 <_start>
@ CHECK-NEXT:   4a: pld     [pc, #72]               @ 0x94 <bar>

@@ Check AddrModeT2_i8s4 instructions, with positive and negative immediates
  .balign 4
  ldrd r0, r1, foo
  ldrd r0, r1, bar
@ CHECK:        50: ldrd    r0, r1, [pc, #-84]      @ 0x0 <foo>
@ CHECK-NEXT:   54: ldrd    r0, r1, [pc, #60]       @ 0x94 <bar>

@@ Same instructions, but the addresses are not 4-byte aligned
  nop
  ldrd r0, r1, foo
  ldrd r0, r1, bar
@ CHECK:        5a: ldrd    r0, r1, [pc, #-92]      @ 0x0 <foo>
@ CHECK-NEXT:   5e: ldrd    r0, r1, [pc, #52]       @ 0x94 <bar>

@@ Check that AddrModeT2_i8s4 instructions that do not use PC-relative
@@ addressingare not annotated
  ldrd  r0, r1, [r2, #8]
@ CHECK:        62: ldrd    r0, r1, [r2, #8]{{$}}

@@ Check AddrMode5 instructions, with positive and negative immediates
  .balign 4
  ldc   p14, c5, foo
  ldcl  p6, c4, bar
  ldc2  p5, c2, foo
  ldc2l p3, c1, bar
@ CHECK:        68: ldc     p14, c5, [pc, #-108]    @ 0x0 <foo>
@ CHECK-NEXT:   6c: ldcl    p6, c4, [pc, #36]       @ 0x94 <bar>
@ CHECK-NEXT:   70: ldc2    p5, c2, [pc, #-116]     @ 0x0 <foo>
@ CHECK-NEXT:   74: ldc2l   p3, c1, [pc, #28]       @ 0x94 <bar>

@@ Same instructions, but the addresses are not 4-byte aligned
  nop
  ldc   p14, c5, foo
  ldcl  p6, c4, bar
  ldc2  p5, c2, foo
  ldc2l p3, c1, bar
@ CHECK:        7a: ldc     p14, c5, [pc, #-124]    @ 0x0 <foo>
@ CHECK-NEXT:   7e: ldcl    p6, c4, [pc, #20]       @ 0x94 <bar>
@ CHECK-NEXT:   82: ldc2    p5, c2, [pc, #-132]     @ 0x0 <foo>
@ CHECK-NEXT:   86: ldc2l   p3, c1, [pc, #12]       @ 0x94 <bar>

@@ Check that AddrMode5 instruction that do not use PC+imm addressing are not
@@ annotated
  ldc   p14, c5, [r1, #8]
  ldc   p14, c5, [pc], {16}
@ CHECK:        8a: ldc     p14, c5, [r1, #8]{{$}}
@ CHECK-NEXT:   8e: ldc     p14, c5, [pc], {16}{{$}}

  .balign 4
bar:
@ CHECK:      00000094 <bar>:
  .word 0x01020304
