@@ Check that PC-relative memory addressing is annotated

@ RUN: llvm-mc %s -triple=armv7 -filetype=obj | \
@ RUN:   llvm-objdump -d --no-show-raw-insn --triple=armv7 - | \
@ RUN:   FileCheck %s

.text
foo:
@ CHECK:      00000000 <foo>:
  .word 0x01020304

_start:
@ CHECK:      00000004 <_start>:

@@ Check a special case immediate for AddrMode_i12
  ldr   r1, [pc, #-0]
@ CHECK-NEXT:    4: ldr   r1, [pc, #-0]           @ 0xc <_start+0x8>

@@ Check AddrMode_i12 instructions, with positive and negative immediates
  ldr   r0, foo
  ldrb  r0, bar
  pli   _start
  pld   bar
@ CHECK-NEXT:    8: ldr   r0, [pc, #-16]          @ 0x0 <foo>
@ CHECK-NEXT:    c: ldrb  r0, [pc, #48]           @ 0x44 <bar>
@ CHECK-NEXT:   10: pli   [pc, #-20]              @ 0x4 <_start>
@ CHECK-NEXT:   14: pld   [pc, #40]               @ 0x44 <bar>

@@ Check that AddrMode_i12 instructions that do not use PC-relative addressing
@@ are not annotated
  ldr   r0, [r1, #8]
@ CHECK-NEXT:   18: ldr   r0, [r1, #8]{{$}}

@@ Check AddrMode3 instructions, with positive and negative immediates
  ldrd  r0, r1, foo
  ldrh  r0, bar
@ CHECK-NEXT:   1c: ldrd  r0, r1, [pc, #-36]      @ 0x0 <foo>
@ CHECK-NEXT:   20: ldrh  r0, [pc, #28]           @ 0x44 <bar>

@@ Check that AddrMode3 instruction that do not use PC+imm addressing are not
@@ annotated
  ldrh  r0, [r1, #8]
  ldrh  r0, [pc, r2]
@ CHECK-NEXT:   24: ldrh  r0, [r1, #8]{{$}}
@ CHECK-NEXT:   28: ldrh  r0, [pc, r2]{{$}}

@@ Check AddrMode5 instructions, with positive and negative immediates
  ldc   p14, c5, foo
  ldcl  p6, c4, bar
  ldc2  p5, c2, foo
  ldc2l p3, c1, bar
@ CHECK-NEXT:   2c: ldc   p14, c5, [pc, #-52]     @ 0x0 <foo>
@ CHECK-NEXT:   30: ldcl  p6, c4, [pc, #12]       @ 0x44 <bar>
@ CHECK-NEXT:   34: ldc2  p5, c2, [pc, #-60]      @ 0x0 <foo>
@ CHECK-NEXT:   38: ldc2l p3, c1, [pc, #4]        @ 0x44 <bar>

@@ Check that AddrMode5 instruction that do not use PC+imm addressing are not
@@ annotated
  ldc   p14, c5, [r1, #8]
  ldc   p14, c5, [pc], {16}
@ CHECK-NEXT:   3c: ldc   p14, c5, [r1, #8]{{$}}
@ CHECK-NEXT:   40: ldc   p14, c5, [pc], {16}{{$}}

bar:
@ CHECK:      00000044 <bar>:
  .word 0x01020304
