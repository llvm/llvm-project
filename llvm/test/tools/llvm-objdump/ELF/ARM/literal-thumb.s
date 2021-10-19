@@ Check that PC-relative memory addressing is annotated

@ RUN: llvm-mc %s -triple=thumbv6m -filetype=obj | \
@ RUN:   llvm-objdump -d --no-show-raw-insn --triple=thumbv6m - | \
@ RUN:   FileCheck %s

.text
_start:
@ CHECK:      00000000 <_start>:

@@ Check AddrModeT1_s instruction, with 4-byte and 2-byte alignment
  ldr r0, bar
  ldr r1, bar
  ldr r2, bar
  ldr r3, bar
@ CHECK-NEXT:   0: ldr    r0, [pc, #4]            @ 0x8 <bar>
@ CHECK-NEXT:   2: ldr    r1, [pc, #4]            @ 0x8 <bar>
@ CHECK-NEXT:   4: ldr    r2, [pc, #0]            @ 0x8 <bar>
@ CHECK-NEXT:   6: ldr    r3, [pc, #0]            @ 0x8 <bar>

  .balign 4
bar:
@ CHECK:      00000008 <bar>:
  .word 0x01020304
