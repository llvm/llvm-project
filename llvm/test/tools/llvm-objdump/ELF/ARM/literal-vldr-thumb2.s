@@ Check that PC-relative memory addressing is annotated

@ RUN: llvm-mc %s -triple=thumbv8a --mattr=+fullfp16 -filetype=obj | \
@ RUN:   llvm-objdump -d --no-show-raw-insn --triple=thumbv8a --mattr=+fullfp16 - | \
@ RUN:   FileCheck %s

.text
foo:
@ CHECK:      00000000 <foo>:
  .short 0x0102
foo2:
@ CHECK:      00000002 <foo2>:
  .short 0x0304

_start:
@@ Check AddrMode5 instructions, with positive and negative immediates
  .balign 4
  vldr d0, foo
  vldr s0, bar
@ CHECK:         4: vldr      d0, [pc, #-8]           @ 0x0 <foo>
@ CHECK-NEXT:    8: vldr      s0, [pc, #56]           @ 0x44 <bar>

@@ Same instructions, but the addresses are not 4-byte aligned
  nop
  vldr d0, foo
  vldr s0, bar
@ CHECK:          e: vldr     d0, [pc, #-16]          @ 0x0 <foo>
@ CHECK-NEXT:    12: vldr     s0, [pc, #48]           @ 0x44 <bar>

@@ Check that AddrMode5 instructions which do not use PC-relative addressing are not annotated
  vldr d0, [r1, #8]
@ CHECK:         16: vldr     d0, [r1, #8]{{$}}

@@ Check AddrMode5FP16 instructions, with positive and negative immediates
  .balign 4
  vldr.16 s0, foo
  vldr.16 s0, foo2
  vldr.16 s1, bar
  vldr.16 s1, bar2
@ CHECK:         1c: vldr.16  s0, [pc, #-32]          @ 0x0 <foo>
@ CHECK-NEXT:    20: vldr.16  s0, [pc, #-34]          @ 0x2 <foo2>
@ CHECK-NEXT:    24: vldr.16  s1, [pc, #28]           @ 0x44 <bar>
@ CHECK-NEXT:    28: vldr.16  s1, [pc, #26]           @ 0x46 <bar2>

@@ Same instructions, but the addresses are not 4-byte aligned
  nop
  vldr.16 s0, foo
  vldr.16 s0, foo2
  vldr.16 s1, bar
  vldr.16 s1, bar2
@ CHECK:         2e: vldr.16  s0, [pc, #-48]          @ 0x0 <foo>
@ CHECK-NEXT:    32: vldr.16  s0, [pc, #-50]          @ 0x2 <foo2>
@ CHECK-NEXT:    36: vldr.16  s1, [pc, #12]           @ 0x44 <bar>
@ CHECK-NEXT:    3a: vldr.16  s1, [pc, #10]           @ 0x46 <bar2>

@@ Check that AddrMode5FP16 instructions which do not use PC-relative addressing are not annotated
  vldr.16 s0, [r1, #8]
@ CHECK:         3e: vldr.16  s0, [r1, #8]{{$}}

  .balign 4
bar:
@ CHECK:      00000044 <bar>:
  .short 0x0102
bar2:
@ CHECK:      00000046 <bar2>:
  .short 0x0304
