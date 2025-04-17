@@ Check that PC-relative memory addressing is annotated

@ RUN: llvm-mc %s -triple=armv8a --mattr=+fullfp16 -filetype=obj | \
@ RUN:   llvm-objdump -d --no-show-raw-insn --triple=armv8a --mattr=+fullfp16 - | \
@ RUN:   FileCheck %s

.text
foo:
@ CHECK:      00000000 <foo>:
  .short 0x0102
foo2:
@ CHECK:      00000002 <foo2>:
  .short 0x0304

_start:
@ CHECK:      00000004 <_start>:
@@ Check AddrMode5 instructions, with positive and negative immediates
  vldr d0, foo
  vldr s0, bar
@ CHECK-NEXT:    4: vldr    d0, [pc, #-12]          @ 0x0 <foo>
@ CHECK-NEXT:    8: vldr    s0, [pc, #20]           @ 0x24 <bar>

@@ Check that AddrMode5 instructions which do not use PC-relative addressing are
@@ not annotated
  vldr d0, [r1, #8]
@ CHECK-NEXT:    c: vldr    d0, [r1, #8]{{$}}

@@ Check AddrMode5FP16 instructions, with positive and negative immediates
  vldr.16 s0, foo
  vldr.16 s0, foo2
  vldr.16 s1, bar
  vldr.16 s1, bar2
@ CHECK-NEXT:   10: vldr.16 s0, [pc, #-24]          @ 0x0 <foo>
@ CHECK-NEXT:   14: vldr.16 s0, [pc, #-26]          @ 0x2 <foo2>
@ CHECK-NEXT:   18: vldr.16 s1, [pc, #4]            @ 0x24 <bar>
@ CHECK-NEXT:   1c: vldr.16 s1, [pc, #2]            @ 0x26 <bar2>

@@ Check that AddrMode5FP16 instructions which do not use PC-relative addressing
@@ are not annotated
  vldr.16 s0, [r1, #8]
@ CHECK-NEXT:   20: vldr.16 s0, [r1, #8]{{$}}

bar:
@ CHECK:      00000024 <bar>:
  .short 0x0102
bar2:
@ CHECK:      00000026 <bar2>:
  .short 0x0304
