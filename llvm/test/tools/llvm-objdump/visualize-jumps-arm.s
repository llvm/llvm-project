// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc < input.s -triple armv8a -filetype=obj | \
// RUN:   llvm-objdump --triple armv8a -d --visualize-jumps=unicode - | \
// RUN:   diff - expected-unicode.txt

// RUN: llvm-mc < input.s -triple armv8a -filetype=obj | \
// RUN:   llvm-objdump --triple armv8a -d --visualize-jumps=ascii - | \
// RUN:   diff - expected-ascii.txt

//--- input.s
test_func:
  // Relocated instructions don't get control-flow edges.
  bl extern_func
  b extern_func
  
  // Two branches to the same label, one forward and one backward.
  b .Llabel1
.Llabel1:
  nop
  b .Llabel1

  // Branch to self, no CFG edge shown
  b .

  // Conditional branches
  beq .Llabel2
.Llabel2:
  nop

  // Branches are sorted with shorter ones to the right, to reduce number of
  // crossings, and keep the lines for short branches short themselves.
  b .Llabel5
  b .Llabel4
  b .Llabel3
.Llabel3:
  nop
.Llabel4:
  nop
.Llabel5:
  nop

  // Sometimes crossings can't be avoided.
  b .Llabel6
  b .Llabel7
.Llabel6:
  nop
.Llabel7:
  nop

  // TODO If a branch goes to another branch instruction, we don't have a way
  // to represent that. Can we improve on this?
  b .Llabel8
.Llabel8:
  b .Llabel9
.Llabel9:
  nop

//--- expected-unicode.txt

<stdin>:	file format elf32-littlearm

Disassembly of section .text:

00000000 <test_func>:
       0:           ebfffffe   	bl	0x0 <test_func>         @ imm = #-0x8
       4:           eafffffe   	b	0x4 <test_func+0x4>     @ imm = #-0x8
       8:       ╭── eaffffff   	b	0xc <test_func+0xc>     @ imm = #-0x4
       c:       ├─> e320f000   	nop
      10:       ╰── eafffffd   	b	0xc <test_func+0xc>     @ imm = #-0xc
      14:           eafffffe   	b	0x14 <test_func+0x14>   @ imm = #-0x8
      18:       ╭── 0affffff   	beq	0x1c <test_func+0x1c>   @ imm = #-0x4
      1c:       ╰─> e320f000   	nop
      20:   ╭────── ea000003   	b	0x34 <test_func+0x34>   @ imm = #0xc
      24:   │ ╭──── ea000001   	b	0x30 <test_func+0x30>   @ imm = #0x4
      28:   │ │ ╭── eaffffff   	b	0x2c <test_func+0x2c>   @ imm = #-0x4
      2c:   │ │ ╰─> e320f000   	nop
      30:   │ ╰───> e320f000   	nop
      34:   ╰─────> e320f000   	nop
      38:       ╭── ea000000   	b	0x40 <test_func+0x40>   @ imm = #0x0
      3c:     ╭─│── ea000000   	b	0x44 <test_func+0x44>   @ imm = #0x0
      40:     │ ╰─> e320f000   	nop
      44:     ╰───> e320f000   	nop
      48:     ╭──── eaffffff   	b	0x4c <test_func+0x4c>   @ imm = #-0x4
      4c:     ╰─│─> eaffffff   	b	0x50 <test_func+0x50>   @ imm = #-0x4
      50:       ╰─> e320f000   	nop
//--- expected-ascii.txt

<stdin>:	file format elf32-littlearm

Disassembly of section .text:

00000000 <test_func>:
       0:           ebfffffe   	bl	0x0 <test_func>         @ imm = #-0x8
       4:           eafffffe   	b	0x4 <test_func+0x4>     @ imm = #-0x8
       8:       /-- eaffffff   	b	0xc <test_func+0xc>     @ imm = #-0x4
       c:       +-> e320f000   	nop
      10:       \-- eafffffd   	b	0xc <test_func+0xc>     @ imm = #-0xc
      14:           eafffffe   	b	0x14 <test_func+0x14>   @ imm = #-0x8
      18:       /-- 0affffff   	beq	0x1c <test_func+0x1c>   @ imm = #-0x4
      1c:       \-> e320f000   	nop
      20:   /------ ea000003   	b	0x34 <test_func+0x34>   @ imm = #0xc
      24:   | /---- ea000001   	b	0x30 <test_func+0x30>   @ imm = #0x4
      28:   | | /-- eaffffff   	b	0x2c <test_func+0x2c>   @ imm = #-0x4
      2c:   | | \-> e320f000   	nop
      30:   | \---> e320f000   	nop
      34:   \-----> e320f000   	nop
      38:       /-- ea000000   	b	0x40 <test_func+0x40>   @ imm = #0x0
      3c:     /-|-- ea000000   	b	0x44 <test_func+0x44>   @ imm = #0x0
      40:     | \-> e320f000   	nop
      44:     \---> e320f000   	nop
      48:     /---- eaffffff   	b	0x4c <test_func+0x4c>   @ imm = #-0x4
      4c:     \-|-> eaffffff   	b	0x50 <test_func+0x50>   @ imm = #-0x4
      50:       \-> e320f000   	nop
