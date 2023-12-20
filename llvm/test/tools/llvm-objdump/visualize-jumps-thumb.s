// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc < input.s -triple thumbv8a -filetype=obj | \
// RUN:   llvm-objdump --triple thumbv8a -d --visualize-jumps=unicode - | \
// RUN:   diff - expected-unicode.txt

// RUN: llvm-mc < input.s -triple thumbv8a -filetype=obj | \
// RUN:   llvm-objdump --triple thumbv8a -d --visualize-jumps=ascii - | \
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

  // Different branch instructions
  b.w .Llabel2
  beq .Llabel2
  bne.w .Llabel2
  cbz r0, .Llabel2
  it le
  ble .Llabel2
  nop
.Llabel2:
  nop
.Llabel2.1:
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
       0:           f7ff fffe  	bl	0x0 <test_func>         @ imm = #-0x4
       4:           f7ff bffe  	b.w	0x4 <test_func+0x4>     @ imm = #-0x4
       8:       ╭── e7ff       	b	0xa <test_func+0xa>     @ imm = #-0x2
       a:       ├─> bf00       	nop
       c:       ╰── e7fd       	b	0xa <test_func+0xa>     @ imm = #-0x6
       e:           e7fe       	b	0xe <test_func+0xe>     @ imm = #-0x4
      10:       ╭── f000 b807  	b.w	0x22 <test_func+0x22>   @ imm = #0xe
      14:       ├── d005       	beq	0x22 <test_func+0x22>   @ imm = #0xa
      16:       ├── f040 8004  	bne.w	0x22 <test_func+0x22>   @ imm = #0x8
      1a:       ├── b110       	cbz	r0, 0x22 <test_func+0x22> @ imm = #0x4
      1c:       │   bfd8       	it	le
      1e:       ├── e000       	ble	0x22 <test_func+0x22>   @ imm = #0x0
      20:       │   bf00       	nop
      22:       ╰─> bf00       	nop
      24:           bf00       	nop
      26:   ╭────── e003       	b	0x30 <test_func+0x30>   @ imm = #0x6
      28:   │ ╭──── e001       	b	0x2e <test_func+0x2e>   @ imm = #0x2
      2a:   │ │ ╭── e7ff       	b	0x2c <test_func+0x2c>   @ imm = #-0x2
      2c:   │ │ ╰─> bf00       	nop
      2e:   │ ╰───> bf00       	nop
      30:   ╰─────> bf00       	nop
      32:       ╭── e000       	b	0x36 <test_func+0x36>   @ imm = #0x0
      34:     ╭─│── e000       	b	0x38 <test_func+0x38>   @ imm = #0x0
      36:     │ ╰─> bf00       	nop
      38:     ╰───> bf00       	nop
      3a:       ╭── e7ff       	b	0x3c <test_func+0x3c>   @ imm = #-0x2
      3c:     ╭─│── e7ff       	b	0x3e <test_func+0x3e>   @ imm = #-0x2
      3e:     ╰───> bf00       	nop
//--- expected-ascii.txt

<stdin>:	file format elf32-littlearm

Disassembly of section .text:

00000000 <test_func>:
       0:           f7ff fffe  	bl	0x0 <test_func>         @ imm = #-0x4
       4:           f7ff bffe  	b.w	0x4 <test_func+0x4>     @ imm = #-0x4
       8:       /-- e7ff       	b	0xa <test_func+0xa>     @ imm = #-0x2
       a:       +-> bf00       	nop
       c:       \-- e7fd       	b	0xa <test_func+0xa>     @ imm = #-0x6
       e:           e7fe       	b	0xe <test_func+0xe>     @ imm = #-0x4
      10:       /-- f000 b807  	b.w	0x22 <test_func+0x22>   @ imm = #0xe
      14:       +-- d005       	beq	0x22 <test_func+0x22>   @ imm = #0xa
      16:       +-- f040 8004  	bne.w	0x22 <test_func+0x22>   @ imm = #0x8
      1a:       +-- b110       	cbz	r0, 0x22 <test_func+0x22> @ imm = #0x4
      1c:       |   bfd8       	it	le
      1e:       +-- e000       	ble	0x22 <test_func+0x22>   @ imm = #0x0
      20:       |   bf00       	nop
      22:       \-> bf00       	nop
      24:           bf00       	nop
      26:   /------ e003       	b	0x30 <test_func+0x30>   @ imm = #0x6
      28:   | /---- e001       	b	0x2e <test_func+0x2e>   @ imm = #0x2
      2a:   | | /-- e7ff       	b	0x2c <test_func+0x2c>   @ imm = #-0x2
      2c:   | | \-> bf00       	nop
      2e:   | \---> bf00       	nop
      30:   \-----> bf00       	nop
      32:       /-- e000       	b	0x36 <test_func+0x36>   @ imm = #0x0
      34:     /-|-- e000       	b	0x38 <test_func+0x38>   @ imm = #0x0
      36:     | \-> bf00       	nop
      38:     \---> bf00       	nop
      3a:       /-- e7ff       	b	0x3c <test_func+0x3c>   @ imm = #-0x2
      3c:     /-|-- e7ff       	b	0x3e <test_func+0x3e>   @ imm = #-0x2
      3e:     \---> bf00       	nop
