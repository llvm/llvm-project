# RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc < input.s -triple aarch64 -filetype=obj | \
// RUN:   llvm-objdump --triple aarch64 -d --visualize-jumps=unicode - | \
// RUN:   diff - expected-unicode.txt

// RUN: llvm-mc < input.s -triple aarch64 -filetype=obj | \
// RUN:   llvm-objdump --triple aarch64 -d --visualize-jumps=ascii - | \
// RUN:   diff - expected-ascii.txt

// RUN: llvm-mc < input.s -triple aarch64 -filetype=obj | \
// RUN:   llvm-objdump --triple aarch64 -d --visualize-jumps=unicode,color - | \
// RUN:   diff - expected-unicode-color.txt

// RUN: llvm-mc < input.s -triple aarch64 -filetype=obj | \
// RUN:   llvm-objdump --triple aarch64 -d --visualize-jumps=unicode --reloc - | \
// RUN:   diff - expected-unicode-relocs.txt

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
  b.eq .Llabel2
  cbz x0, .Llabel2
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

  // Graph lines need to be drawn on the same output line as relocations.
  b .Llabel10
  bl extern_func
.Llabel10:
  nop

//--- expected-unicode.txt

<stdin>:	file format elf64-littleaarch64

Disassembly of section .text:

0000000000000000 <test_func>:
       0:           94000000   	bl	0x0 <test_func>
       4:           14000000   	b	0x4 <test_func+0x4>
       8:       ╭── 14000001   	b	0xc <test_func+0xc>
       c:       ├─> d503201f   	nop
      10:       ╰── 17ffffff   	b	0xc <test_func+0xc>
      14:           14000000   	b	0x14 <test_func+0x14>
      18:       ╭── 54000040   	b.eq	0x20 <test_func+0x20>
      1c:       ├── b4000020   	cbz	x0, 0x20 <test_func+0x20>
      20:       ╰─> d503201f   	nop
      24:   ╭────── 14000005   	b	0x38 <test_func+0x38>
      28:   │ ╭──── 14000003   	b	0x34 <test_func+0x34>
      2c:   │ │ ╭── 14000001   	b	0x30 <test_func+0x30>
      30:   │ │ ╰─> d503201f   	nop
      34:   │ ╰───> d503201f   	nop
      38:   ╰─────> d503201f   	nop
      3c:       ╭── 14000002   	b	0x44 <test_func+0x44>
      40:     ╭─│── 14000002   	b	0x48 <test_func+0x48>
      44:     │ ╰─> d503201f   	nop
      48:     ╰───> d503201f   	nop
      4c:       ╭── 14000001   	b	0x50 <test_func+0x50>
      50:     ╭─│── 14000001   	b	0x54 <test_func+0x54>
      54:     ╰───> d503201f   	nop
      58:       ╭── 14000002   	b	0x60 <test_func+0x60>
      5c:       │   94000000   	bl	0x5c <test_func+0x5c>
      60:       ╰─> d503201f   	nop
//--- expected-ascii.txt

<stdin>:	file format elf64-littleaarch64

Disassembly of section .text:

0000000000000000 <test_func>:
       0:           94000000   	bl	0x0 <test_func>
       4:           14000000   	b	0x4 <test_func+0x4>
       8:       /-- 14000001   	b	0xc <test_func+0xc>
       c:       +-> d503201f   	nop
      10:       \-- 17ffffff   	b	0xc <test_func+0xc>
      14:           14000000   	b	0x14 <test_func+0x14>
      18:       /-- 54000040   	b.eq	0x20 <test_func+0x20>
      1c:       +-- b4000020   	cbz	x0, 0x20 <test_func+0x20>
      20:       \-> d503201f   	nop
      24:   /------ 14000005   	b	0x38 <test_func+0x38>
      28:   | /---- 14000003   	b	0x34 <test_func+0x34>
      2c:   | | /-- 14000001   	b	0x30 <test_func+0x30>
      30:   | | \-> d503201f   	nop
      34:   | \---> d503201f   	nop
      38:   \-----> d503201f   	nop
      3c:       /-- 14000002   	b	0x44 <test_func+0x44>
      40:     /-|-- 14000002   	b	0x48 <test_func+0x48>
      44:     | \-> d503201f   	nop
      48:     \---> d503201f   	nop
      4c:       /-- 14000001   	b	0x50 <test_func+0x50>
      50:     /-|-- 14000001   	b	0x54 <test_func+0x54>
      54:     \---> d503201f   	nop
      58:       /-- 14000002   	b	0x60 <test_func+0x60>
      5c:       |   94000000   	bl	0x5c <test_func+0x5c>
      60:       \-> d503201f   	nop
//--- expected-unicode-color.txt

<stdin>:	file format elf64-littleaarch64

Disassembly of section .text:

0000000000000000 <test_func>:
       0:          [0m 94000000   	bl	0x0 <test_func>
       4:          [0m 14000000   	b	0x4 <test_func+0x4>
       8:      [0;31m ╭[0;31m──[0m 14000001   	b	0xc <test_func+0xc>
       c:      [0;31m ├[0;31m─>[0m d503201f   	nop
      10:      [0;31m ╰[0;31m──[0m 17ffffff   	b	0xc <test_func+0xc>
      14:          [0m 14000000   	b	0x14 <test_func+0x14>
      18:      [0;32m ╭[0;32m──[0m 54000040   	b.eq	0x20 <test_func+0x20>
      1c:      [0;32m ├[0;32m──[0m b4000020   	cbz	x0, 0x20 <test_func+0x20>
      20:      [0;32m ╰[0;32m─>[0m d503201f   	nop
      24:  [0;33m ╭[0;33m──[0;33m──[0;33m──[0m 14000005   	b	0x38 <test_func+0x38>
      28:  [0;33m │[0;34m ╭[0;34m──[0;34m──[0m 14000003   	b	0x34 <test_func+0x34>
      2c:  [0;33m │[0;34m │[0;35m ╭[0;35m──[0m 14000001   	b	0x30 <test_func+0x30>
      30:  [0;33m │[0;34m │[0;35m ╰[0;35m─>[0m d503201f   	nop
      34:  [0;33m │[0;34m ╰[0;34m──[0;34m─>[0m d503201f   	nop
      38:  [0;33m ╰[0;33m──[0;33m──[0;33m─>[0m d503201f   	nop
      3c:      [0;36m ╭[0;36m──[0m 14000002   	b	0x44 <test_func+0x44>
      40:    [0;31m ╭[0;31m─[0;36m│[0;31m──[0m 14000002   	b	0x48 <test_func+0x48>
      44:    [0;31m │[0;36m ╰[0;36m─>[0m d503201f   	nop
      48:    [0;31m ╰[0;31m──[0;31m─>[0m d503201f   	nop
      4c:      [0;32m ╭[0;32m──[0m 14000001   	b	0x50 <test_func+0x50>
      50:    [0;33m ╭[0;33m─[0;32m│[0;33m──[0m 14000001   	b	0x54 <test_func+0x54>
      54:    [0;33m ╰[0;33m──[0;33m─>[0m d503201f   	nop
      58:      [0;34m ╭[0;34m──[0m 14000002   	b	0x60 <test_func+0x60>
      5c:      [0;34m │  [0m 94000000   	bl	0x5c <test_func+0x5c>
      60:      [0;34m ╰[0;34m─>[0m d503201f   	nop
//--- expected-unicode-relocs.txt

<stdin>:	file format elf64-littleaarch64

Disassembly of section .text:

0000000000000000 <test_func>:
       0:           94000000   	bl	0x0 <test_func>
                   		0000000000000000:  R_AARCH64_CALL26	extern_func
       4:           14000000   	b	0x4 <test_func+0x4>
                   		0000000000000004:  R_AARCH64_JUMP26	extern_func
       8:       ╭── 14000001   	b	0xc <test_func+0xc>
       c:       ├─> d503201f   	nop
      10:       ╰── 17ffffff   	b	0xc <test_func+0xc>
      14:           14000000   	b	0x14 <test_func+0x14>
      18:       ╭── 54000040   	b.eq	0x20 <test_func+0x20>
      1c:       ├── b4000020   	cbz	x0, 0x20 <test_func+0x20>
      20:       ╰─> d503201f   	nop
      24:   ╭────── 14000005   	b	0x38 <test_func+0x38>
      28:   │ ╭──── 14000003   	b	0x34 <test_func+0x34>
      2c:   │ │ ╭── 14000001   	b	0x30 <test_func+0x30>
      30:   │ │ ╰─> d503201f   	nop
      34:   │ ╰───> d503201f   	nop
      38:   ╰─────> d503201f   	nop
      3c:       ╭── 14000002   	b	0x44 <test_func+0x44>
      40:     ╭─│── 14000002   	b	0x48 <test_func+0x48>
      44:     │ ╰─> d503201f   	nop
      48:     ╰───> d503201f   	nop
      4c:       ╭── 14000001   	b	0x50 <test_func+0x50>
      50:     ╭─│── 14000001   	b	0x54 <test_func+0x54>
      54:     ╰───> d503201f   	nop
      58:       ╭── 14000002   	b	0x60 <test_func+0x60>
      5c:       │   94000000   	bl	0x5c <test_func+0x5c>
                │  		000000000000005c:  R_AARCH64_CALL26	extern_func
      60:       ╰─> d503201f   	nop
