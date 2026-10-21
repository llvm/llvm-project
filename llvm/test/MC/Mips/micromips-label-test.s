# RUN: llvm-mc %s -triple=mipsel -mcpu=mips32r2 -mattr=+micromips -filetype=obj -o %t.o
# RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=CHECK
# RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=INITIAL
# RUN: llvm-readelf -s %t.o | FileCheck %s --check-prefix=MODE
# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+micromips -filetype=obj -o %t.o
# RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=CHECK
# RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=INITIAL
# RUN: llvm-readelf -s %t.o | FileCheck %s --check-prefix=MODE

## Command-line microMIPS mode must mark code symbols even without a
## .set micromips directive. Data labels must remain unmarked.
.globl function, code, data, initial_normal
.type function,@function
function:
  nop
code:
  nop
data:
  .word 0
.set nomicromips
.type initial_normal,@function
initial_normal:
  nop

# INITIAL: Name: function
# INITIAL: Type: Function
# INITIAL: Other [ (0x80)
# INITIAL-NEXT: STO_MIPS_MICROMIPS
# INITIAL: Name: code
# INITIAL: Type: None
# INITIAL: Other [ (0x80)
# INITIAL-NEXT: STO_MIPS_MICROMIPS
# INITIAL: Name: data
# INITIAL: Other: 0
# INITIAL: Name: initial_normal
# INITIAL: Type: Function
# INITIAL: Other: 0

  .text
  .set nomicromips
f:
  nop
g:
  .set micromips
  nop
h:
  .word 0
k:
  .long 0
l:
  .hword 0
m:
  .2byte 0
n:
  .4byte 0
o:
  .8byte 0
i:
  nop
j:
  .set nomicromips
  nop
# CHECK: Symbols [
# CHECK:   Symbol {
# CHECK:     Name: f
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: g
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other [ (0x80)
# CHECK:       STO_MIPS_MICROMIPS
# CHECK:     ]
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: h
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: k
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: l
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: m
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: n
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:   Symbol {
# CHECK:     Name: o
# CHECK:     Binding: Local
# CHECK:     Type: None
# CHECK:     Other: 0
# CHECK:     Section: .text
# CHECK:   }
# CHECK:        Symbol {
# CHECK-NEXT:     Name: i
# CHECK:          Binding: Local
# CHECK-NEXT:     Type: None
# CHECK-NEXT:     Other [ (0x80)
# CHECK-NEXT:       STO_MIPS_MICROMIPS
# CHECK-NEXT:     ]
# CHECK-NEXT:     Section: .text
# CHECK-NEXT:   }
# CHECK-NEXT:   Symbol {
# CHECK-NEXT:     Name: j
# CHECK:          Binding: Local
# CHECK-NEXT:     Type: None
# CHECK-NEXT:     Other: 0
# CHECK-NEXT:     Section: .text
# CHECK-NEXT:   }
# CHECK: ]

.set micromips
.set noreorder
.text
.globl initial, normal, nested, restored_normal, restored_micro, reset
initial:
 nop
.set push
.set nomicromips
normal:
 nop
.set push
.set micromips
nested:
 nop
.set pop
restored_normal:
 nop
.set pop
restored_micro:
 nop
.set nomicromips
.set mips0
reset:
 nop

# MODE-DAG: NOTYPE GLOBAL DEFAULT [<other: 0x80>] {{[0-9]+}} initial
# MODE-DAG: NOTYPE GLOBAL DEFAULT {{[0-9]+}} normal
# MODE-DAG: NOTYPE GLOBAL DEFAULT [<other: 0x80>] {{[0-9]+}} nested
# MODE-DAG: NOTYPE GLOBAL DEFAULT {{[0-9]+}} restored_normal
# MODE-DAG: NOTYPE GLOBAL DEFAULT [<other: 0x80>] {{[0-9]+}} restored_micro
# MODE-DAG: NOTYPE GLOBAL DEFAULT [<other: 0x80>] {{[0-9]+}} reset

# Data in an executable section must not inherit the next instruction's mode.
.globl bytes, space, fill, word, after_data, before_section, in_section
bytes:
 .ascii "data"
 nop
space:
 .space 4
 nop
fill:
 .fill 2, 2, 0
 nop
word:
 .word 0
 nop
after_data:
 nop
before_section:
 .pushsection .rodata
 .word 0
 .popsection
in_section:
 nop

# MODE-DAG: NOTYPE GLOBAL DEFAULT {{[0-9]+}} bytes
# MODE-DAG: NOTYPE GLOBAL DEFAULT {{[0-9]+}} space
# MODE-DAG: NOTYPE GLOBAL DEFAULT {{[0-9]+}} fill
# MODE-DAG: NOTYPE GLOBAL DEFAULT {{[0-9]+}} word
# MODE-DAG: NOTYPE GLOBAL DEFAULT [<other: 0x80>] {{[0-9]+}} after_data
# MODE-DAG: NOTYPE GLOBAL DEFAULT {{[0-9]+}} before_section
# MODE-DAG: NOTYPE GLOBAL DEFAULT [<other: 0x80>] {{[0-9]+}} in_section

# Aliases inherit the target's ISA flags, independently of the current mode.
# Visibility is not part of getOther() and must not be copied to the alias.
.set nomicromips
.globl standard_alias, micro_alias, mips16_function, mips16_alias
.hidden mips16_function
micro_alias = function
.set micromips
standard_alias = initial_normal
.set mips16
.type mips16_function,@function
mips16_function:
.set nomips16
mips16_alias = mips16_function

# MODE-DAG: FUNC GLOBAL DEFAULT {{[0-9]+}} standard_alias
# MODE-DAG: FUNC GLOBAL DEFAULT [<other: 0x80>] {{[0-9]+}} micro_alias
# MODE-DAG: FUNC GLOBAL DEFAULT [<other: 0xf0>] {{[0-9]+}} mips16_alias
