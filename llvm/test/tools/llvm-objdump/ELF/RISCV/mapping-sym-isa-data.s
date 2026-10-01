## Exercise the interaction between ISA mapping symbols ("$x<ISA>") and data
## mapping symbols ("$d") emitted by .word / .byte directives.  A $d must
## not disturb the active per-region decoder; once the data region ends and
## code resumes, the previous region's ISA decoder must remain in effect.
## Also covers the corner case where the assembler emits a bare "$x" (no
## ISA suffix) after data: that symbol is *not* an ISA mapping symbol and
## must not clobber the most recent $x<ISA> in the region.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn %t.o | FileCheck %s

.text
nop
# CHECK:      0:      	addi	zero, zero, 0x0

.option push
.option arch, +v
## First V instruction: per-region decoder must have V.
vadd.vv v0, v1, v2
# CHECK-NEXT: 4:      	vadd.vv	v0, v1, v2

## Data embedded inside the V region emits "$d" then a bare "$x" afterwards.
## llvm-objdump should dump the data as .word and *not* lose track of V when
## code resumes.
.word 0xdeadbeef
# CHECK-NEXT: 8:{{.*}}.word	0xdeadbeef

vadd.vv v3, v4, v5
# CHECK-NEXT: c:      	vadd.vv	v3, v4, v5
.option pop

nop
# CHECK-NEXT: 10:      	addi	zero, zero, 0x0
