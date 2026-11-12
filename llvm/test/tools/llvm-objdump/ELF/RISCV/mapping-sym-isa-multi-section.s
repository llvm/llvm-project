## Two text sections each carry their own independent ISA mapping symbols.
## Verifies that the per-section AllRISCVISAMappingSymbols map is keyed
## correctly and one section's ISA regions do not bleed into the other.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn %t.o | FileCheck %s

# CHECK-LABEL: Disassembly of section .text.a:
.section .text.a, "ax"
.option push
.option arch, +v
vadd.vv v0, v1, v2
# CHECK: 0:      	vadd.vv	v0, v1, v2
.option pop
nop
# CHECK-NEXT: 4:      	addi	zero, zero, 0x0

# CHECK-LABEL: Disassembly of section .text.b:
.section .text.b, "ax"
## Start in base ISA; V should *not* leak in from .text.a.
.insn 4, 0x02110057
# CHECK: 0:      	<unknown>

.option push
.option arch, +v
vadd.vv v8, v9, v10
# CHECK-NEXT: 4:      	vadd.vv	v8, v9, v10
.option pop
