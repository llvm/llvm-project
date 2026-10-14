## Verify that --disassemble-symbols still honours per-region ISA mapping
## symbols when disassembling only a subset of a section.  When the chosen
## symbol lies inside a V-enabled region, the V encoding there must decode,
## and when another symbol lies outside the V region, the same encoding
## must come out as <unknown>.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o

# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn \
# RUN:   --disassemble-symbols=in_v %t.o | FileCheck %s --check-prefix=INV
# INV-LABEL: <in_v>:
# INV-NEXT:  vadd.vv	v0, v1, v2

# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn \
# RUN:   --disassemble-symbols=outside_v %t.o \
# RUN:   | FileCheck %s --check-prefix=OUT
# OUT-LABEL: <outside_v>:
# OUT-NEXT:  <unknown>

.text
.option push
.option arch, +v
.globl in_v
in_v:
vadd.vv v0, v1, v2
.option pop

.globl outside_v
outside_v:
.insn 4, 0x02110057
