## Verify that an absolute ".option arch, <full-arch>" replacement (not the
## incremental "+v" form) produces an ISA mapping symbol whose normalized
## arch string llvm-objdump can parse, and that the region-local decoder
## picks up the full ISA described there.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn %t.o | FileCheck %s

.text
nop
# CHECK:      0:      	addi	zero, zero, 0x0

## Absolute replacement: the emitted ISA string is the fully-normalized
## arch, so V (and its z-extensions / vlen markers) must be present in the
## per-region decoder even though the base STI has none of them.
.option push
.option arch, rv64gc_v
vadd.vv v0, v1, v2
# CHECK-NEXT: 4:      	vadd.vv	v0, v1, v2
.option pop

## Back to base ISA once .option pop restores the previous state.
nop
# CHECK-NEXT: 8:      	addi	zero, zero, 0x0
