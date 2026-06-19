## Test that llvm-objdump silently drops --mattr for a region when the
## command-line --mattr would conflict with the region's ISA mapping symbol.
##
## The .option arch, +f region emits an "$x<...>_f..." mapping symbol that
## advertises the F extension.  Passing --mattr=+zfinx on top is mutually
## exclusive with +f; RISCVISAInfo::parseFeatures rejects the combined set,
## so llvm-objdump drops --mattr for this region and decodes using the
## mapping symbol (i.e. as an F-extension fadd.s using FPR names).

# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.o
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+zfinx %t.o \
# RUN:     | FileCheck %s

.text

## fadd.s in a +f region.  The assembler emits an ISA mapping symbol for
## the +f arch at the start of this region.
.option push
.option arch, +f
fadd.s fa0, fa1, fa2
# CHECK: fadd.s fa0, fa1, fa2
.option pop
