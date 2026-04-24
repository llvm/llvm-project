## Test that llvm-objdump's --mattr is applied on top of the ISA mapping
## symbol.  The command-line --mattr wins over both Tag_RISCV_arch and the
## mapping-symbol ISA, so extensions the user passes are available to the
## decoder even when the mapping symbol does not list them.  If --mattr
## would conflict with the mapping symbol (e.g. +zfinx vs +f), the --mattr
## layer is silently dropped for that region and the mapping symbol alone
## is used.

## Case 1: --mattr adds extensions not present in the mapping symbol.
## The object is assembled with no knowledge of Xqci, so the mapping symbol
## reflects only the base rv32i.  Passing --mattr=+xqcilia to llvm-objdump
## layers that extension on top of the mapping symbol and lets the
## corresponding raw-byte instruction decode as qc.e.addai instead of
## <unknown>.
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.32.o
# RUN: llvm-objdump -d --no-show-raw-insn \
# RUN:     --mattr=+xqcilia %t.32.o \
# RUN:     | FileCheck --check-prefix=ADD %s

## Case 2: --mattr conflicts with the mapping symbol.
## The .option arch, +f region emits an "$x<...>_f..." mapping symbol that
## advertises the F extension.  Passing --mattr=+zfinx on top is mutually
## exclusive with +f; RISCVISAInfo::parseFeatures rejects the combined set,
## so llvm-objdump drops --mattr for this region and decodes using the
## mapping symbol (i.e. as an F-extension fadd.s using FPR names).
# RUN: llvm-objdump -d --no-show-raw-insn \
# RUN:     --mattr=+zfinx %t.32.o \
# RUN:     | FileCheck --check-prefix=CONFLICT %s

.text

## Case 1 payload: qc.e.addai a0, 0xff00ff (Xqcilia, 6-byte encoding).
## Encoding from llvm/test/MC/RISCV/insn_xqci.s.
.insn 6, 0x00ff00ff251f
# ADD: qc.e.addai a0, 0xff00ff

## Case 2 payload: fadd.s in a +f region.  The assembler emits an ISA
## mapping symbol for the +f arch at the start of this region.
.option push
.option arch, +f
fadd.s fa0, fa1, fa2
# CONFLICT: fadd.s fa0, fa1, fa2
.option pop
