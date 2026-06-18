## Test that llvm-objdump's --mattr is applied on top of the ISA mapping
## symbol when it adds extensions not present in the mapping symbol.
##
## The object is assembled with no knowledge of Xqci, so the mapping symbol
## reflects only the base rv32i.  Passing --mattr=+xqcilia to llvm-objdump
## layers that extension on top of the mapping symbol and lets the
## corresponding raw-byte instruction decode as qc.e.addai instead of
## <unknown>.

# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.o
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+xqcilia %t.o \
# RUN:     | FileCheck %s
## Without --mattr=+xqcilia, the mapping symbol's base rv32i is used as-is,
## and the Xqcilia-encoded instruction decodes as <unknown>.
# RUN: llvm-objdump -d --no-show-raw-insn %t.o \
# RUN:     | FileCheck %s --check-prefix=CHECK-NO-MATTR

.text

## qc.e.addai a0, 0xff00ff (Xqcilia, 6-byte encoding).
## Encoding from llvm/test/MC/RISCV/insn_xqci.s.
.insn 6, 0x00ff00ff251f
# CHECK: qc.e.addai a0, 0xff00ff
# CHECK-NO-MATTR: <unknown>
