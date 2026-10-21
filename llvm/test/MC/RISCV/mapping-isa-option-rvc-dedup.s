## Test deduplication of ISA mapping symbols for .option rvc when the C
## extension is already active.  The assembler must emit only the initial
## "$x<ISAString>" symbol because neither .option rvc nor the subsequent
## .option arch, +c change the active ISA.

# RUN: llvm-mc -triple=riscv32 -mattr=+c -filetype=obj -o %t32.o %s
# RUN: llvm-readobj --symbols %t32.o | FileCheck %s --check-prefix=RV32
# RUN: llvm-mc -triple=riscv64 -mattr=+c -filetype=obj -o %t64.o %s
# RUN: llvm-readobj --symbols %t64.o | FileCheck %s --check-prefix=RV64

.text
nop
# Initial mapping symbol records the base ISA (C already enabled via -mattr).
# RV32: Name: $xrv32i2p1_c2p0_zca1p0
# RV32-NEXT: Value: 0x0
# RV64: Name: $xrv64i2p1_c2p0_zca1p0
# RV64-NEXT: Value: 0x0

.option rvc
nop

.option arch, +c
nop

# No additional "$x..." mapping symbol should be emitted after the initial one,
# because neither .option rvc nor .option arch, +c change the active ISA.
# RV32-NOT: Name: $x
# RV64-NOT: Name: $x
