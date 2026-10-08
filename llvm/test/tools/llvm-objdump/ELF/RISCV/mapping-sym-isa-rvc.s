## Verify that ISA mapping symbols emitted on .option rvc/.option norvc let
## llvm-objdump switch the compressed-instruction decoder on and off per
## region.  Starting from a base rv64i / rv32i STI (no C), the middle region
## is the only one where a compressed encoding should decode; outside the
## RVC window the 4-byte decoder is in effect.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.64.o
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn %t.64.o \
# RUN:   | FileCheck %s

# RUN: llvm-mc -triple=riscv32 -filetype=obj %s -o %t.32.o
# RUN: llvm-objdump -d -M no-aliases --no-show-raw-insn %t.32.o \
# RUN:   | FileCheck %s

.text
## Base (no C): 4-byte nop.
nop
# CHECK:      0:      	addi	zero, zero, 0x0

## Region 1: C enabled.  Both an explicit c.nop and a plain nop (which the
## assembler will relax to c.nop here) decode as compressed, so the per-
## region decoder clearly has C.
.option rvc
c.nop
# CHECK-NEXT: 4:      	c.nop
nop
# CHECK-NEXT: 6:      	c.nop

## Region 2: C disabled again; 4-byte nop round-trips as the full 32-bit
## encoding, confirming the mapping symbol switched the decoder back.
.option norvc
nop
# CHECK-NEXT: 8:      	addi	zero, zero, 0x0
