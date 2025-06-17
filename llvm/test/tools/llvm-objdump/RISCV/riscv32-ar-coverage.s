# RUN: llvm-objdump -d %p/Inputs/riscv32-ar-coverage | FileCheck %s

# CHECK: 00001000 <_start>:
# CHECK-NEXT:     1000: 00000517     	auipc	a0, 0x0
# CHECK-NEXT:     1004: 0559         	addi	a0, a0, 0x16 <target>
# CHECK-NEXT:     1006: 00000517     	auipc	a0, 0x0
# CHECK-NEXT:     100a: 6910         	ld	a2, 0x10(a0) <target>
# CHECK-NEXT:     100c: 00000517     	auipc	a0, 0x0
# CHECK-NEXT:     1010: 00c53523     	sd	a2, 0xa(a0) <target>
# CHECK-NEXT:     1014: 0000         	unimp

# the structure of this test file is similar to that of riscv64-ar-coverage
# with the major difference being that these tests are focused on instructions
# for 32 bit architecture

.global _start
.text
_start:
  auipc a0, 0x0
  addi a0, a0, 0x16   # addi -- behavior changes with differentr architectures

  auipc a0, 0x0
  c.ld a2, 0x10(a0)   # zclsd instruction

  auipc a0, 0x0
  sd a2, 0xa(a0)      # zilsd instruction

.skip 0x2
target:
  ret:
