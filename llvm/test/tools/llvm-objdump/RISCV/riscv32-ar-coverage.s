# RUN: llvm-mc -riscv-add-build-attributes -triple=riscv32 -filetype=obj -mattr=+zclsd,+zilsd %s -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK: 00000000 <_start>:
# CHECK-NEXT        0: 00000517     	auipc	a0, 0x0
# CHECK-NEXT        4: 0559         	addi	a0, a0, 0x16 <target>
# CHECK-NEXT        6: 00000517     	auipc	a0, 0x0
# CHECK-NEXT        a: 6910         	ld	a2, 0x10(a0) <target>
# CHECK-NEXT        c: 00000517     	auipc	a0, 0x0
# CHECK-NEXT       10: 00c53523     	sd	a2, 0xa(a0) <target>
# CHECK-NEXT       14: 0000         	unimp

## The structure of this test file is similar to that of riscv64-ar-coverage
## with the major difference being that these tests are focused on instructions
## for 32 bit architecture.

.global _start
.text
_start:
  auipc a0, 0x0
  addi a0, a0, 0x16   ## addi -- behavior changes with different architectures.

  ## Test Zclsd and Zilsd instructions respectively
  auipc a0, 0x0
  c.ld a2, 0x10(a0)

  auipc a0, 0x0
  sd a2, 0xa(a0)

.skip 0x2
target:
  ret:
