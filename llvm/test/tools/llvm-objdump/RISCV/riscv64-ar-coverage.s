# RUN: llvm-mc -riscv-add-build-attributes -triple=riscv64 -filetype=obj -mattr=+f,+c,+zcb %s -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK: 0000000000000000 <_start>:
# CHECK-NEXT:        0: 00010517     	auipc	a0, 0x10
# CHECK-NEXT:        4: 00450513     	addi	a0, a0, 0x4 <target>
# CHECK-NEXT:        8: 00010517     	auipc	a0, 0x10
# CHECK-NEXT:        c: 1571         	addi	a0, a0, -0x4 <target>
# CHECK-NEXT:        e: 6541         	lui	a0, 0x10
# CHECK-NEXT:       10: 0045059b     	addiw	a1, a0, 0x4 <target>
# CHECK-NEXT:       14: 6541         	lui	a0, 0x10
# CHECK-NEXT:       16: 2511         	addiw	a0, a0, 0x4 <target>
# CHECK-NEXT:       18: 00110537     	lui	a0, 0x110
# CHECK-NEXT:       1c: c50c         	sw	a1, 0x8(a0) <far_target>
# CHECK-NEXT:       1e: 00110537     	lui	a0, 0x110
# CHECK-NEXT:       22: 4508         	lw	a0, 0x8(a0) <far_target>
# CHECK-NEXT:       24: 6541         	lui	a0, 0x10
# CHECK-NEXT:       26: 6585         	lui	a1, 0x1
# CHECK-NEXT:       28: 0306         	slli	t1, t1, 0x1
# CHECK-NEXT:       2a: 0511         	addi	a0, a0, 0x4 <target>
# CHECK-NEXT:       2c: 0505         	addi	a0, a0, 0x1
# CHECK-NEXT:       2e: 00002427     	fsw	ft0, 0x8(zero) <_start+0x8>
# CHECK-NEXT:       32: 00110097     	auipc	ra, 0x110
# CHECK-NEXT:       36: fda080e7     	jalr	-0x26(ra) <func>
# CHECK-NEXT:       3a: 6445         	lui	s0, 0x11
# CHECK-NEXT:       3c: 8800         	sb	s0, 0x0(s0) <zcb>
# CHECK-NEXT:       3e: 4522         	lw	a0, 0x8(sp)

## The core of the feature being added was address resolution for instruction 
## sequences where a register is populated by immediate values via two
## separate instructions. First by an instruction that provides the upper bits
## (auipc, lui, etc) followed by another instruction for the lower bits (addi,
## jalr, ld, etc.).

.global _start
.text

_start:
  ## Test block 1-3 each focus on a certain starting instruction in a sequence. 
  ## Starting instructions are the ones that provide the upper bits. The other
  ## instruction in the sequence is the one that provides the lower bits. The
  ## second instruction is arbitrarily chosen to increase code coverage.

  ## Test block #1.
  lla a0, target
  auipc a0, 0x10
  c.addi a0, -0x4

  ## Test block #2.
  c.lui a0, 0x10
  addiw a1, a0, 0x4
  c.lui a0, 0x10
  c.addiw a0, 0x4

  ## Test block #3.
  lui a0, 0x110
  sw a1, 0x8(a0)
  lui a0, 0x110
  c.lw a0, 0x8(a0)

  ## Test block 4 tests instruction interleaving. Essentially the code's
  ## ability to keep track of a valid sequence even if multiple other unrelated
  ## instructions separate the two.
  lui a0, 0x10
  lui a1, 0x1        ## Unrelated instruction.
  slli t1, t1, 0x1   ## Unrelated instruction.
  addi a0, a0, 0x4
  addi a0, a0, 0x1   ## Verify register tracking terminates.

  ## Test 5 checks instructions providing upper bits do not change the tracked
  ## value of zero register. Also ensures load/store instructions accessing data
  ## relative to the zero register trigger address resolution. The latter kind
  ## of instructions are essentially memory accesses relative to the zero
  ## register.
  fsw f0, 0x8(x0)

  ## Test 6 ensures that the newly added functionality is compatible with
  ## code that already worked for branch instructions.
  call func

  ## Test #7 -- zcb extension.
  lui x8, 0x11
  c.sb x8, 0(x8)

  ## Test #8 -- stack based load/stores.
  c.lwsp a0, 0x8(sp)

## These are the labels that the instructions above are expected to resolve to.
.skip 0xffc4
target:
  .word 1
.skip 0xff8
zcb:
  .word 1
.skip 0xff004
far_target:
  .word 2
func:
  ret
