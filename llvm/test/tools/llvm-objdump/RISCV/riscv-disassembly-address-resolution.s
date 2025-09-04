# RUN: llvm-mc -riscv-add-build-attributes -triple=riscv64 -filetype=obj -mattr=+d,+c,+zcb %s -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK: 0000000000000000 <_start>:
# CHECK-NEXT:        0: 00010517     	auipc	a0, 0x10
# CHECK-NEXT:        4: 01450513     	addi	a0, a0, 0x14 <target>
# CHECK-NEXT:        8: 00010517     	auipc	a0, 0x10
# CHECK-NEXT:        c: 0531         	addi	a0, a0, 0xc <target>
# CHECK-NEXT:        e: 6541         	lui	a0, 0x10
# CHECK-NEXT:       10: 0145059b     	addiw	a1, a0, 0x14 <target>
# CHECK-NEXT:       14: 6541         	lui	a0, 0x10
# CHECK-NEXT:       16: 2551         	addiw	a0, a0, 0x14 <target>
# CHECK-NEXT:       18: 00110537     	lui	a0, 0x110
# CHECK-NEXT:       1c: c90c         	sw	a1, 0x10(a0) <far_target>
# CHECK-NEXT:       1e: 00110537     	lui	a0, 0x110
# CHECK-NEXT:       22: 4908         	lw	a0, 0x10(a0) <far_target>
# CHECK-NEXT:       24: 6541         	lui	a0, 0x10
# CHECK-NEXT:       26: 6585         	lui	a1, 0x1
# CHECK-NEXT:       28: 0306         	slli	t1, t1, 0x1
# CHECK-NEXT:       2a: 0551         	addi	a0, a0, 0x14 <target>
# CHECK-NEXT:       2c: 0505         	addi	a0, a0, 0x1
# CHECK-NEXT:       2e: 00002427     	fsw	ft0, 0x8(zero) <_start+0x8>
# CHECK-NEXT:       32: 00100017     	auipc	zero, 0x100
# CHECK-NEXT:       36: 00002427     	fsw	ft0, 0x8(zero) <_start+0x8>
# CHECK-NEXT:       3a: 00110097     	auipc	ra, 0x110
# CHECK-NEXT:       3e: fda080e7     	jalr	-0x26(ra) <func>
# CHECK-NEXT:       42: 01000517     	auipc	a0, 0x1000
# CHECK-NEXT:       46: 00110517     	auipc	a0, 0x110
# CHECK-NEXT:       4a: fca50513     	addi	a0, a0, -0x36 <far_target>


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
  c.addi a0, 0xc

  ## Test block #2.
  c.lui a0, 0x10
  addiw a1, a0, 0x14
  c.lui a0, 0x10
  c.addiw a0, 0x14

  ## Test block #3.
  lui a0, 0x110
  sw a1, 0x10(a0)
  lui a0, 0x110
  c.lw a0, 0x10(a0)

  ## Test block 4 tests instruction interleaving. Essentially the code's
  ## ability to keep track of a valid sequence even if multiple other unrelated
  ## instructions separate the two. In effect, the resolution must occur
  ## alongside the instruction marked below with the upper bits provided by the
  ## first instruction in the test. The instructions marked to be unrelated
  ## operate on unrelated registers and should not affect the instruction
  ## sequence formed around them. The last instruction in the test operates on the same
  ## register as the sequence but should NOT have an address resolution since
  ## the sequence terminated in the previous instruction.
  lui a0, 0x10       ## Part of sequence. Provides upper bits
  lui a1, 0x1        ## Unrelated instruction.
  slli t1, t1, 0x1   ## Unrelated instruction.
  addi a0, a0, 0x14   ## End of sequence. Provides lower bits. Resolution here
  addi a0, a0, 0x1   ## Verify register tracking terminates. NO resolution here

  ## Test 5 checks that address resolution works for instructions that make
  ## sense to have address resolution occur without an instruction providing
  ## the upper bits. Such instructions include load/stores relative to the
  ## zero register and short jumps pc-relative jumps
  fsw f0, 0x8(x0)

  ## Test 6 checks instructions providing upper bits do not change the tracked
  ## value of zero register.
  auipc x0, 0x100
  fsw f0, 0x8(x0)

  ## Test 7 ensures that the newly added functionality is compatible with
  ## code that already worked for branch instructions.
  call func

  ## Test 8 checks that subsequent upper bits operations on the same register
  ## correctly update the tracked register value to the value written by the
  ## latest instruction. Resolution must occur based on the update upper bit
  ## value.
  auipc a0, 0x1000     ## Initial  upper bit value
  lla a0, far_target   ## Pseudo instruction provides AUIPC. Resolution occurs
                       ## based on value written by this instruction

## These are the labels that the instructions above are expected to resolve to.
.skip 0xffc6
target:
  .word 1
.skip 0xffff8
far_target:
  .word 2
func:
  ret
