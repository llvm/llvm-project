# RUN: llvm-objdump -d %p/Inputs/riscv-ar-coverage | FileCheck %s

# CHECK: 0000000000001000 <_start>:
# CHECK-NEXT:     1000: 00001517     	auipc	a0, 0x1
# CHECK-NEXT:     1004: 00450513     	addi	a0, a0, 0x4 <target>
# CHECK-NEXT:     1008: 00001517     	auipc	a0, 0x1
# CHECK-NEXT:     100c: 1571         	addi	a0, a0, -0x4 <target>
# CHECK-NEXT:     100e: 6509         	lui	a0, 0x2
# CHECK-NEXT:     1010: 0045059b     	addiw	a1, a0, 0x4 <target>
# CHECK-NEXT:     1014: 6509         	lui	a0, 0x2
# CHECK-NEXT:     1016: 2511         	addiw	a0, a0, 0x4 <target>
# CHECK-NEXT:     1018: 00102537     	lui	a0, 0x102
# CHECK-NEXT:     101c: c50c         	sw	a1, 0x8(a0) <far_target>
# CHECK-NEXT:     101e: 00102537     	lui	a0, 0x102
# CHECK-NEXT:     1022: 4508         	lw	a0, 0x8(a0) <far_target>
# CHECK-NEXT:     1024: 6509         	lui	a0, 0x2
# CHECK-NEXT:     1026: 6585         	lui	a1, 0x1
# CHECK-NEXT:     1028: 0306         	slli	t1, t1, 0x1
# CHECK-NEXT:     102a: 0511         	addi	a0, a0, 0x4 <target>
# CHECK-NEXT:     102c: 0505         	addi	a0, a0, 0x1
# CHECK-NEXT:     102e: 00200037     	lui	zero, 0x200
# CHECK-NEXT:     1032: 00a02423     	sw	a0, 0x8(zero)
# CHECK-NEXT:     1036: 00101097     	auipc	ra, 0x101
# CHECK-NEXT:     103a: fd6080e7     	jalr	-0x2a(ra) <func>

.global _start
.text

# The core of the feature being added was address resolution for instruction 
# sequences where an register is populated by immediate values via two
# separate instructions. First by an instruction that provides the upper bits
# (auipc, lui ...) followed by another instruction for the lower bits (addi,
# jalr, ld ...).


_start:
  # Test block 1-3 each focus on a certain starting instruction in a sequences, 
  # the ones that provide the upper bits. The other sequence is another
  # instruction the provides the lower bits. The second instruction is
  # arbitrarily chosen to increase code coverage

  # test block #1
  lla a0, target     # addi

  auipc a0, 0x1
  c.addi a0, -0x4    # c.addi

  # test block #2
  c.lui a0, 0x2
  addiw a1, a0, 0x4  # addiw

  c.lui a0, 0x2
  c.addiw a0, 0x4    # c.addiw

  # test block #3
  lui a0, 0x102
  sw a1, 0x8(a0)     # sw

  lui a0, 0x102
  c.lw a0, 0x8(a0)   # lw

  # Test block 4 tests instruction interleaving, essentially the code's
  # ability to keep track of a valid sequence even if multiple other unrelated
  # instructions separate the two

  # test #4
  lui a0, 0x2
  lui a1, 0x1        # unrelated instruction
  slli t1, t1, 0x1   # unrelated instruction
  addi a0, a0, 0x4
  addi a0, a0, 0x1

  # Test 5 ensures that an instruction writing into the zero register does
  # not trigger resolution because that register's value cannot change and
  # the sequence is equivalent to never running the first instruction

  # test #5
  lui x0, 0x200
  sw a0, 0x8(x0)

  # Test 6 ensures that the newly added functionality is compatible with
  # code that already worked for branch instructions

  # test #6
  call func

# these are the labels that the instructions above are expecteed to resolve to
.section .data
.skip 0x4
target:
  .word 1
.skip 0x100000
far_target:
  .word 2
func:
  ret