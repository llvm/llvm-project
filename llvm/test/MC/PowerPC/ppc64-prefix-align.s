# RUN: llvm-mc -triple powerpc64-unknown-linux-gnu --filetype=obj -o - %s | \
# RUN:   llvm-objdump -D  -r - | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-linux-gnu --filetype=obj -o - %s | \
# RUN:   llvm-objdump -D  -r - | FileCheck -check-prefix=CHECK-LE %s

# The purpose of this test is to make sure that 8 byte instructions do not
# cross 64 byte boundaries. If an 8 byte instruction is about to cross such
# a boundary then a nop should be added so that the 8 byte instruction starts
# 4 bytes later and does not cross the boundary.
# This instruction is 8 bytes: paddi 1, 2, 8589934576, 0
# This instruction is 4 bytes: addi 2, 3, 15
# The branches are also 4 bytes each: beq 0, LAB1 (or LAB2)

beq 0, LAB1               # 4
beq 1, LAB2               # 8
# CHECK-BE:       0: 41 82 00 c0        bt 2, .+192
# CHECK-BE-NEXT:  4: 41 86 00 f8        bt 6, .+248
# CHECK-LE:       0: c0 00 82 41        bt 2, .+192
# CHECK-LE-NEXT:  4: f8 00 86 41        bt 6, .+248
paddi 1, 2, 8589934576, 0 # 16
paddi 1, 2, 8589934576, 0 # 24
paddi 1, 2, 8589934576, 0 # 32
paddi 1, 2, 8589934576, 0 # 40
paddi 1, 2, 8589934576, 0 # 48
paddi 1, 2, 8589934576, 0 # 56
addi 2, 3, 15             # 60
# Below the lines 40: and 44: contain the 8 byte instruction.
# We check to make sure that the nop is added at 3c: so that the 8 byte
# instruction can start at 40: which is 64 bytes aligned.
# CHECK-BE:      38:	38 43 00 0f
# CHECK-BE-NEXT: 3c:	60 00 00 00 	nop
# CHECK-BE-NEXT: 40:	06 01 ff ff
# CHECK-BE-NEXT: 44:	38 22 ff f0
# CHECK-LE:      38:	0f 00 43 38
# CHECK-LE-NEXT: 3c:	00 00 00 60 	nop
# CHECK-LE-NEXT: 40:	ff ff 01 06
# CHECK-LE-NEXT: 44:	f0 ff 22 38
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0 # 64
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
addi 2, 3, 15             # 60
# CHECK-BE:      b8:	38 43 00 0f
# CHECK-BE-NEXT: bc:	60 00 00 00 	nop
# CHECK-BE:      <LAB1>:
# CHECK-BE-NEXT: c0:	06 01 ff ff
# CHECK-BE-NEXT: c4:	38 22 ff f0
# CHECK-LE:      b8:	0f 00 43 38
# CHECK-LE-NEXT: bc:	00 00 00 60 	nop
# CHECK-LE:      <LAB1>:
# CHECK-LE-NEXT: c0:	ff ff 01 06
# CHECK-LE-NEXT: c4:	f0 ff 22 38
LAB1: paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
paddi 1, 2, 8589934576, 0
addi 2, 3, 15             # 60
# CHECK-BE:      f8:	38 43 00 0f
# CHECK-BE:      <LAB2>:
# CHECK-BE-NEXT: fc:	60 00 00 00 	nop
# CHECK-BE-NEXT: 100:	06 01 ff ff
# CHECK-BE-NEXT: 104:	38 22 ff f0
# CHECK-LE:      f8:	0f 00 43 38
# CHECK-LE:      <LAB2>:
# CHECK-LE-NEXT: fc:	00 00 00 60 	nop
# CHECK-LE-NEXT: 100:	ff ff 01 06
# CHECK-LE-NEXT: 104:	f0 ff 22 38
LAB2:
  paddi 1, 2, 8589934576, 0



