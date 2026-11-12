# RUN: llvm-mc -filetype=obj -triple=csky -mcpu=ck803 < %s \
# RUN:     | llvm-objdump --mattr=+2e3 --no-show-raw-insn -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-CK803 %s

# CHECK-OBJ-CK803:            0:      	addu16	r1, r2, r3
# CHECK-OBJ-CK803-NEXT:       2:      	bt16	0x8
# CHECK-OBJ-CK803-NEXT:       4:      	jmpi32  0x3a9a0
# CHECK-OBJ-CK803-NEXT:       8:        bsr32   0x3a990
# CHECK-OBJ-CK803-NEXT:       c:      	addu16	r3, r2, r1
# CHECK-OBJ-CK803:        3a98e:      	addu16	r1, r2, r3
# CHECK-OBJ-CK803-NEXT:   3a990:      	bf16	0x3a996
# CHECK-OBJ-CK803-NEXT:   3a992:      	jmpi32  0x3a9a4
# CHECK-OBJ-CK803-NEXT:   3a996:      	jmpi32	0x3a9a4
# CHECK-OBJ-CK803-NEXT:   3a99a:      	addu16	r3, r2, r1
# CHECK-OBJ-CK803-NEXT:   3a99c:      	br16	0x3a99e


        addu16 r1, r2, r3
.L1:
	jbf .L2
	jbsr .L2
	addu16 r3, r2, r1

	.rept 60000
	nop
	.endr


	addu16 r1, r2, r3
.L2:
	jbt .L1
	jbr .L1
	addu16 r3, r2, r1
	jbr .L3
.L3:
