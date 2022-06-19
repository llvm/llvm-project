# RUN: llvm-mc -filetype=obj -triple=csky  < %s \
# RUN:     | llvm-objdump --mattr=+e1 --no-show-raw-insn -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-CK801 %s
# RUN: llvm-mc -filetype=obj -triple=csky -mcpu=ck802 < %s \
# RUN:     | llvm-objdump --mattr=+e2 --no-show-raw-insn -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-CK802 %s

# CHECK-OBJ-CK801:            0:      	addu16	r1, r2, r3
# CHECK-OBJ-CK801-NEXT:       2:      	bt16	0x8
# CHECK-OBJ-CK801-NEXT:       4:      	br32	0x4bc
# CHECK-OBJ-CK801-NEXT:       8:      	addu16	r3, r2, r1
# CHECK-OBJ-CK801:          4ba:      	addu16	r1, r2, r3
# CHECK-OBJ-CK801-NEXT:     4bc:      	bf16	0x4c2
# CHECK-OBJ-CK801-NEXT:     4be:      	br32	0x2
# CHECK-OBJ-CK801-NEXT:     4c2:      	br32	0x2
# CHECK-OBJ-CK801-NEXT:     4c6:      	addu16	r3, r2, r1
# CHECK-OBJ-CK801-NEXT:     4c8:      	br16	0x4ca

# CHECK-OBJ-CK802:            0:      	addu16	r1, r2, r3
# CHECK-OBJ-CK802-NEXT:       2:      	bf32	0x96a
# CHECK-OBJ-CK802-NEXT:       6:      	addu16	r3, r2, r1
# CHECK-OBJ-CK802:          968:      	addu16	r1, r2, r3
# CHECK-OBJ-CK802-NEXT:     96a:      	bt32	0x2
# CHECK-OBJ-CK802-NEXT:     96e:      	br32	0x2
# CHECK-OBJ-CK802-NEXT:     972:      	addu16	r3, r2, r1
# CHECK-OBJ-CK802-NEXT:     974:      	br16	0x976

	addu16 r1, r2, r3
.L1:
	jbf .L2
	addu16 r3, r2, r1

	.rept 600
	nop
	.endr


	addu16 r1, r2, r3
.L2:
	jbt .L1
	jbr .L1
	addu16 r3, r2, r1
	jbr .L3
.L3:
