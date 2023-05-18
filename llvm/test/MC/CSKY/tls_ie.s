# RUN: llvm-mc -filetype=obj -triple=csky -mattr=+2e3 < %s \
# RUN:     | llvm-objdump --mattr=+2e3 --no-show-raw-insn -M no-aliases -d -r - | FileCheck %s

.LTLS0:
	lrw16 r3, xxx@GOTTPOFF
	grs32 r2, .LTLS0
	addu16 r3, r3, r2
	ld16.w r3, (r3, 0)
        lsli32  r2, r31, 0
	str32.w r0, (r2, r3 << 0)


# CHECK:            0:      	lrw16	r3, 0x14
# CHECK-NEXT:       2:      	grs32	r2, 0x0
# CHECK-NEXT:       6:      	addu16	r3, r3, r2
# CHECK-NEXT:       8:      	ld16.w	r3, (r3, 0x0)
# CHECK-NEXT:       a:      	lsli32	r2, r31, 0
# CHECK-NEXT:       e:      	str32.w	r0, (r2, r3 << 0)
# CHECK-NEXT:       12:      	bkpt

# CHECK:           14:	00 00 00 00	.word	0x00000000
# CHECK-NEXT:           00000014:  R_CKCORE_TLS_IE32	xxx+0x14
