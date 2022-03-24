# RUN: llvm-mc -filetype=obj -triple=csky -mattr=+2e3  < %s \
# RUN:     | llvm-objdump --mattr=+2e3 --no-show-raw-insn -M no-aliases -d -r - | FileCheck %s

	lrw16 r3, xxx@TPOFF
        lsli32    r2, r31, 0
	str32.w r0, (r2, r3 << 0)

# CHECK:            0:      	lrw16	r3, 0xc <$d.0>
# CHECK-NEXT:       2:      	lsli32    r2, r31, 0
# CHECK-NEXT:       6:      	str32.w	r0, (r2, r3 << 0)
# CHECK-NEXT:       a:      	bkpt

# CHECK:            c:	00 00 00 00	.word	0x00000000
# CHECK-NEXT:           0000000c:  R_CKCORE_TLS_LE32	xxx
