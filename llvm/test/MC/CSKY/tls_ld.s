# RUN: llvm-mc -filetype=obj -triple=csky -mattr=+2e3  < %s \
# RUN:     | llvm-objdump --mattr=+2e3 --no-show-raw-insn -M no-aliases -d -r - | FileCheck %s

.LTLS0:
	lrw16 r0, xxx@TLSLDM32
	grs32 r2, .LTLS0
	addu16 r0, r0, r2
	lrw16 r3, __tls_get_addr@PLT
	ldr32.w r3, (rgb, r3 << 0)
	jsr16 r3
	lrw16 r3, xxx@TLSLDO32
	str32.w r4, (r0, r3 << 0)

# CHECK:            0:      	lrw16	r0, 0x18 <$d.0>
# CHECK-NEXT:       2:      	grs32	r2, 0x0
# CHECK-NEXT:       6:      	addu16	r0, r0, r2
# CHECK-NEXT:       8:      	lrw16	r3, 0x1c <$d.0+0x4>
# CHECK-NEXT:       a:      	ldr32.w	r3, (r28, r3 << 0)
# CHECK-NEXT:       e:      	jsr16	r3
# CHECK-NEXT:       10:      	lrw16	r3, 0x20 <$d.0+0x8>
# CHECK-NEXT:       12:      	str32.w	r4, (r0, r3 << 0)
# CHECK-NEXT:       16:      	bkpt

# CHECK:            18:	00 00 00 00	.word	0x00000000
# CHECK-NEXT:           00000018:  R_CKCORE_TLS_LDM32	xxx+0x18
# CHECK-NEXT:       1c:	00 00 00 00	.word	0x00000000
# CHECK-NEXT:           0000001c:  R_CKCORE_PLT32	__tls_get_addr
# CHECK-NEXT:       20:	00 00 00 00	.word	0x00000000
# CHECK-NEXT:           00000020:  R_CKCORE_TLS_LDO32	xxx
